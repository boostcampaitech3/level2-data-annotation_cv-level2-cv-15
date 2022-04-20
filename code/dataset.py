import os.path as osp
import math
import json
from PIL import Image

import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.geometric.resize import LongestMaxSize, SmallestMaxSize
from torch.utils.data import Dataset
from shapely.geometry import Polygon
from tqdm import tqdm
from augmentation import ComposedTransformation, CropMethod_1


def cal_distance(x1, y1, x2, y2):
    '''calculate the Euclidean distance'''
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def move_points(vertices, index1, index2, r, coef):
    '''move the two points to shrink edge
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        index1  : offset of point1
        index2  : offset of point2
        r       : [r1, r2, r3, r4] in paper
        coef    : shrink ratio in paper
    Output:
        vertices: vertices where one edge has been shinked
    '''
    index1 = index1 % 4
    index2 = index2 % 4
    x1_index = index1 * 2 + 0
    y1_index = index1 * 2 + 1
    x2_index = index2 * 2 + 0
    y2_index = index2 * 2 + 1

    r1 = r[index1]
    r2 = r[index2]
    length_x = vertices[x1_index] - vertices[x2_index]
    length_y = vertices[y1_index] - vertices[y2_index]
    length = cal_distance(vertices[x1_index], vertices[y1_index], vertices[x2_index], vertices[y2_index])
    if length > 1:
        ratio = (r1 * coef) / length
        vertices[x1_index] += ratio * (-length_x)
        vertices[y1_index] += ratio * (-length_y)
        ratio = (r2 * coef) / length
        vertices[x2_index] += ratio * length_x
        vertices[y2_index] += ratio * length_y
    return vertices


def shrink_poly(vertices, coef=0.3):
    '''shrink the text region
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        coef    : shrink ratio in paper
    Output:
        v       : vertices of shrinked text region <numpy.ndarray, (8,)>
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    r1 = min(cal_distance(x1,y1,x2,y2), cal_distance(x1,y1,x4,y4))
    r2 = min(cal_distance(x2,y2,x1,y1), cal_distance(x2,y2,x3,y3))
    r3 = min(cal_distance(x3,y3,x2,y2), cal_distance(x3,y3,x4,y4))
    r4 = min(cal_distance(x4,y4,x1,y1), cal_distance(x4,y4,x3,y3))
    r = [r1, r2, r3, r4]

    # obtain offset to perform move_points() automatically
    if cal_distance(x1,y1,x2,y2) + cal_distance(x3,y3,x4,y4) > \
       cal_distance(x2,y2,x3,y3) + cal_distance(x1,y1,x4,y4):
        offset = 0 # two longer edges are (x1y1-x2y2) & (x3y3-x4y4)
    else:
        offset = 1 # two longer edges are (x2y2-x3y3) & (x4y4-x1y1)

    v = vertices.copy()
    v = move_points(v, 0 + offset, 1 + offset, r, coef)
    v = move_points(v, 2 + offset, 3 + offset, r, coef)
    v = move_points(v, 1 + offset, 2 + offset, r, coef)
    v = move_points(v, 3 + offset, 4 + offset, r, coef)
    return v


def get_rotate_mat(theta):
    '''positive theta value means rotate clockwise'''
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def rotate_vertices(vertices, theta, anchor=None):
    '''rotate vertices around anchor
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        theta   : angle in radian measure
        anchor  : fixed position during rotation
    Output:
        rotated vertices <numpy.ndarray, (8,)>
    '''
    v = vertices.reshape((4,2)).T
    if anchor is None:
        anchor = v[:,:1]
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)


def get_boundary(vertices):
    '''get the tight boundary around given vertices
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the boundary
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return x_min, x_max, y_min, y_max


def cal_error(vertices):
    '''default orientation is x1y1 : left-top, x2y2 : right-top, x3y3 : right-bot, x4y4 : left-bot
    calculate the difference between the vertices orientation and default orientation
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        err     : difference measure
    '''
    x_min, x_max, y_min, y_max = get_boundary(vertices)
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    err = cal_distance(x1, y1, x_min, y_min) + cal_distance(x2, y2, x_max, y_min) + \
          cal_distance(x3, y3, x_max, y_max) + cal_distance(x4, y4, x_min, y_max)
    return err


def find_min_rect_angle(vertices):
    '''find the best angle to rotate poly and obtain min rectangle
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the best angle <radian measure>
    '''
    angle_interval = 1
    angle_list = list(range(-90, 90, angle_interval))
    area_list = []
    for theta in angle_list:
        rotated = rotate_vertices(vertices, theta / 180 * math.pi)
        x1, y1, x2, y2, x3, y3, x4, y4 = rotated
        temp_area = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * \
                    (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
        area_list.append(temp_area)

    sorted_area_index = sorted(list(range(len(area_list))), key=lambda k: area_list[k])
    min_error = float('inf')
    best_index = -1
    rank_num = 10
    # find the best angle with correct orientation
    for index in sorted_area_index[:rank_num]:
        rotated = rotate_vertices(vertices, angle_list[index] / 180 * math.pi)
        temp_error = cal_error(rotated)
        if temp_error < min_error:
            min_error = temp_error
            best_index = index
    return angle_list[best_index] / 180 * math.pi


def is_cross_text(start_loc, length, vertices):
    '''check if the crop image crosses text regions
    Input:
        start_loc: left-top position
        length   : length of crop image
        vertices : vertices of text regions <numpy.ndarray, (n,8)>
    Output:
        True if crop image crosses text region
    '''
    if vertices.size == 0:
        return False
    start_w, start_h = start_loc
    a = np.array([start_w, start_h, start_w + length, start_h, start_w + length, start_h + length,
                  start_w, start_h + length]).reshape((4, 2))
    p1 = Polygon(a).convex_hull
    for vertice in vertices:
        p2 = Polygon(vertice.reshape((4, 2))).convex_hull
        inter = p1.intersection(p2).area
        if 0.01 <= inter / p2.area <= 0.99:
            return True
    return False


def crop_img(img, vertices, labels, length):
    '''crop img patches to obtain batch and augment
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        labels      : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
        length      : length of cropped image region
    Output:
        region      : cropped image region
        new_vertices: new vertices in cropped region
    '''
    h, w = img.height, img.width
    # confirm the shortest side of image >= length
    if h >= w and w < length:
        img = img.resize((length, int(h * length / w)), Image.BILINEAR)
    elif h < w and h < length:
        img = img.resize((int(w * length / h), length), Image.BILINEAR)
    ratio_w = img.width / w
    ratio_h = img.height / h
    assert(ratio_w >= 1 and ratio_h >= 1)

    new_vertices = np.zeros(vertices.shape)
    if vertices.size > 0:
        new_vertices[:,[0,2,4,6]] = vertices[:,[0,2,4,6]] * ratio_w
        new_vertices[:,[1,3,5,7]] = vertices[:,[1,3,5,7]] * ratio_h

    # find random position
    remain_h = img.height - length
    remain_w = img.width - length
    flag = True
    cnt = 0
    while flag and cnt < 1000:
        cnt += 1
        start_w = int(np.random.rand() * remain_w)
        start_h = int(np.random.rand() * remain_h)
        flag = is_cross_text([start_w, start_h], length, new_vertices[labels==1,:])
    box = (start_w, start_h, start_w + length, start_h + length)
    region = img.crop(box)
    if new_vertices.size == 0:
        return region, new_vertices

    new_vertices[:,[0,2,4,6]] -= start_w
    new_vertices[:,[1,3,5,7]] -= start_h
    return region, new_vertices


def rotate_all_pixels(rotate_mat, anchor_x, anchor_y, length):
    '''get rotated locations of all pixels for next stages
    Input:
        rotate_mat: rotatation matrix
        anchor_x  : fixed x position
        anchor_y  : fixed y position
        length    : length of image
    Output:
        rotated_x : rotated x positions <numpy.ndarray, (length,length)>
        rotated_y : rotated y positions <numpy.ndarray, (length,length)>
    '''
    x = np.arange(length)
    y = np.arange(length)
    x, y = np.meshgrid(x, y)
    x_lin = x.reshape((1, x.size))
    y_lin = y.reshape((1, x.size))
    coord_mat = np.concatenate((x_lin, y_lin), 0)
    rotated_coord = np.dot(rotate_mat, coord_mat - np.array([[anchor_x], [anchor_y]])) + \
                                                   np.array([[anchor_x], [anchor_y]])
    rotated_x = rotated_coord[0, :].reshape(x.shape)
    rotated_y = rotated_coord[1, :].reshape(y.shape)
    return rotated_x, rotated_y


def resize_img(img, vertices, size):
    h, w = img.height, img.width
    ratio = size / max(h, w)
    if w > h:
        img = img.resize((size, int(h * ratio)), Image.BILINEAR)
    else:
        img = img.resize((int(w * ratio), size), Image.BILINEAR)
    new_vertices = vertices * ratio
    return img, new_vertices


def adjust_height(img, vertices, ratio=0.2):
    '''adjust height of image to aug data
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        ratio       : height changes in [0.8, 1.2]
    Output:
        img         : adjusted PIL Image
        new_vertices: adjusted vertices
    '''
    ratio_h = 1 + ratio * (np.random.rand() * 2 - 1)
    old_h = img.height
    new_h = int(np.around(old_h * ratio_h))
    img = img.resize((img.width, new_h), Image.BILINEAR)

    new_vertices = vertices.copy()
    if vertices.size > 0:
        new_vertices[:,[1,3,5,7]] = vertices[:,[1,3,5,7]] * (new_h / old_h)
    return img, new_vertices


def rotate_img(img, vertices, angle_range=10):
    '''rotate image [-10, 10] degree to aug data
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        angle_range : rotate range
    Output:
        img         : rotated PIL Image
        new_vertices: rotated vertices
    '''
    center_x = (img.width - 1) / 2
    center_y = (img.height - 1) / 2
    angle = angle_range * (np.random.rand() * 2 - 1)
    img = img.rotate(angle, Image.BILINEAR)
    new_vertices = np.zeros(vertices.shape)
    for i, vertice in enumerate(vertices):
        new_vertices[i,:] = rotate_vertices(vertice, -angle / 180 * math.pi, np.array([[center_x],[center_y]]))
    return img, new_vertices


def generate_roi_mask(image, vertices, labels):
    mask = np.ones(image.shape[:2], dtype=np.float32)
    ignored_polys = []
    for vertice, label in zip(vertices, labels):
        if label == 0:
            ignored_polys.append(np.around(vertice.reshape((4, 2))).astype(np.int32))
    cv2.fillPoly(mask, ignored_polys, 0)
    return mask


def filter_vertices(vertices, labels, ignore_under=0, drop_under=0):
    if drop_under == 0 and ignore_under == 0:
        return vertices, labels

    new_vertices, new_labels = vertices.copy(), labels.copy()

    areas = np.array([Polygon(v.reshape((4, 2))).convex_hull.area for v in vertices])
    labels[areas < ignore_under] = 0

    if drop_under > 0:
        passed = areas >= drop_under
        new_vertices, new_labels = new_vertices[passed], new_labels[passed]

    return new_vertices, new_labels

class SceneTextDataset2(Dataset):
    def __init__(self, root_dir, split='train', image_size=1024, crop_size=512, color_jitter=True,
                 normalize=True, augmentation=True):
        with open(osp.join(root_dir, 'ufo/{}.json'.format(split)), 'r') as f:
            anno = json.load(f)

        self.anno = anno
        self.image_fnames = sorted(anno['images'].keys())
        self.image_dir = osp.join(root_dir, 'images')

        self.augmentation = augmentation
        self.image_size, self.crop_size = image_size, crop_size
        self.color_jitter, self.normalize = color_jitter, normalize

        self.images = []
        self.vertices = []
        self.labels = []


        self.transforms = []
        for crop_size in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
            self.transforms.append(ComposedTransformation(
                crop_aspect_ratio=1.0, crop_size=(crop_size, crop_size),
                hflip=False, vflip=False, random_translate=True,
                resize_to=512,
                min_image_overlap=0.9, min_bbox_overlap=1.0, min_bbox_count=1, allow_partial_occurrence=False,
                max_random_trials=100,
                brightness=0.5, contrast=0.5, saturation=0.25, hue=0.25,
                normalize=True, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), to_tensor=False
            ))

        self.transform2 = CropMethod_1()

    def load_image(self):
        for image_fname in tqdm(self.image_fnames):
            image_fpath = osp.join(self.image_dir, image_fname)
            image = cv2.imread(image_fpath)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

            vertices, labels = [], []
            for word_info in self.anno['images'][image_fname]['words'].values():
                vertices.append(np.array(word_info['points']).flatten())
                labels.append(int(not word_info['illegibility']))
            vertices, labels = np.array(vertices, dtype=np.float32), np.array(labels, dtype=np.int64)

            vertices, labels = filter_vertices(vertices, labels, ignore_under=10, drop_under=1)
            image, vertices = resize_img(image, vertices, self.image_size)

            self.images.append(image)
            self.vertices.append(vertices)
            self.labels.append(labels)

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, idx):
        image = self.images[idx]
        vertices = self.vertices[idx]
        labels = self.labels[idx]


        image, vertices = adjust_height(image, vertices)
        image, vertices = rotate_img(image, vertices, angle_range=10)

        trans = random.choice([0,1])
        if trans==0:
            transform = random.choice(self.transforms)
            transformed = transform(image=image, word_bboxes=vertices.reshape(-1,4,2))
            image = transformed['image']
            vertices = transformed['word_bboxes']
        else:
            image, vertices, labels = self.transform2(image, vertices, labels)


        word_bboxes = np.reshape(vertices, (-1, 4, 2))
        roi_mask = generate_roi_mask(image, vertices, labels)

        return image, word_bboxes, roi_mask

class SceneTextDataset(Dataset):
    def __init__(self, root_dir, split='train', image_size=1024, crop_size=512, color_jitter=True,
                 normalize=True, valid=False):
        with open(osp.join(root_dir, 'ufo/{}.json'.format(split)), 'r') as f:
            anno = json.load(f)

        self.anno = anno
        self.image_fnames = sorted(anno['images'].keys())
        self.image_dir = osp.join(root_dir, 'images')

        self.image_size, self.crop_size = image_size, crop_size
        self.color_jitter, self.normalize = color_jitter, normalize
        self.valid = valid

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, idx):
        image_fname = self.image_fnames[idx]
        image_fpath = osp.join(self.image_dir, image_fname)

        vertices, labels = [], []
        for word_info in self.anno['images'][image_fname]['words'].values():
            vertices.append(np.array(word_info['points']).flatten())
            labels.append(int(not word_info['illegibility']))
        vertices, labels = np.array(vertices, dtype=np.float32), np.array(labels, dtype=np.int64)

        vertices, labels = filter_vertices(vertices, labels, ignore_under=10, drop_under=1)

        image = Image.open(image_fpath)
        image, vertices = resize_img(image, vertices, self.image_size)
        image, vertices = adjust_height(image, vertices)
        image, vertices = rotate_img(image, vertices)
        image, vertices = crop_img(image, vertices, labels, self.crop_size)

        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image)

        funcs = []
        if self.color_jitter:
            funcs.append(A.ColorJitter(0.5, 0.5, 0.5, 0.25))
        if self.normalize:
            funcs.append(A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

        if not self.valid:
            transform = A.Compose(funcs)

            image = transform(image=image)['image']
        word_bboxes = np.reshape(vertices, (-1, 4, 2))
        roi_mask = generate_roi_mask(image, vertices, labels)

        return image, word_bboxes, roi_mask


class ValidSceneTextDataset(SceneTextDataset2):
    def __init__(self, root_dir, split='valid', image_size=1024, crop_size=None, color_jitter=False,
                 normalize=True, map_scale=0.25, to_tensor=True):
        super().__init__(root_dir, split, image_size, crop_size, color_jitter, False, None)

        self.orig_sizes = []
        self.transcriptions = []
        self.map_scale = map_scale
        self.to_tensor = to_tensor
        self.prep_fn = A.Compose([
            LongestMaxSize(image_size), A.PadIfNeeded(min_height=image_size, min_width=image_size,
                                                    position=A.PadIfNeeded.PositionType.TOP_LEFT),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2()])
    
    def load_image(self):
        for image_fname in tqdm(self.image_fnames):
            image_fpath = osp.join(self.image_dir, image_fname)
            image = cv2.imread(image_fpath)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            self.orig_sizes.append(image.shape[:2])

            vertices, labels, transcriptions = [], [], []
            for word_info in self.anno['images'][image_fname]['words'].values():
                vertices.append(np.array(word_info['points']).flatten())
                labels.append(int(not word_info['illegibility']))
                transcriptions.append(word_info['transcription'])
            vertices, labels = np.array(vertices, dtype=np.float32), np.array(labels, dtype=np.int64)

            vertices, labels = filter_vertices(vertices, labels, ignore_under=10, drop_under=1)
            image, vertices = resize_img(image, vertices, self.image_size)

            self.images.append(self.prep_fn(image=image)['image'])
            self.vertices.append(vertices.reshape(-1,4,2))
            self.labels.append(labels)
            self.transcriptions.append(transcriptions)

    def __getitem__(self, idx):
        image = self.images[idx]
        vertices = self.vertices[idx]
        labels = self.labels[idx]
        orig_size = self.orig_sizes[idx]
        transcriptions = self.transcriptions[idx]

        image = image.permute(1, 2, 0)
        roi_mask = generate_roi_mask(image, vertices, labels)
        score_map, geo_map = generate_score_geo_maps(image, vertices, map_scale=self.map_scale)

        mask_size = int(image.shape[0] * self.map_scale), int(image.shape[1] * self.map_scale)
        
        roi_mask = cv2.resize(roi_mask, dsize=mask_size)
        if roi_mask.ndim == 2:
            roi_mask = np.expand_dims(roi_mask, axis=2)

        if self.to_tensor:
            image = torch.Tensor(image).permute(2, 0, 1)
            score_map = torch.Tensor(score_map).permute(2, 0, 1)
            geo_map = torch.Tensor(geo_map).permute(2, 0, 1)
            roi_mask = torch.Tensor(roi_mask).permute(2, 0, 1)

        return image, score_map, geo_map, roi_mask, vertices, orig_size, labels, transcriptions, self.image_fnames[idx]

    def collate_fn(batchs):
        imgs = []
        score_maps = []
        geo_maps = []
        roi_masks = []
        vertices = []
        orig_sizes = []
        labels = []
        transcriptions = []
        fnames = []
        for data in batchs:
            imgs.append(data[0])
            score_maps.append(data[1])
            geo_maps.append(data[2])
            roi_masks.append(data[3])
            vertices.append(data[4])
            orig_sizes.append(data[5])
            labels.append(data[6])
            transcriptions.append(data[7])
            fnames.append(data[8])
        
        imgs = torch.stack(imgs, dim=0)
        score_maps = torch.stack(score_maps, dim=0)
        geo_maps = torch.stack(geo_maps, dim=0)
        roi_masks = torch.stack(roi_masks, dim=0)

        return imgs, score_maps, geo_maps, roi_masks, vertices, orig_sizes, labels, transcriptions, fnames