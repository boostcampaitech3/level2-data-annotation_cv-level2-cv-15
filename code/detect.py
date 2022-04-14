import numpy as np
import torch
import lanms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.geometric.resize import LongestMaxSize

from dataset import get_rotate_mat


def is_valid_poly(res, score_shape, scale):
    '''check if the poly in image scope
    Input:
        res        : restored poly in original image
        score_shape: score map shape
        scale      : feature map -> image
    Output:
        True if valid
    '''
    cnt = 0
    for i in range(res.shape[1]):
        if (res[0, i] < 0 or res[0, i] >= score_shape[1] * scale or res[1, i] < 0 or
            res[1, i] >= score_shape[0] * scale):
            cnt += 1
    return cnt <= 1


def restore_polys(valid_pos, valid_geo, score_shape, scale=4):
    '''restore polys from feature maps in given positions
    Input:
        valid_pos  : potential text positions <numpy.ndarray, (n,2)>
        valid_geo  : geometry in valid_pos <numpy.ndarray, (5,n)>
        score_shape: shape of score map
        scale      : image / feature map
    Output:
        restored polys <numpy.ndarray, (n,8)>, index
    '''
    polys = []
    index = []
    valid_pos *= scale
    d = valid_geo[:4, :] # 4 x N
    angle = valid_geo[4, :] # N,

    for i in range(valid_pos.shape[0]):
        x = valid_pos[i, 0]
        y = valid_pos[i, 1]
        y_min = y - d[0, i]
        y_max = y + d[1, i]
        x_min = x - d[2, i]
        x_max = x + d[3, i]
        rotate_mat = get_rotate_mat(-angle[i])

        temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
        temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
        coordidates = np.concatenate((temp_x, temp_y), axis=0)
        res = np.dot(rotate_mat, coordidates)
        res[0, :] += x
        res[1, :] += y

        if is_valid_poly(res, score_shape, scale):
            index.append(i)
            polys.append([res[0, 0], res[1, 0], res[0, 1], res[1, 1], res[0, 2], res[1, 2],
                          res[0, 3], res[1, 3]])
    return np.array(polys), index


def get_bboxes(score, geo, score_thresh=0.9, nms_thresh=0.2):
    '''get boxes from feature map
    Input:
        score       : score map from model <numpy.ndarray, (1,row,col)>
        geo         : geo map from model <numpy.ndarray, (5,row,col)>
        score_thresh: threshold to segment score map
        nms_thresh  : threshold in nms
    Output:
        boxes       : final polys <numpy.ndarray, (n,9)>
    '''
    score = score[0, :, :]
    xy_text = np.argwhere(score > score_thresh) # n x 2, format is [r, c]
    if xy_text.size == 0:
        return None

    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    valid_pos = xy_text[:, ::-1].copy()  # n x 2, [x, y]
    valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]]  # 5 x n
    polys_restored, index = restore_polys(valid_pos, valid_geo, score.shape)
    if polys_restored.size == 0:
        return None

    boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = polys_restored
    boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
    return boxes


def detect(model, images, input_size):
    prep_fn = A.Compose([
        LongestMaxSize(input_size), A.PadIfNeeded(min_height=input_size, min_width=input_size,
                                                  position=A.PadIfNeeded.PositionType.TOP_LEFT),
        A.Normalize(), ToTensorV2()])
    device = list(model.parameters())[0].device

    batch, orig_sizes = [], []
    for image in images:
        orig_sizes.append(image.shape[:2])
        batch.append(prep_fn(image=image)['image'])
    batch = torch.stack(batch, dim=0).to(device)

    with torch.no_grad():
        score_maps, geo_maps = model(batch)
    score_maps, geo_maps = score_maps.cpu().numpy(), geo_maps.cpu().numpy()

    by_sample_bboxes = []
    for score_map, geo_map, orig_size in zip(score_maps, geo_maps, orig_sizes):
        map_margin = int(abs(orig_size[0] - orig_size[1]) * 0.25 * input_size / max(orig_size))
        if orig_size[0] > orig_size[1]:
            score_map, geo_map = score_map[:, :, :-map_margin], geo_map[:, :, :-map_margin]
        else:
            score_map, geo_map = score_map[:, :-map_margin, :], geo_map[:, :-map_margin, :]

        bboxes = get_bboxes(score_map, geo_map)
        if bboxes is None:
            bboxes = np.zeros((0, 4, 2), dtype=np.float32)
        else:
            bboxes = bboxes[:, :8].reshape(-1, 4, 2)
            bboxes *= max(orig_size) / input_size

        by_sample_bboxes.append(bboxes)

    return by_sample_bboxes
