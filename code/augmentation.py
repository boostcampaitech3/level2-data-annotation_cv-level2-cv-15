from os import EX_CANTCREAT
import numpy as np
import cv2
import albumentations as A

import numpy.random as npr
from albumentations.pytorch import ToTensorV2
from shapely.geometry import Polygon


def transform_by_matrix(matrix, image=None, oh=None, ow=None, word_bboxes=[],
                        by_word_char_bboxes=[], masks=[], inverse=False):
    """
    Args:
        matrix (ndarray): (3, 3) shaped transformation matrix.
        image (ndarray): (H, W, C) shaped ndarray.
        oh (int): Output height.
        ow (int): Output width.
        word_bboxes (List[ndarray]): List of (N, 2) shaped ndarrays.
        by_word_char_bboxes (List[ndarray]): Lists of (N, 4, 2) shaped ndarrays.
        masks (List[ndarray]): List of (H, W) shaped ndarray the same size as the image.
        inverse (bool): Whether to apply inverse transformation.
    """
    if image is not None or masks is not None:
        assert oh is not None and ow is not None

    output_dict = dict()

    if inverse:
        matrix = np.linalg.pinv(matrix)

    if image is not None:
        output_dict['image'] = cv2.warpPerspective(image, matrix, dsize=(ow, oh))

    if word_bboxes is None:
        output_dict['word_bboxes'] = None
    elif len(word_bboxes) > 0:
        num_points = list(map(len, word_bboxes))
        points = np.concatenate([np.reshape(bbox, (-1, 2)) for bbox in word_bboxes])  # (N, 2)
        points = cv2.perspectiveTransform(
            np.reshape(points, (1, -1, 2)).astype(np.float32), matrix).reshape(-1, 2)  # (N, 2)
        output_dict['word_bboxes'] = [
            points[i:i + n] for i, n in zip(np.cumsum([0] + num_points)[:-1], num_points)]
    else:
        output_dict['word_bboxes'] = []

    if by_word_char_bboxes is None:
        output_dict['by_word_char_bboxes'] = None
    elif len(by_word_char_bboxes) > 0:
        word_lens = list(map(len, by_word_char_bboxes))
        points = np.concatenate([np.reshape(bboxes, (-1, 2)) for bboxes in by_word_char_bboxes])  # (N, 2)
        points = cv2.perspectiveTransform(
            np.reshape(points, (1, -1, 2)).astype(np.float32), matrix).reshape(-1, 4, 2)  # (N, 4, 2)
        output_dict['by_word_char_bboxes'] = [
            points[i:i + n] for i, n in zip(np.cumsum([0] + word_lens)[:-1], word_lens)]
    else:
        output_dict['by_word_char_bboxes'] = []

    if masks is None:
        output_dict['masks'] = None
    else:
        output_dict['masks'] = [cv2.warpPerspective(mask, matrix, dsize=(ow, oh)) for mask in masks]

    return output_dict


class GeoTransformation:
    """
    Args:
    """
    def __init__(
        self,
        rotate_anchors=None, rotate_range=None,
        crop_aspect_ratio=None, crop_size=1.0, crop_size_by='longest', hflip=False, vflip=False,
        random_translate=False, min_image_overlap=0, min_bbox_overlap=0, min_bbox_count=0,
        allow_partial_occurrence=True,
        resize_to=None, keep_aspect_ratio=False, resize_based_on='longest', max_random_trials=100
    ):
        if rotate_anchors is None:
            self.rotate_anchors = None
        elif type(rotate_anchors) in [int, float]:
            self.rotate_anchors = [rotate_anchors]
        else:
            self.rotate_anchors = list(rotate_anchors)

        if rotate_range is None:
            self.rotate_range = None
        elif type(rotate_range) in [int, float]:
            assert rotate_range >= 0
            self.rotate_range = (-rotate_range, rotate_range)
        elif len(rotate_range) == 2:
            assert rotate_range[0] <= rotate_range[1]
            self.rotate_range = tuple(rotate_range)
        else:
            raise TypeError

        if crop_aspect_ratio is None:
            self.crop_aspect_ratio = None
        elif type(crop_aspect_ratio) in [int, float]:
            self.crop_aspect_ratio = float(crop_aspect_ratio)
        elif len(crop_aspect_ratio) == 2:
            self.crop_aspect_ratio = tuple(crop_aspect_ratio)
        else:
            raise TypeError

        if type(crop_size) in [int, float]:
            self.crop_size = crop_size
        elif len(crop_size) == 2:
            assert type(crop_size[0]) == type(crop_size[1])
            self.crop_size = tuple(crop_size)
        else:
            raise TypeError

        assert crop_size_by in ['width', 'height', 'longest']
        self.crop_size_by = crop_size_by

        self.hflip, self.vflip = hflip, vflip

        self.random_translate = random_translate

        self.min_image_overlap = max(min_image_overlap or 0, 0)
        self.min_bbox_overlap = max(min_bbox_overlap or 0, 0)
        self.min_bbox_count = max(min_bbox_count or 0, 0)
        self.allow_partial_occurrence = allow_partial_occurrence

        self.max_random_trials = max_random_trials

        if resize_to is None:
            self.resize_to = resize_to
        elif type(resize_to) in [int, float]:
            if not keep_aspect_ratio:
                self.resize_to = (resize_to, resize_to)
            else:
                self.resize_to = resize_to
        elif len(resize_to) == 2:
            assert not keep_aspect_ratio
            assert type(resize_to[0]) == type(resize_to[1])
            self.resize_to = tuple(resize_to)
        assert resize_based_on in ['width', 'height', 'longest']
        self.keep_aspect_ratio, self.resize_based_on = keep_aspect_ratio, resize_based_on

    def __call__(self, image, word_bboxes=[], by_word_char_bboxes=[], masks=[]):
        return self.crop_rotate_resize(image, word_bboxes=word_bboxes,
                                       by_word_char_bboxes=by_word_char_bboxes, masks=masks)

    def _get_theta(self):
        if self.rotate_anchors is None:
            theta = 0
        else:
            theta = npr.choice(self.rotate_anchors)
        if self.rotate_range is not None:
            theta += npr.uniform(*self.rotate_range)

        return theta

    def _get_patch_size(self, ih, iw):
        if (self.crop_aspect_ratio is None and isinstance(self.crop_size, float) and
            self.crop_size == 1.0):
            return ih, iw

        if self.crop_aspect_ratio is None:
            aspect_ratio = iw / ih
        elif isinstance(self.crop_aspect_ratio, float):
            aspect_ratio = self.crop_aspect_ratio
        else:
            aspect_ratio = np.exp(npr.uniform(*np.log(self.crop_aspect_ratio)))

        if isinstance(self.crop_size, tuple):
            if isinstance(self.crop_size[0], int):
                crop_size = npr.randint(self.crop_size[0], self.crop_size[1])
            elif self.crop_size[0]:
                crop_size = np.exp(npr.uniform(*np.log(self.crop_size)))
        else:
            crop_size = self.crop_size

        if self.crop_size_by == 'longest' and iw >= ih or self.crop_size_by == 'width':
            if isinstance(crop_size, int):
                pw = crop_size
                ph = int(pw / aspect_ratio)
            else:
                pw = int(iw * crop_size)
                ph = int(iw * crop_size / aspect_ratio)
        else:
            if isinstance(crop_size, int):
                ph = crop_size
                pw = int(ph * aspect_ratio)
            else:
                ph = int(ih * crop_size)
                pw = int(ih * crop_size * aspect_ratio)

        return ph, pw

    def _get_patch_quad(self, theta, ph, pw):
        cos, sin = np.cos(theta * np.pi / 180), np.sin(theta * np.pi / 180)
        hpx, hpy = 0.5 * pw, 0.5 * ph  # half patch size
        quad = np.array([[-hpx, -hpy], [hpx, -hpy], [hpx, hpy], [-hpx, hpy]], dtype=np.float32)
        rotation_m = np.array([[cos, sin], [-sin, cos]], dtype=np.float32)
        quad = np.matmul(quad, rotation_m)  # patch quadrilateral in relative coords

        return quad

    def _get_located_patch_quad(self, ih, iw, patch_quad_rel, bboxes=[]):
        image_poly = Polygon([[0, 0], [iw, 0], [iw, ih], [0, ih]])
        if self.min_image_overlap is not None:
            center_patch_poly = Polygon(
                np.array([0.5 * ih, 0.5 * iw], dtype=np.float32) + patch_quad_rel)
            max_available_overlap = (
                image_poly.intersection(center_patch_poly).area / center_patch_poly.area)
            min_image_overlap = min(self.min_image_overlap, max_available_overlap)
        else:
            min_image_overlap = None

        if self.min_bbox_count > 0:
            min_bbox_count = min(self.min_bbox_count, len(bboxes))
        else:
            min_bbox_count = 0

        cx_margin, cy_margin = np.sort(patch_quad_rel[:, 0])[2], np.sort(patch_quad_rel[:, 1])[2]

        found_randomly = False
        for trial_idx in range(self.max_random_trials):
            cx, cy = npr.uniform(cx_margin, iw - cx_margin), npr.uniform(cy_margin, ih - cy_margin)
            patch_quad = np.array([cx, cy], dtype=np.float32) + patch_quad_rel
            patch_poly = Polygon(patch_quad)
            
            if min_image_overlap:
                image_overlap = patch_poly.intersection(image_poly).area / patch_poly.area
                # 이미지에서 벗어난 영역이 특정 비율보다 높으면 탈락
                if image_overlap < min_image_overlap:
                    continue

            if (self.min_bbox_count or not self.allow_partial_occurrence) and self.min_bbox_overlap:
                bbox_count = 0
                partial_occurrence = False
                
                for bbox in bboxes:
                    bbox_poly = None
                    bbox_poly = Polygon(bbox)
                    
                    if bbox_poly.area <= 0:
                        continue

                    try:
                        bbox_overlap = bbox_poly.intersection(patch_poly).area / bbox_poly.area
                    except Exception as ex:
                        break
                        # print(f"bbox_poly.area = {bbox_poly.area}")
                        # print(f"patch_poly = {patch_poly}")
                        # print(f"bbox_poly = {bbox_poly}")
                        # print(f"patch_quad = {patch_quad}")
                        # print(f"patch_quad_rel = {patch_quad_rel}")
                        # print(f"np.array([cx, cy], dtype=np.float32) = {np.array([cx, cy], dtype=np.float32)}")
                        # print(f"Error = {ex}")

                    if bbox_overlap >= self.min_bbox_overlap:
                        bbox_count += 1
                    if (not self.allow_partial_occurrence and bbox_overlap > 0 and
                        bbox_overlap < self.min_bbox_overlap):
                        partial_occurrence = True
                        break
                
                # 부분적으로 나타나는 개체가 있으면 탈락
                if partial_occurrence:
                    continue
                # 온전히 포함하는 개체가 특정 개수 미만이면 탈락
                elif self.min_bbox_count and bbox_count < self.min_bbox_count:
                    continue

            found_randomly = True
            break

        if found_randomly:
            return patch_quad, trial_idx + 1
        else:
            return None, trial_idx + 1

    def crop_rotate_resize(self, image, word_bboxes=[], by_word_char_bboxes=[], masks=[]):
        """
        Args:
            image (ndarray): (H, W, C) shaped ndarray.
            masks (List[ndarray]): List of (H, W) shaped ndarray the same size as the image.
        """
        ih, iw = image.shape[:2]  # image height and width

        theta = self._get_theta()
        ph, pw = self._get_patch_size(ih, iw)

        patch_quad_rel = self._get_patch_quad(theta, ph, pw)

        if not self.random_translate:
            cx, cy = 0.5 * iw, 0.5 * ih
            patch_quad = np.array([cx, cy], dtype=np.float32) + patch_quad_rel
            num_trials = 0
        else:
            patch_quad, num_trials = self._get_located_patch_quad(ih, iw, patch_quad_rel,
                                                                  bboxes=word_bboxes)

        vflip, hflip = self.vflip and npr.randint(2) > 0, self.hflip and npr.randint(2) > 0

        if self.resize_to is None:
            oh, ow = ih, iw
        elif self.keep_aspect_ratio:  # `resize_to`: Union[int, float]
            if self.resize_based_on == 'height' or self.resize_based_on == 'longest' and ih >= iw:
                oh = ih * self.resize_to if isinstance(self.resize_to, float) else self.resize_to
                ow = int(oh * iw / ih)
            else:
                ow = iw * self.resize_to if isinstance(self.resize_to, float) else self.resize_to
                oh = int(ow * ih / iw)
        elif isinstance(self.resize_to[0], float):  # `resize_to`: tuple[float, float]
            oh, ow = ih * self.resize_to[0], iw * self.resize_to[1]
        else:  # `resize_to`: tuple[int, int]
            oh, ow = self.resize_to

        if theta == 0 and (ph, pw) == (ih, iw) and (oh, ow) == (ih, iw) and not (hflip or vflip):
            M = None
            transformed = dict(image=image, word_bboxes=word_bboxes,
                               by_word_char_bboxes=by_word_char_bboxes, masks=masks)
        else:
            dst = np.array([[0, 0], [ow, 0], [ow, oh], [0, oh]], dtype=np.float32)
            if patch_quad is not None:
                src = patch_quad
            else:
                if ow / oh >= iw / ih:
                    pad = int(ow * ih / oh) - iw
                    off = npr.randint(pad + 1)  # offset
                    src = np.array(
                        [[-off, 0], [iw + pad - off, 0], [iw + pad - off, ih], [-off, ih]],
                        dtype=np.float32)
                else:
                    pad = int(oh * iw / ow) - ih
                    off = npr.randint(pad + 1)  # offset
                    src = np.array(
                        [[0, -off], [iw, -off], [iw, ih + pad - off], [0, ih + pad - off]],
                        dtype=np.float32)

            if hflip:
                src = src[[1, 0, 3, 2]]
            if vflip:
                src = src[[3, 2, 1, 0]]

            M = cv2.getPerspectiveTransform(src, dst)
            transformed = transform_by_matrix(M, image=image, oh=oh, ow=ow, word_bboxes=word_bboxes,
                                              by_word_char_bboxes=by_word_char_bboxes, masks=masks)

        found_randomly = self.random_translate and patch_quad is not None

        return dict(found_randomly=found_randomly, num_trials=num_trials, matrix=M, **transformed)


class ComposedTransformation:
    def __init__(
        self,
        rotate_anchors=None, rotate_range=None,
        crop_aspect_ratio=None, crop_size=1.0, crop_size_by='longest', hflip=False, vflip=False,
        random_translate=False, min_image_overlap=0, min_bbox_overlap=0, min_bbox_count=0,
        allow_partial_occurrence=True,
        resize_to=None, keep_aspect_ratio=False, resize_based_on='longest', max_random_trials=100,
        brightness=0, contrast=0, saturation=0, hue=0,
        normalize=False, mean=None, std=None, to_tensor=False
    ):
        self.geo_transform_fn = GeoTransformation(
            rotate_anchors=rotate_anchors, rotate_range=rotate_range,
            crop_aspect_ratio=crop_aspect_ratio, crop_size=crop_size, crop_size_by=crop_size_by,
            hflip=hflip, vflip=vflip, random_translate=random_translate,
            min_image_overlap=min_image_overlap, min_bbox_overlap=min_bbox_overlap,
            min_bbox_count=min_bbox_count, allow_partial_occurrence=allow_partial_occurrence,
            resize_to=resize_to, keep_aspect_ratio=keep_aspect_ratio,
            resize_based_on=resize_based_on, max_random_trials=max_random_trials)

        alb_fns = []
        if brightness > 0 or contrast > 0 or saturation > 0 or hue > 0:
            alb_fns.append(A.ColorJitter(
                brightness=brightness, contrast=contrast, saturation=saturation, hue=hue, p=1))

        if normalize:
            kwargs = dict()
            if mean is not None:
                kwargs['mean'] = mean
            if std is not None:
                kwargs['std'] = std
            alb_fns.append(A.Normalize(**kwargs))

        if to_tensor:
            alb_fns.append(ToTensorV2())

        self.alb_transform_fn = A.Compose(alb_fns)

    def __call__(self, image, word_bboxes=[], by_word_char_bboxes=[], masks=[], height_pad_to=None,
                 width_pad_to=None):
        # TODO Seems that normalization should be performed before padding.

        geo_result = self.geo_transform_fn(image, word_bboxes=word_bboxes,
                                           by_word_char_bboxes=by_word_char_bboxes, masks=masks)

        if height_pad_to is not None or width_pad_to is not None:
            min_height = height_pad_to or geo_result['image'].shape[0]
            min_width = width_pad_to or geo_result['image'].shape[1]
            alb_transform_fn = A.Compose([
                A.PadIfNeeded(min_height=min_height, min_width=min_width,
                              border_mode=cv2.BORDER_CONSTANT,
                              position=A.PadIfNeeded.PositionType.TOP_LEFT),
                self.alb_transform_fn])
        else:
            alb_transform_fn = self.alb_transform_fn
        final_result = alb_transform_fn(image=geo_result['image'])
        del geo_result['image']

        return dict(image=final_result['image'], **geo_result)