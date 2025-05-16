import numpy as np
from numpy.typing import NDArray
import cv2
from optree import treespec_namedtuple

def scale_transform(sx, sy):
    pass

def translation_transform(tx, ty):
    pass

def pad_transform(pl, pr, pt, pb):
    pass

def crop_transform(crop_roi):
    pass

def pad_affine_matrix(src_mat):
    pass


class ImageTransformation:
    def __init__(self, transformation=None, dst_size=None):
        self.transformation = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) if transformation is None else transformation
        self.dst_size = dst_size

    def apply_scale(self, sx, sy):
        return self

    def apply_pad(self, px, py):
        return self

    def apply_crop(self, crop_roi):
        return self

    def transform_image(self, img: NDArray):
        return self

    def transform_points(self, points: NDArray):
        pass

    def transform_bbox(self, bbox: NDArray):
        pass

    def transform_mask(self, mask: NDArray):
        # interp_mode = cv2.INTER_NEAREST if exact_interpolation else cv2.INTER_CUBIC
        # img = cv2.resize(img, intermediate_size, interpolation=interp_mode)
        pass #return cv2.warpAffine(img, M, )
