from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
import cv2


def invert_affine_matrix(m):
    assert m.shape == (2, 3), "Affine matrix must be 2x3"

    # Extract components
    a = m[:, :2]
    t = m[:, 2]

    # Invert linear part
    a_inv = np.linalg.inv(a)

    # Invert translation
    t_inv = -a_inv @ t

    # Reconstruct inverse matrix
    m_inv = np.hstack([a_inv, t_inv.reshape(2, 1)])
    return m_inv

def create_affine_matrix_2(src_image_size, translation: Tuple[float, float] | NDArray= (0.0, 0.0),
                         scale: float | Tuple[float, float] = 0.0, rotation_angle_deg: float = 0.0):
    w, h = src_image_size
    theta = np.deg2rad(rotation_angle_deg)
    cx, cy = w/2, h/2
    scale_x = scale if isinstance(scale, float) or isinstance(scale, int) else scale[0]
    scale_y = scale if isinstance(scale, float) or isinstance(scale, int) else scale[1]
    tx, ty = translation

    # Rotation + non-uniform scaling
    a = scale_x * np.cos(theta)
    b = -scale_y * np.sin(theta)
    c = scale_x * np.sin(theta)
    d = scale_y * np.cos(theta)

    # Shift origin to center, rotate+scale, shift back, then translate
    m = np.array([[a, b, (1 - a) * cx - b * cy + tx],
                        [c, d, -c * cx + (1 - d) * cy + ty]], dtype=np.float32)
    return m


def affine_to_homography(affine_matrix):
    if affine_matrix.shape != (2, 3):
        raise ValueError("Input must be a 2x3 affine matrix.")

    homography = np.vstack([affine_matrix, [0, 0, 1]])
    return homography


class ImageTransformation:
    def __init__(self, transformation=None, src_size=None, dst_size=None):
        if transformation is None:
            self.transformation = np.eye(3)
            self.transformation_inv = np.eye(3)
        elif transformation.shape == (3,3):
            self.transformation = transformation
            self.transformation_inv = np.linalg.inv(transformation)
        elif transformation.shape == (2,3):
            self.transformation = affine_to_homography(transformation)
            self.transformation_inv = np.linalg.inv(transformation)
        else:
            raise Exception(f"Unknown transformation shape - {transformation.shape}")
        self.dst_size = dst_size
        self.src_size = src_size

    def apply_scale(self, sx, sy):
        return self

    def apply_pad(self, px, py):
        return self

    def apply_crop(self, crop_roi):
        return self

    def transform_image(self, img: NDArray):
        image = cv2.warpPerspective(img, self.transformation, self.dst_size, flags=cv2.INTER_LINEAR)
        return image

    def transform_points(self, points: NDArray, inverse=False):
        m = self.transformation if inverse is False else self.transformation_inv
        init_shape = points.shape
        points = points.reshape(-1,2)

        num_points = points.shape[0]
        homogeneous_points = np.hstack([points, np.ones((num_points, 1))])

        # Apply the homography matrix
        transformed_homogeneous = m @ homogeneous_points.T

        # Convert back to Cartesian coordinates
        transformed_homogeneous /= transformed_homogeneous[2, :]
        transformed_points = transformed_homogeneous[:2, :].T

        return transformed_points.reshape(init_shape)

    def transform_bbox(self, bbox: NDArray):
        pass

    def transform_mask(self, mask: NDArray):
        # interp_mode = cv2.INTER_NEAREST if exact_interpolation else cv2.INTER_CUBIC
        # img = cv2.resize(img, intermediate_size, interpolation=interp_mode)
        pass #return cv2.warpAffine(img, M, )

    @staticmethod
    def crop_image_around_bbox(bbox, src_size, dst_size, boundary=0):
        x1, y1, x2, y2 = bbox

        # Add boundary
        x1 -= boundary
        y1 -= boundary
        x2 += boundary
        y2 += boundary

        bbox_width = x2 - x1
        bbox_height = y2 - y1
        dst_width, dst_height = dst_size

        # Compute scale to fit bbox into dst_size, preserving aspect ratio
        scale_x = dst_width / bbox_width
        scale_y = dst_height / bbox_height
        scale = min(scale_x, scale_y)

        # Scaled size of the bbox region
        scaled_width = bbox_width * scale
        scaled_height = bbox_height * scale

        # Padding needed to center the scaled bbox
        pad_x = (dst_width - scaled_width) / 2
        pad_y = (dst_height - scaled_height) / 2

        # Translation to move (x1, y1) to (pad_x, pad_y)
        tx = pad_x - x1 * scale
        ty = pad_y - y1 * scale

        m =  np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float32)
        return ImageTransformation(m, src_size=src_size, dst_size=dst_size)

    @staticmethod
    def crop_and_align_image(corners, corners_dst, src_size, dst_size):
        img_width, img_height = dst_size
        if corners_dst is None:
            corners_dst = np.array([[0, 0], [img_width, 0], [img_width, img_height], [0, img_height]], dtype=np.float32)
        m = cv2.getPerspectiveTransform(corners.astype(np.float32), corners_dst)
        return ImageTransformation(m, src_size=src_size, dst_size=dst_size)

    @staticmethod
    def affine_warp(src_points, dst_points, src_size, dst_size):
        pass