import numpy as np
from numpy.typing import NDArray
from typing import Union, Tuple
from .pinhole_camera_maths import get_pixel_direction, get_pixel_point_lies_in, find_camera_pose_from_pnp
from .pinhole_camera_maths import focal_length_to_fov, fov_to_focal_length,unpack_camera_matrix
import cv2


class PinholeCamera:
    def __init__(self, camera_matrix: NDArray, res: Tuple[int, int], distortion_coeffs: NDArray = np.zeros(5),
                 p: NDArray = np.zeros(3), r: NDArray = np.eye(3)):
        self.camera_matrix = np.copy(camera_matrix)
        self.distortion_coeffs = np.copy(distortion_coeffs)
        self.xres, self.yres = res
        self.p = np.copy(p)
        self.r = np.copy(r)

    def reproject_points_to_3d(self, x: Union[float, NDArray], y: Union[float, NDArray], d: float = 1.0):
        pass

    def project_points_to_image_plane(self, points: NDArray, return_as_int=False):
        fx, fy, cx, cy = unpack_camera_matrix(self.camera_matrix)

        hfov, vfov = focal_length_to_fov(fx, self.xres), focal_length_to_fov(fy, self.yres)
        return get_pixel_point_lies_in(points, self.p, self.r, (self.xres, self.yres), (hfov, vfov), (cx, cy))

    def find_camera_pose_from_pnp(self, object_points: NDArray, image_points: NDArray):
        self.p, self.r = find_camera_pose_from_pnp(self.camera_matrix, object_points, image_points, self.distortion_coeffs)