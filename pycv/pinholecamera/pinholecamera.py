import numpy as np
from numpy.typing import NDArray
from typing import Union, Tuple
from .pinhole_camera_maths import deproject_to_3d_vector, project_points_to_2d, find_camera_pose_from_pnp, \
    rotation_matrix_to_axes, distort_points, undistort_points
from .pinhole_camera_maths import focal_length_to_fov, unpack_camera_matrix, rotation_matrix_to_lookpos, lookpos_to_rotation_matrix
import cv2
from ..core import unstack_coords


class PinholeCamera:
    def __init__(self, camera_matrix: NDArray, res: Tuple[int, int], distortion_coeffs: NDArray = np.zeros(5),
                 p: NDArray = np.zeros(3), r: NDArray = np.eye(3)):
        """

        :param camera_matrix:
        :param res: (width, height)
        :param distortion_coeffs:
        :param p:
        :param r:
        """
        self.camera_matrix = np.copy(camera_matrix)
        self.distortion_coeffs = np.copy(distortion_coeffs)
        self.xres, self.yres = res
        self.p = np.copy(p)
        self.r = np.copy(r)

    def deproject_to_3d_vector(self, points, apply_distortion=True, normalise=True) -> NDArray:
        if apply_distortion:
            points = self.distort_points(points)
        return deproject_to_3d_vector(points, self.r, self.camera_matrix, normalise=normalise)

    def project_points_to_2d(self, points: NDArray, return_as_int=False):
        fx, fy, cx, cy = unpack_camera_matrix(self.camera_matrix)

        hfov, vfov = focal_length_to_fov(fx, self.xres), focal_length_to_fov(fy, self.yres)
        points = project_points_to_2d(points, self.p, self.r, self.camera_matrix)
        if return_as_int:
            points = points.astype(np.int32)
        return points

    def distort_points(self, points: NDArray):
        """

        :param points:
        :return:
        """
        return distort_points(points, self.camera_matrix, self.distortion_coeffs)

    def undistort_points(self, points: NDArray):
        """

        :param points:
        :return:
        """
        return undistort_points(points, self.camera_matrix, self.distortion_coeffs)

    def set_lookpos(self, lookpos, y = None):
        """

        :param lookpos:
        :param y:
        :return:
        """
        if y is None:
            _, y, _ = self.axes()
        self.r = lookpos_to_rotation_matrix(self.p, lookpos, y)


    def generate_rays(self, apply_distortion=True, direction_only=False, normalise=True) -> NDArray:
        """
        Generate a set of rays for each pixel in Open3D's format for use in the Open3D raycasting. Each Open3D ray is
            a vector of length 6, where the first 3 elements correspond to the origin of the ray (the camera position),
            and the last 3 elements are the direction vector of the ray
        :return: a 3D array of shape (yres, xres, n_samples, 6) corresponding to the open3d rays for each pixel
        :rtype: np.ndarray
        """

        xx, yy = np.meshgrid(np.arange(self.xres), np.arange(self.yres))
        pixel_coords = np.zeros((*xx.shape, 2), dtype=np.float32)
        pixel_coords[:, :, 0] = xx
        pixel_coords[:, :, 1] = yy

        init_shape = pixel_coords.shape
        pixel_direction = self.deproject_to_3d_vector(pixel_coords, apply_distortion=apply_distortion, normalise=normalise)

        if direction_only:
            rays = pixel_direction.reshape((*init_shape[:-1], 3))
        else:
            rays = np.zeros((pixel_coords.shape[0], 6), dtype=np.float32)
            rays[:, :3] = self.p
            rays[:, 3:] = pixel_direction
            rays = rays.reshape((*init_shape[:-1], 6))
        return rays

    def hfov(self):
        return focal_length_to_fov(self.fx(), self.xres)

    def vfov(self):
        return focal_length_to_fov(self.fy(), self.yres)

    def cx(self):
        _, _, cx, _ = unpack_camera_matrix(self.camera_matrix)
        return cx

    def cy(self):
        _, _, _, cy = unpack_camera_matrix(self.camera_matrix)
        return cy

    def fx(self):
        fx, _, _, _ = unpack_camera_matrix(self.camera_matrix)
        return fx

    def fy(self):
        _, fy, _, _ = unpack_camera_matrix(self.camera_matrix)
        return fy

    def res(self):
        return self.xres, self.yres

    def optical_centre(self):
        return self.cx(), self.cy()

    def lookpos(self):
        _, _, z_axis = self.axes()
        return self.p + z_axis

    def axes(self):
        axes = rotation_matrix_to_axes(self.r)
        return axes

    def extrinsics(self):
        dst = np.eye(4)
        dst[:3, :3] = self.r
        dst[3, :3] = self.p
        return dst

    def pixel_size(self, distance, using_fov=True):
        if using_fov:
            hfov = self.hfov()
            return distance*np.tan(np.radians(hfov/2))/(self.xres/2)
        else:
            return distance / self.fx()

    def find_camera_pose_from_pnp(self, object_points: NDArray, image_points: NDArray):
        self.p, self.r = find_camera_pose_from_pnp(self.camera_matrix, object_points, image_points, self.distortion_coeffs)
