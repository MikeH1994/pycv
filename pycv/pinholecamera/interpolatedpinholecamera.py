import numpy as np
from numpy.typing import NDArray
from typing import Union, Tuple, List
import pycv
from .pinholecameramaths import deproject_to_3d_vector, project_points_to_2d, find_camera_pose_from_pnp
from .interpolateddistortionmap import InterpolatedDistortionMap
from .pinholecamera import PinholeCamera
from ..imageutils import InterpolatedImage


class InterpolatedPinholeCamera(PinholeCamera):
    def __init__(self, camera_matrix, res: Tuple[int, int], distortion_map: InterpolatedDistortionMap):
        super().__init__(camera_matrix, res)
        self.distortion_map = distortion_map

    def distort_points(self, points: NDArray):
        """
        Applies distortion to a set of image points.
        :param points: Shape: (..., 2)
        :return:
        """
        return self.distortion_map.distort_points(points)

    def undistort_points(self, points: NDArray):
        """
        Applies undistortion to a set of image points.
        :param points: Shape: (..., 2)
        :return:
        """
        return self.distortion_map.undistort_points(points)

    @staticmethod
    def create_average_pinhole_cameras(cameras: List[PinholeCamera]):
        fx = []
        fy = []
        cx = []
        cy = []
        d_map_x = []
        d_map_y = []
        u_map_x = []
        u_map_y = []

        for camera in cameras:
            fx.append(camera.fx())
            fy.append(camera.fy())
            cx.append(camera.cx())
            cy.append(camera.cy())
            dx, dy = pycv.unstack(camera.create_distortion_map())
            ux, uy = pycv.unstack(camera.create_undistortion_map())
            d_map_x.append(dx)
            d_map_y.append(dy)
            u_map_x.append(ux)
            u_map_y.append(uy)

        fx = np.mean(np.array(fx)).item()
        fy = np.mean(np.array(fy)).item()
        cx = np.mean(np.array(cx)).item()
        cy = np.mean(np.array(cy)).item()
        d_map_x = InterpolatedImage(np.mean(np.array(d_map_x), axis=0))
        d_map_y = InterpolatedImage(np.mean(np.array(d_map_y), axis=0))
        u_map_x = InterpolatedImage(np.mean(np.array(u_map_x), axis=0))
        u_map_y = InterpolatedImage(np.mean(np.array(u_map_y), axis=0))

        res = cameras[0].res()
        camera_matrix = pycv.create_camera_matrix(fx, fy, cx, cy)
        distortion_map = InterpolatedDistortionMap(d_map_x, d_map_y, u_map_x, u_map_y)
        distortion_map.verify((0, res[0]), (0, res[1]))
        return InterpolatedPinholeCamera(camera_matrix, res, distortion_map)
