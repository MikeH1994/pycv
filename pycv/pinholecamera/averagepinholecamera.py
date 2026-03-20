import numpy as np
from numpy.typing import NDArray
from typing import Union, Tuple, List
import pycv
from .pinholecameramaths import deproject_to_3d_vector, project_points_to_2d, find_camera_pose_from_pnp
from .interpolateddistortionmap import InterpolatedDistortionMap
from .pinholecamera import PinholeCamera
from ..imageutils import InterpolatedImage


class AveragePinholeCamera(PinholeCamera):
    def __init__(self, cameras: List[PinholeCamera]):
        if len(cameras) == 0:
            raise Exception("No cameras passed to average pinhole camera.")
        self.cameras = cameras
        camera_matrix = np.mean(np.array([cam.camera_matrix for cam in cameras]), axis=0)
        res = cameras[0].res()
        super().__init__(camera_matrix, res)


    def distort_points(self, points: NDArray):
        """
        Applies distortion to a set of image points.
        :param points: Shape: (..., 2)
        :return:
        """
        return np.mean(np.array([cam.distort_points(points) for cam in self.cameras]), axis=0)

    def undistort_points(self, points: NDArray):
        """
        Applies undistortion to a set of image points.
        :param points: Shape: (..., 2)
        :return:
        """
        return np.mean(np.array([cam.undistort_points(points) for cam in self.cameras]), axis=0)
