from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from typing import Tuple, Union, List, Any
import pycv
from pycv.pinholecamera import PinholeCamera, InterpolatedDistortionMap, InterpolatedPinholeCamera, AveragePinholeCamera
from .camera_calibration import CameraCalibration
from ..imageutils import InterpolatedImage


class FittedCameraCalibration:
    def __init__(self, calibrations: List[CameraCalibration], distances: List[float]):
        self.calibrations = calibrations
        self.res = calibrations[0].image_size
        self.distances = distances
        self.data = {
            "fx": [calib.get_parameter("fx") for calib in calibrations],
            "fy": [calib.get_parameter("fy") for calib in calibrations],
            "cx": [calib.get_parameter("cx") for calib in calibrations],
            "cy": [calib.get_parameter("cy") for calib in calibrations],
            "dist_x": [calib.create_pinhole_camera().create_distortion_map()[:, :, 0] for calib in calibrations],
            "dist_y": [calib.create_pinhole_camera().create_distortion_map()[:, :, 1] for calib in calibrations],
            "undist_x": [calib.create_pinhole_camera().create_undistortion_map()[:, :, 0] for calib in calibrations],
            "undist_y": [calib.create_pinhole_camera().create_undistortion_map()[:, :, 1] for calib in calibrations]
        }
        self._coeffs = {}
        self._residuals = {}
        degree = 1
        self.dof = len(calibrations) - (degree + 1)
        for coeff_name, coeff_data in self.data.items():
            if "dist" not in coeff_name:
                x = np.array(self.distances)
                y = np.array(coeff_data)
                coefficients = np.polyfit(x, y, degree)
                residuals = y - np.polyval(coefficients, x)
                residuals = np.sqrt(np.sum(residuals ** 2)/self.dof)
            else:
                width, height = calibrations[0].image_size
                coefficients = np.zeros((height, width, degree + 1), dtype=np.float32)
                residuals = np.zeros((height, width), dtype=np.float32)
                for i in range(width):
                    for j in range(height):
                        x = np.array(distances)
                        y = np.array([d[j, i] for d in coeff_data])
                        coefficients[j, i, :] = np.polyfit(x, y, degree)
                        err = y - np.polyval(coefficients[j, i, :], x)
                        residuals[j, i] = np.sqrt(np.sum(err ** 2)/self.dof)

            self._coeffs[coeff_name] = coefficients
            self._residuals[coeff_name] = residuals

    def evaluate(self, param_name, distance) -> Union[np.ndarray, float]:
        coeffs = self._coeffs[param_name]
        n_coeffs = coeffs.shape[-1]
        x = 0
        if "dist" in param_name:
            for j in range(n_coeffs):
                x += coeffs[:, :, j] * distance**(n_coeffs - j - 1)
        else:
            for j in range(n_coeffs):
                x += coeffs[j] * distance**(n_coeffs - j - 1)
        return x

    def residual_standard_error(self, parameter):
        return self._residuals[parameter]

    def fx(self, distance):
        return self.evaluate("fx", distance)

    def fy(self, distance):
        return self.evaluate("fy", distance)

    def cx(self, distance):
        return self.evaluate("cx", distance)

    def cy(self, distance):
        return self.evaluate("cy", distance)

    def distortion_map(self, distance):
        dx = self.evaluate("dist_x", distance)
        distortion_x = InterpolatedImage(dx)
        distortion_y = InterpolatedImage(self.evaluate("dist_y", distance))
        undistortion_x = InterpolatedImage(self.evaluate("undist_x", distance))
        undistortion_y = InterpolatedImage(self.evaluate("undist_y", distance))
        return InterpolatedDistortionMap(distortion_x, distortion_y, undistortion_x, undistortion_y)

    def create_pinhole_camera(self, distance) -> InterpolatedPinholeCamera:
        fx = self.fx(distance)
        fy = self.fy(distance)
        cx = self.cx(distance)
        cy = self.cy(distance)
        distortion_map = self.distortion_map(distance)
        camera_matrix = pycv.create_camera_matrix(fx, fy, cx, cy)
        return InterpolatedPinholeCamera(camera_matrix, self.res, distortion_map)
