from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from typing import Tuple, Union, List, Any
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import pickle
import pycv
from pycv.core import convert_to_8_bit
from pycv.constants import *
from pycv.pinholecamera import PinholeCamera
from pycv.pinholecamera import unpack_camera_matrix, distortion_coefficients_to_dict
import json
import os

class CalibrationTarget:
    board_size: Union[Tuple[int, int]] = None
    grid_width: Union[float, None] = None
    grid_type: Union[int, None] = None
    board_type: Union[int, None] = None
    object_points: Union[List[NDArray], None] = None

    def __init__(self, board_size: Tuple[int, int], checker_width: float, checker_height=None, board_type: int = CALIB_BOARD_TYPE_CHECKERBOARD):
        self.board_size: Tuple[int, int] = board_size
        self.grid_width: float = checker_width
        self.grid_height = checker_width if checker_height is None else checker_height
        self.board_type: int = board_type
        if board_type == CALIB_BOARD_TYPE_CHECKERBOARD or board_type == CALIB_BOARD_TYPE_CIRCLE_GRID_SYMMETRIC:
            self.grid_type = cv2.CALIB_CB_SYMMETRIC_GRID
        elif board_type == CALIB_BOARD_TYPE_CIRCLE_GRID_ASYMMETRIC:
            self.grid_type = cv2.CALIB_CB_ASYMMETRIC_GRID
        else:
            raise Exception("Unknown board type supplied")
        self.object_points: NDArray = create_calibration_target_object_points(self.board_size, self.grid_width, self.grid_height)

    def get_object_points(self, pos=np.array([0, 0, 0]), rot=np.eye(3)) -> NDArray:
        return self.object_points @ rot.T + pos

class CameraCalibration:
    default_calibration_flags: int = cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6 | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_TANGENT_DIST
    device_name: str = ""
    image_size: Union[Tuple[int, int], None] = None
    image_points_per_frame: Union[List[NDArray], None] = None
    object_points_per_frame: Union[List[NDArray], None] = None
    residual_errors_per_frame: Union[List[NDArray], None] = None
    image_keys: Union[List[Union[str, int]], None] = None
    camera_matrix: Union[NDArray, None] = None
    distortion_coeffs: Union[NDArray, None] = None
    rvecs: Union[List[NDArray]] = None
    tvecs: Union[List[NDArray]] = None
    rms: float = 0.0
    n_frames: int = 0
    n_frames_failed: int = 0
    parameter_errors = {}

    def __init__(self, device_name="device"):
        self.image_points_per_frame: List[NDArray] = []
        self.image_keys: List[Union[str, int]] = []
        self.object_points_per_frame: List[NDArray] = []
        self.target_positions = []
        self.target_rotations = []
        self.device_name = device_name

    def add_calibration_point(self, img: NDArray, target: CalibrationTarget, key: Union[str, int]=None, display=False, verbose=False):
        key = len(self.image_keys) if key is None else key
        assert(key not in self.image_keys), "Key already exists!"
        xy_image_size = img.shape[:2][::-1]
        if self.image_size is None:
            # reversing numpy dimensions as opencv uses (width, height) instead of (height, width) like numpy
            # (and any sane person) would do
            self.image_size = xy_image_size
        assert(self.image_size == xy_image_size), "Dimensions of image do not match previous calibration point-\n" \
                                                      "current: {} stored: {}".format(xy_image_size, self.image_size)
        self.n_frames += 1
        if target.board_type == CALIB_BOARD_TYPE_CHECKERBOARD:
            success, image_points, overlayed_image = find_checkerboard_corners(img, target.board_size, create_image=display)
        else:
            success, image_points, overlayed_image = find_circles_grid(img, target, create_image=display)
        if success:
            self.image_points_per_frame.append(image_points)
            self.object_points_per_frame.append(target.object_points.astype(np.float32))
            self.image_keys.append(key)
        else:
            self.n_frames_failed += 1

        if overlayed_image is not None and display:
            plt.imshow(overlayed_image)
            plt.show()

        if verbose:
            print(f"    Device: {self.device_name} Calibration point: {key} Status: {success}")

    def calibrate(self, alpha: float = None, calibration_flags: int = None, verbose=False, init_camera_matrix=None, init_distortion_coeffs=None):
        calibration_flags = self.default_calibration_flags if calibration_flags is None else calibration_flags
        if init_camera_matrix is not None or init_distortion_coeffs is not None:
            calibration_flags |= cv2.CALIB_USE_INTRINSIC_GUESS

        self.rms, self.camera_matrix, self.distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(self.object_points_per_frame,
                                                                                         self.image_points_per_frame,
                                                                                         self.image_size, cameraMatrix=init_camera_matrix, distCoeffs=init_distortion_coeffs,
                                                                                         flags=calibration_flags)

        self.target_positions = []
        self.target_rotations = []
        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.reshape(3)
            self.target_rotations.append(R)
            self.target_positions.append(t)

        if alpha is not None:
            newcamera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.distortion_coeffs,
                                                                  self.image_size, alpha)
            print(self.camera_matrix)
            print("-----------------")
            print(newcamera_matrix)
            print(roi)
            self.camera_matrix = newcamera_matrix

        if verbose:
            self.print()

        return self.rms

    def get_parameter(self, param):
        return self.get_parameters()[param]

    def get_parameters(self):
        assert(self.camera_matrix is not None)
        fx, fy, cx, cy = unpack_camera_matrix(self.camera_matrix)

        w, h = self.image_size
        hfov, vfov = pycv.focal_length_to_fov(fx, w), pycv.focal_length_to_fov(fy, h)
        rms = self.rms
        ret = {
            "cx": cx, "cy": cy, "fx": fx, "fy": fy, "w": w, "h": h, "hfov": hfov, "vfov": vfov,
            "n_images": self.n_frames, "n_images_used": self.n_frames - self.n_frames_failed, "rms":rms,
        }
        ret.update(distortion_coefficients_to_dict(self.distortion_coeffs))
        return ret

    def get_parameter_errors(self, param, return_type_if_missing=0.0):
        if param in self.parameter_errors:
            return param
        else:
            return return_type_if_missing

    def undistort_image(self, img: NDArray):
        return cv2.undistort(img, self.camera_matrix, self.distortion_coeffs)

    def compute_reprojection_error(self):
        mean_error = 0
        """for i in range(len(self.object_points_per_frame)):
            imgpoints2, _ = cv2.projectPoints(self.object_points_per_frame[i], rvecs[i], tvecs[i], self.camera_matrix, self.distortion_coeffs)
            error = cv2.norm(self.image_points_per_frame[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        print("total error: {}".format(mean_error / len(objpoints)))"""

    def print(self, lpad="\t"):
        p = self.get_parameters()
        lpad2 = lpad+"\t"
        print("{}Device '{}' summary:".format(lpad, self.device_name))
        print("{}images used = {}/{}".format(lpad2, p["n_images"], p["n_images_used"]))
        print("{}rms         = {:.4f}px".format(lpad2, p["rms"]))
        print("{}(w,h)       = ({},{})".format(lpad2, p["w"], p["h"]))
        print("{}(cx,cy)     = ({:.3f},{:.3f})".format(lpad2, p["cx"], p["cy"]))
        print("{}(fx,fy)     = ({:.3f},{:.3f})".format(lpad2, p["fx"], p["fy"]))
        print("{}(hfov,vfov) = ({:.3f},{:.3f})".format(lpad2, p["hfov"], p["vfov"]))
        print("{} {}".format(lpad2, self.distortion_coeffs))

    def save(self, fpath: str, verbose=False):
        if fpath.endswith(".pck"):
            with open(fpath, 'wb') as f:
                pickle.dump(self.__dict__, f)
            if verbose:
                print(f"Device '{self.device_name}' calibration saved to {fpath}")
        elif fpath.endswith(".json"):
            data = {
                "camera_matrix": self.camera_matrix.tolist(),
                "distortion_coeffs": self.distortion_coeffs.tolist(),
                "image_size": self.image_size,
                "target_positions": [m.tolist() for m in self.target_positions],
                "target_rotations": [m.tolist() for m in self.target_rotations],
                "rms": self.rms,
                "image_points_per_frame": [m.tolist() for m in self.image_points_per_frame]
            }
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        else:
            raise NotImplementedError("Unknown save format: {}".format(fpath))

    def load(self, fpath: str, verbose=False):
        if fpath.endswith(".pck"):
            with open(fpath, "rb") as f:
                data = pickle.load(f)
                if data is None:
                    raise Exception("Invalid data loaded")
                self.__dict__ = data
        elif fpath.endswith(".json"):
            raise NotImplementedError("Unknown save format: {}".format(fpath))
        else:
            raise NotImplementedError("Unknown save format: {}".format(fpath))

        if verbose:
            print(f"Device '{self.device_name}' calibration loaded from {fpath}")
            self.print()

    def create_pinhole_camera(self, include_distortion=True) -> PinholeCamera:
        dist_coeffs = self.distortion_coeffs if include_distortion else np.zeros(5)
        return PinholeCamera(self.camera_matrix, self.image_size, distortion_coeffs=dist_coeffs)

def create_calibration_target_object_points(board_size: Tuple[int, int], dx: float, dy = None):
    """
    Creates a grid calibration target
    :param board_size: a tuple of the form (width, height),
        containing the dimension of the grid
    :param dx: the distance between each point on the grid
    :return:
    """
    width, height = board_size
    dy = dx if dy is None else dy
    object_points = []
    for j in range(height):
        for i in range(width):
            object_points.append([i * dx, j * dy, 0.0])
    return np.array(object_points)


def find_circles_grid(img, board_size, use_larger_blobs=False, create_image=True, grid_type=cv2.CALIB_CB_SYMMETRIC_GRID):
    if use_larger_blobs:
        params = cv2.SimpleBlobDetector_Params()
        params.maxArea = 1e5
        blob_detector = cv2.SimpleBlobDetector_create(params)
    else:
        blob_detector = None
    success, image_points = cv2.findCirclesGrid(img,board_size, grid_type, blobDetector=blob_detector)
    if success is False and use_larger_blobs is False:
        # as we didn't find it using the default arguments, try again with larger blob size
        return find_circles_grid(img, board_size, use_larger_blobs=True, create_image=create_image, grid_type=grid_type)

    overlayed_image = None
    if success and create_image:
        img_rgb = pycv.to_rgb(img)
        overlayed_image = cv2.drawChessboardCorners(img_rgb, board_size, image_points, success)

    return success, image_points, overlayed_image


def find_checkerboard_corners(img, board_size: Tuple[int, int], create_image: bool = True, winSize=(11, 11), zeroZone=(-1, -1)):
    img8 = img if img.dtype == np.uint8 else convert_to_8_bit(img)
    success, corners = cv2.findChessboardCornersSB(img8, board_size, flags=cv2.CALIB_CB_EXHAUSTIVE)

    if success is True:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-8)
        corners = cv2.cornerSubPix(img, corners, winSize=winSize, zeroZone=zeroZone, criteria=criteria)

    overlayed_image = None
    if success and create_image:
        img_rgb = pycv.to_rgb(img8)
        overlayed_image = cv2.drawChessboardCorners(img_rgb, board_size, corners, success)
    return success, corners, overlayed_image
