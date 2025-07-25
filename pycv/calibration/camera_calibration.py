from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from typing import Tuple, Union, List, Any
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import pickle
import pycv
from pycv.constants import *
from pycv.pinholecamera import PinholeCamera


class CalibrationTarget:
    board_size: Union[Tuple[int, int]] = None
    grid_width: Union[float, None] = None
    grid_type: Union[int, None] = None
    board_type: Union[int, None] = None
    object_points: Union[List[NDArray], None] = None

    def __init__(self, board_size: Tuple[int, int], grid_width: float, board_type: int):
        self.board_size: Tuple[int, int] = board_size
        self.grid_width: float = grid_width
        self.board_type: int = board_type
        self.grid_type: int = CALIB_CB_SYMMETRIC_GRID if self.board_type != CALIB_CB_CIRCLE_GRID_ASYMMETRIC else CALIB_CB_SYMMETRIC_GRID
        self.object_points: NDArray = create_calibration_target_object_points(self.board_size, self.grid_width)


class CameraCalibration:
    default_calibration_flags: int = cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_TANGENT_DIST
    device_name: str = ""
    image_size: Union[Tuple[int, int], None] = None
    image_points_per_frame: Union[List[NDArray], None] = None
    object_points_per_frame: Union[List[NDArray], None] = None
    image_keys: Union[List[Union[str, int]], None] = None
    camera_matrix: Union[NDArray, None] = None
    distortion_coeffs: Union[NDArray, None] = None
    rvecs: Union[List[NDArray]] = None
    tvecs: Union[List[NDArray]] = None
    rms: float = 0.0
    n_frames: int = 0
    n_frames_failed: int = 0

    def __init__(self, device_name="device"):
        self.image_points_per_frame: List[NDArray] = []
        self.image_keys: List[Union[str, int]] = []
        self.object_points_per_frame: List[NDArray] = []
        self.device_name = device_name

    def add_calibration_point(self, img: NDArray, target: CalibrationTarget, key: Union[str, int]=None, plot=False, verbose=False):
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
        if target.board_type == CALIB_CB_CHECKERBOARD:
            success, image_points, overlayed_image = add_calibration_point_checkerboard(img, target, create_image=plot)
        else:
            success, image_points, overlayed_image = add_calibration_point_circle_grid(img, target, create_image=plot)
        if success:
            self.image_points_per_frame.append(image_points)
            self.object_points_per_frame.append(target.object_points.astype(np.float32))
            self.image_keys.append(key)
        else:
            self.n_frames_failed += 1

        if overlayed_image is not None and plot:
            plt.imshow(overlayed_image)
            plt.show()

        if verbose:
            print(f"    Device: {self.device_name} Calibration point: {key} Status: {success}")

    def calibrate(self, alpha: float = None, calibration_flags: int = None, verbose=False):
        calibration_flags = self.default_calibration_flags if calibration_flags is None else calibration_flags
        self.rms, self.camera_matrix, self.distortion_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(self.object_points_per_frame,
                                                                                         self.image_points_per_frame,
                                                                                         self.image_size, None, None,
                                                                                         flags=calibration_flags)
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
        m = self.camera_matrix
        cx, cy = m[0, 2], m[1, 2]
        fx, fy = m[0, 0], m[1, 1]
        w, h = self.image_size
        hfov, vfov = pycv.focal_length_to_fov(fx, w), pycv.focal_length_to_fov(fy, h)
        rms = self.rms
        return {
            "cx": cx, "cy": cy, "fx": fx, "fy": fy, "w": w, "h": h, "hfov": hfov, "vfov": vfov,
            "n_images": self.n_frames, "n_images_used": self.n_frames - self.n_frames_failed, "rms":rms,
        }

    def undistort_image(self, img: NDArray):
        return cv2.undistort(img, self.camera_matrix, self.distortion_coeffs)

    def compute_reprojection_error(self):
        mean_error = 0
        for i in range(len(self.object_points_per_frame)):
            imgpoints2, _ = cv2.projectPoints(self.object_points_per_frame[i], rvecs[i], tvecs[i], self.camera_matrix, self.distortion_coeffs)
            error = cv2.norm(self.image_points_per_frame[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        print("total error: {}".format(mean_error / len(objpoints)))

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
        with open(fpath, 'wb') as f:
            pickle.dump(self.__dict__, f)
        if verbose:
            print(f"Device '{self.device_name}' calibration saved to {fpath}")

    def load(self, fpath: str, verbose=False):
        with open(fpath, "rb") as f:
            data = pickle.load(f)
            if data is None:
                raise Exception("Invalid data loaded")
            self.__dict__ = data
        if verbose:
            print(f"Device '{self.device_name}' calibration loaded from {fpath}")
            self.print()

    def create_pinhole_camera(self, include_distortion=True) -> PinholeCamera:
        dist_coeffs = self.distortion_coeffs if include_distortion else np.zeros(5)
        return PinholeCamera(self.camera_matrix, self.image_size, distortion_coeffs=dist_coeffs)

def create_calibration_target_object_points(board_size: Tuple[int, int], dx: float):
    """
    Creates a grid calibration target
    :param board_size: a tuple of the form (width, height),
        containing the dimension of the grid
    :param dx: the distance between each point on the grid
    :return:
    """
    width, height = board_size
    dx = dx
    object_points = []
    for j in range(height):
        for i in range(width):
            object_points.append([i * dx, j * dx, 0.0])
    return np.array(object_points)


def add_calibration_point_circle_grid(img, calib_target: CalibrationTarget, use_larger_blobs=False, create_image=True):
    if use_larger_blobs:
        params = cv2.SimpleBlobDetector_Params()
        params.maxArea = 1e5
        blob_detector = cv2.SimpleBlobDetector_create(params)
    else:
        blob_detector = None
    success, image_points = cv2.findCirclesGrid(img, calib_target.board_size, calib_target.grid_type, blobDetector=blob_detector)
    if success is False and use_larger_blobs is False:
        # as we didn't find it using the default arguments, try again with larger blob size
        return add_calibration_point_circle_grid(img, calib_target, use_larger_blobs=True, create_image=create_image)

    overlayed_image = None
    if success and create_image:
        img_rgb = pycv.to_rgb(img)
        overlayed_image = cv2.drawChessboardCorners(img_rgb, calib_target.board_size, image_points, success)

    return success, image_points, overlayed_image


def add_calibration_point_checkerboard(img, target: CalibrationTarget, create_image: bool = True):
    success, corners = cv2.findChessboardCornersSB(img, target.board_size, flags=cv2.CALIB_CB_EXHAUSTIVE)

    #if success is True:
    #    corners = cv2.cornerSubPix(img, corners, target.board_size, (-1, -1))

    overlayed_image = None
    if success and create_image:
        img_rgb = pycv.to_rgb(img)
        overlayed_image = cv2.drawChessboardCorners(img_rgb, target.board_size, corners, success)

    return success, corners, overlayed_image

