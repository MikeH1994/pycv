from __future__ import annotations
import numpy as np
import os.path
import cv2
import matplotlib.pyplot as plt
import pycv
import glob
from tqdm.auto import tqdm
from pycv.calibration import CalibrationTarget, CameraCalibration
from scipy.interpolate import interp1d
import json
from typing import List
from typing import Dict
from scipy.spatial.transform import Rotation as R


class CalibIOParser:
    @staticmethod
    def load(json_fpath) -> List[CameraCalibration]:
        dst = []
        camera_dicts = CalibIOParser._parse_file(json_fpath)
        for i, dicts in enumerate(camera_dicts):
            d = dicts["parameters"]
            cal = CameraCalibration()
            cal.camera_matrix = pycv.create_camera_matrix(d["f"], d["f"], d["cx"], d["cy"])
            cal.distortion_coeffs = pycv.create_distortion_coeffs(d["k1"], d["k2"], d["k3"], d["p1"], d["p2"], d["k4"], d["k5"], d["k6"])
            cal.image_size = dicts["image_size"]
            cal.image_points_per_frame = dicts["image_localisation"]
            cal.object_points_per_frame = [dicts["object_points"] for _ in range(len(dicts["image_localisation"]))]
            cal.rvecs = dicts["rvecs"]
            cal.tvecs = dicts["tvecs"]
            dst.append(cal)
        return dst

    @staticmethod
    def _parse_file(json_fpath) -> List[Dict]:
        camera_list = []
        with open(json_fpath) as f:
            data = json.load(f)

        image_0_info = data["fileInfo"][0]
        for _ in image_0_info:
            camera_info = {
                "name": os.path.basename(json_fpath),
                "parameters": {"k1": 0, "k2": 0, "k3": 0, "k4": 0, "k5": 0, "k6": 0, "p1": 0, "p2": 0},
                "rvecs": [],
                "tvecs": [],
                "image_size": (0, 0),
                "errors": [],
                "image_localisation": [],
                "successful_poses": [],
                "object_points": []
            }
            camera_list.append(camera_info)

        # get successful poses, image point localisation and errors
        for image_residuals in data["residuals"]:
            camera_id = image_residuals["cameraId"]
            x = []
            y = []
            errors_x = []
            errors_y = []
            for keypoint in image_residuals["residuals"]:
                x.append(keypoint["point"]["x"])
                y.append(keypoint["point"]["y"])
                errors_x.append(keypoint["error"]["x"])
                errors_y.append(keypoint["error"]["y"])
            camera_list[camera_id]["image_localisation"].append(np.stack((x, y), axis=-1))
            camera_list[camera_id]["errors"].append(np.stack((errors_x, errors_y), axis=-1))
            camera_list[camera_id]["successful_poses"].append(image_residuals["poseId"])

        # get calibration parameters and image size
        for i, cal in enumerate(data["calibration"]["cameras"]):
            for param_name, d in cal["model"]["ptr_wrapper"]["data"]["parameters"].items():
                if d["state"] == 0:
                    camera_list[i]["parameters"][param_name] = d["val"]
            image_size = cal["model"]["ptr_wrapper"]["data"]["CameraModelCRT"]["CameraModelBase"]["imageSize"]
            width, height = image_size["width"], image_size["height"]
            camera_list[i]["image_size"] = (width, height)

        # get object points (calibration target)
        object_points = []
        for pt in data["calibration"]["targets"][0]["objectPoints"]:
             object_points.append([pt["x"], pt["y"], pt["z"]])
        object_points = np.array(object_points, dtype=np.float32)
        for camera in camera_list:
            camera["object_points"] = object_points

        # get target poses
        for camera_id, camera in enumerate(camera_list):
            rvecs = []
            tvecs = []
            for i, pose in enumerate(data["calibration"]["poses"]):
                if i not in camera["successful_poses"]:
                    continue

                d = pose["transform"]["rotation"]
                rvec = np.array([d["rx"], d["ry"], d["rz"]], dtype=np.float32)
                d = pose["transform"]["translation"]
                tvec = np.array([d["x"], d["y"], d["z"]], dtype=np.float32)

                tvecs.append(tvec)
                rvecs.append(rvec)

            camera["rvecs"] = rvecs
            camera["tvecs"] = tvecs

        return camera_list
