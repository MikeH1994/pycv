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

class CalibIOParser:
    @staticmethod
    def load(json_fpath) -> List[CameraCalibration]:
        dst = []
        camera_dicts = CalibIOParser._parse_file(json_fpath)
        for i, dicts in enumerate(camera_dicts):
            d = dicts["parameters"]
            cal = CameraCalibration()
            cal.camera_matrix = pycv.create_camera_matrix(d["f"], d["f"], d["cx"], d["cy"])
            cal.distortion_coeffs = pycv.create_distortion_coeffs(d["k1"], d["k2"], d["k3"], d["p1"], d["p2"],
                                                                  d["k4"], d["k5"], d["k6"])
            dst.append(cal)
        return dst

    @staticmethod
    def _parse_file(json_fpath) -> List[Dict]:
        dst = []
        with open(json_fpath) as f:
            data = json.load(f)

        image_0_info = data["fileInfo"][0]
        for _ in image_0_info:
            camera_info = {
                "name": os.path.basename(json_fpath),
                "points_x": [],
                "points_y": [],
                "errors_x": [],
                "errors_y": [],
                "parameters": {"k1": 0, "k2": 0, "k3": 0, "k4": 0, "k5": 0, "k6": 0, "p1": 0, "p2": 0},
                "image_localisation": {},
            }
            dst.append(camera_info)

        for image_residuals in data["residuals"]:
            camera_id = image_residuals["cameraId"]
            for keypoint in image_residuals["residuals"]:
                dst[camera_id]["points_x"].append(keypoint["point"]["x"])
                dst[camera_id]["points_y"].append(keypoint["point"]["y"])
                dst[camera_id]["errors_x"].append(keypoint["error"]["x"])
                dst[camera_id]["errors_y"].append(keypoint["error"]["y"])

        for dat in data["detections"]:
            camera_id = dat["cameraId"]
            pose_id = dat["poseId"]
            x = []
            y = []
            for feature_point in dat["featurePoints"]:
                x.append(feature_point["point"]["x"])
                y.append(feature_point["point"]["y"])
            dst[camera_id]["image_localisation"][pose_id] = (
            np.array(x, dtype=np.float32), np.array(y, dtype=np.float32))

        for i, cal in enumerate(data["calibration"]["cameras"]):
            for param_name, d in cal["model"]["ptr_wrapper"]["data"]["parameters"].items():
                if d["state"] == 0:
                    dst[i]["parameters"][param_name] = d["val"]


        return dst
