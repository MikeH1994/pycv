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


class CalibIOParser:
    def __init__(self):
        pass

    def parse(self, json_fpath) -> List[CameraCalibration]:
        dst = []
        with open(json_fpath) as f:
            data = json.load(f)

        image_0_info = data["fileInfo"][0]
        for camera_dict in image_0_info:
            fpath = camera_dict["filePath"]
            name = name_from_fpath(fpath)
            using_median = "median" in fpath
            distance = float(os.path.basename(os.path.dirname(json_fpath)).split("m")[0].split(" ")[-1])

            camera_info = {
                "name": name,
                "using_median": using_median,
                "distance": distance,
                "points_x": [],
                "points_y": [],
                "errors_x": [],
                "errors_y": [],
                "image_localisation": {}
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

        return dst

