from __future__ import annotations
from albumentations.core.composition import Compose
from typing import List, Tuple, Dict
import numpy as np
from numpy.typing import NDArray
import torch
import cv2
from ..base.base_dataset import BaseDataset
from ..core.configuration import KeypointRCNNConfiguration
from .utils import albumentations_keypoints_to_rcnn_format


class KeypointRCNNDataset(BaseDataset):
    """
    Generic class for keypoints
    """
    image_fpaths: List[str] = None
    annotation_fpaths: List[str] = None

    def __init__(self, num_classes, num_keypoints):
        self.num_keypoints=num_keypoints
        self.num_classes = num_classes
        super().__init__(use_keypoints=True, use_bboxes=True)

    def initialise(self):
        """
        The implementation of initialise() from inheriting classes should initialise two class variables- image_fpaths,
        and annotations_fpaths. image_fpaths and annotatio
        image_fpaths: List[str] = None
        annotations_fpaths: List[str] = None
        :return:
        """
        raise Exception("Base function initialise() has not been implemented")

    def process_image(self, image: NDArray) -> NDArray:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image

    def load_annotations(self, annotations_filepath, width, height) -> Dict[str, list]:
        """
        Return a dict containing:
            (keypoints, keypoint_classes, bboxes, bbox_classes) \n
            Keypoints: a list of tuples containing the (x,y) coordinates- e.g. [(264, 203), (86, 88), (254, 160)],
                of the shape (n_points, 2)
            keypoint_classes: a list of classes for each point- e.g. [0, 0, 0, 1, 1, 1, 2, 2, 2], of the shape
                (n_points, )
            bboxes: a list of bounding boxes of the form [(a, b, c, d), (a, b, c, d)].
                e.g. (x_min, y_min, x_max, y_max), of the shape (n_bboxes, 4)
            bbox_classes: a list of classes corresponding to each object, of the shape (n_bboxes, 4)

        :param annotations_filepath:
        :param width:
        :param height:
        :return:
        """
        raise Exception("Base function load_annotations(annotations_filepath) has not been implemented")

    def clip_dataset(self, length: int):
        assert(self.image_fpaths is not None), "image_fpaths should be initialised in initialise()"
        assert(self.annotation_fpaths is not None), "annotations_fpaths should be initialised in initialise()"

        if length <= 0:
            return

        self.image_fpaths = self.image_fpaths[:length]
        self.annotation_fpaths = self.annotation_fpaths[:length]

    def __getitem__(self, idx):
        """

        "input": img,
        "target": target,
        "image_filepath": image_fpath,
        "annotation_filepath": annotations_fpath

        Targets is a dict containing:
            "boxes": tensor of the shape (n_bboxes, 4)
            "keypoints": tensor of the shape (n_bboxes, n_keypoints_per_bbox, 3)
            "labels":  tensor of the shape (n_bboxes)
            "image_id": single element tensor (1)
            "area": tensor of the shape (n_bboxes)
            "iscrowd": label that indicates if a box is crowded (?). All elements are returned false

            "boxes": the bounding boxes. shape: (n_bboxes, 4), dtype: float64
            "keypoints": the keypoints associated with each bounding box.
                shape: (n_bboxes, n_keypoints_per_class, 3), dtype: float64
            "labels": shape: (n_bboxes, ), dtype: int64
            "area": shape: (n_bboxes, ), dtype: float64
            "iscrowd": shape: (n_bboxes, ), dtype: bool
        }




        :param idx:
        :return:
        """
        assert(self.image_fpaths is not None), "image_fpaths should be initialised in initialise()"
        assert(self.annotation_fpaths is not None), "mask_fpaths should be initialised in initialise()"

        image_fpath = self.image_fpaths[idx]
        annotations_fpath = self.annotation_fpaths[idx]

        img = self.load_img(image_fpath)
        img = self.process_image(img)
        assert(len(img.shape) == 3 and img.shape[2] == 3), "The input image should be 3 channel"
        height, width = img.shape[:2]
        d  = self.load_annotations(annotations_fpath, width, height)

        bboxes = d["boxes"]
        keypoints = d["keypoints"]
        keypoints_visibility = d["visibility"]
        bbox_classes = d["labels"]
        n_bboxes = len(bboxes)
        bbox_indices = list(range(n_bboxes))
        assert(len(keypoints) == n_bboxes*self.num_keypoints), "{} vs {}".format(len(keypoints),
                                                                                 n_bboxes*self.num_keypoints)
        f = self.img_transform(image=img, keypoints=keypoints, bboxes=bboxes, class_labels=bbox_indices)
        img = f["image"].float()
        bboxes = f["bboxes"]
        keypoints = f["keypoints"]
        bbox_indices = f["class_labels"]
        bbox_classes = [bbox_classes[i] for i in bbox_indices]

        rcnn_data = albumentations_keypoints_to_rcnn_format(keypoints, keypoints_visibility, self.num_keypoints, bboxes,
                                                            bbox_indices, bbox_classes, width, height)

        target = {"boxes": torch.as_tensor(rcnn_data["boxes"],dtype=torch.float),
                  "keypoints": torch.as_tensor(rcnn_data["keypoints"],dtype=torch.float),
                  "labels": torch.as_tensor(rcnn_data["labels"], dtype=torch.int64),
                  "image_id": torch.tensor([idx], dtype=torch.int64),
                  "area": torch.tensor(rcnn_data["area"], dtype=torch.float),
                  "iscrowd": torch.as_tensor(rcnn_data["iscrowd"], dtype=torch.bool),
                  }

        return {
            "input": img,
            "target": target,
            "image_filepath": image_fpath,
            "annotation_filepath": annotations_fpath
        }

    def __len__(self):
        return len(self.image_fpaths)

    @staticmethod
    def collate_fn(batch: List):
        images = []
        targets = []
        image_filepaths = []
        annotation_filepaths = []
        for x in batch:
            skip = True if None in [x["input"], x["target"], x["image_filepath"], x["annotation_filepath"]] else False

            if skip is False:
                images.append(x["input"])
                targets.append(x["target"])
                image_filepaths.append(x["image_filepath"])
                annotation_filepaths.append(x["annotation_filepath"])

        images = torch.stack([x for x in images])

        return {
            "inputs": images,
            "targets": targets,
            "image_filepaths": image_filepaths,
            "annotation_filepaths": annotation_filepaths,
            "batch_size": images.shape[0]
        }
