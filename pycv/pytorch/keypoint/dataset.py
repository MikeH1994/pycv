from __future__ import annotations
from albumentations.core.composition import Compose
from typing import List, Tuple, Dict
import numpy as np
from numpy.typing import NDArray
import torch
import cv2
from ..base.base_dataset import BaseDataset
from ..core.configuration import KeypointRCNNConfiguration


class KeypointDataset(BaseDataset):
    """
    Generic class for keypoints
    """
    image_fpaths: List[str] = None
    annotation_fpaths: List[str] = None

    def __init__(self, num_keypoints):
        self.num_keypoints=num_keypoints
        super().__init__(use_keypoints=True, use_bboxes=False)

    def initialise(self):
        """
        The implementation of initialise() from inheriting classes should initialise two class variables- image_fpaths,
        and annotations_fpaths. image_fpaths and annotatio
        image_fpaths: List[str] = None
        annotations_fpaths: List[str] = None
        :return:
        """
        raise Exception("Base function initialise() has not been implemented")

    def load_annotations(self, keypoints_filepath) -> NDArray:
        """
        :param keypoints_filepath:
        :return:
        """
        raise Exception("Base function load_annotations(annotations_filepath) has not been implemented")

    def process_image(self, image: NDArray) -> NDArray:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image

    def clip_dataset(self, length: int):
        assert(self.image_fpaths is not None), "image_fpaths should be initialised in initialise()"
        assert(self.annotation_fpaths is not None), "annotations_fpaths should be initialised in initialise()"

        if length <= 0:
            return

        self.image_fpaths = self.image_fpaths[:length]
        self.annotation_fpaths = self.annotation_fpaths[:length]

    def __getitem__(self, idx):
        """

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
        keypoints = self.load_annotations(annotations_fpath)
        assert(len(keypoints) == self.num_keypoints)

        f = self.img_transform(image=img, keypoints=keypoints)
        img = f["image"].float()
        keypoints = torch.tensor(f["keypoints"]).reshape(-1)

        return {
            "input": img,
            "target": keypoints,
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
        targets = torch.stack([x for x in targets])

        return {
            "inputs": images,
            "targets": targets,
            "image_filepaths": image_filepaths,
            "annotation_filepaths": annotation_filepaths,
            "batch_size": images.shape[0]
        }
