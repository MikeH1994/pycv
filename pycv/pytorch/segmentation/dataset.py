from __future__ import annotations
from albumentations.core.composition import Compose
from typing import List
import numpy as np
from numpy.typing import NDArray
import torch
import cv2
from ..base.base_dataset import BaseDataset
from ..core.configuration import SegmentationConfiguration



class SegmentationDataset(BaseDataset):
    """
    Generic class for
    """
    image_fpaths: List[str] = None
    mask_fpaths: List[str] = None
    n_classes: int

    def __init__(self, n_classes: int):
        super().__init__(use_keypoints=False, use_bboxes=False)
        self.n_classes = n_classes

    def initialise(self):
        raise Exception("Base function initialise() has not been implemented")

    def process_image(self, image: NDArray) -> NDArray:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image

    def process_mask(self, mask: NDArray) -> NDArray:
        return mask

    def clip_dataset(self, length: int):
        assert(self.image_fpaths is not None), "image_fpaths should be initialised in initialise()"
        assert(self.mask_fpaths is not None), "mask_fpaths should be initialised in initialise()"

        if length <= 0:
            return

        self.image_fpaths = self.image_fpaths[:length]
        self.mask_fpaths = self.mask_fpaths[:length]

    def __getitem__(self, idx):
        assert(self.image_fpaths is not None), "image_fpaths should be initialised in initialise()"
        assert(self.mask_fpaths is not None), "mask_fpaths should be initialised in initialise()"

        image_fpath = self.image_fpaths[idx]
        mask_fpath = self.mask_fpaths[idx]

        img = self.load_img(image_fpath)
        img = self.process_image(img)
        mask = self.load_img(mask_fpath)
        mask = self.process_mask(mask)

        assert(img.shape[:2] == mask.shape[:2]), "Image and mask do not have the same dimensions"
        assert(len(img.shape) == 3 and img.shape[2] == 3), "The input image should be 3 channel"
        assert(len(mask.shape) == 2 and mask.dtype == np.uint8), "Mask should be uint8"
        if self.n_classes == 1:
            assert(np.max(mask) <= 1)
        else:
            assert(np.max(mask) < self.n_classes)

        f = self.img_transform(image=img, mask=mask)
        img = f["image"].float()
        mask = f["mask"].float()

        return {
            "input": img,
            "target": mask,
            "image_filepath": image_fpath,
            "mask_filepath": mask_fpath,
        }

    def __len__(self):
        return len(self.image_fpaths)

    def get_raw_image(self, idx: int):
        image_fpath = self.image_fpaths[idx]
        return self.process_image(self.load_img(image_fpath))

    @staticmethod
    def collate_fn(batch: List):
        images = []
        masks = []
        image_filepaths = []
        mask_filepaths = []
        for x in batch:
            skip = True if None in [x["input"], x["target"], x["image_filepath"], x["mask_filepath"]] else False
            if skip is False:
                images.append(x["input"])
                masks.append(x["target"])
                image_filepaths.append(x["image_filepath"])
                mask_filepaths.append(x["mask_filepath"])

        images = torch.stack([x for x in images])
        masks = torch.stack([x for x in masks])

        return {
            "inputs": images,
            "targets": masks,
            "image_filepaths": image_filepaths,
            "mask_filepaths": mask_filepaths,
            "batch_size": images.shape[0]
        }
