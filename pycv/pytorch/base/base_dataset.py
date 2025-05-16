from __future__ import annotations
import glob
import os
import albumentations as albumentations
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from albumentations.core.composition import Compose
from torch.utils.data import Dataset
from typing import Tuple, Callable, Union
from numpy.typing import NDArray
from torch.utils.data import DataLoader
import warnings
from ..core.configuration import Configuration, ImageTransformParams
from ..core.utils import convert_numpy_to_8_bit
from ..core.image_transforms import get_image_transforms
from ..core.utils import calc_mean_std


class BaseDataset(Dataset):
    def __init__(self, use_keypoints=False, use_bboxes=False):
        """

        :param config:
        :param img_transform:
        """
        self.use_keypoints = use_keypoints
        self.use_bboxes = use_bboxes
        self.initialise_transforms(None, normalize=False)
        self.initialise()
        if len(self) == 0:
            warnings.warn("Dataset is empty")

    def load_img(self, fpath: str, min_value: float = None, max_value: float = None) -> NDArray:
        """

        :param fpath:
        :param min_value:
        :param max_value:
        :return:
        """
        if fpath.endswith(".npy"):
            img = np.load(fpath)
        else:
            img = cv2.imread(fpath, -1)
        return convert_numpy_to_8_bit(img, min_val=min_value, max_val=max_value)

    def initialise(self):
        raise Exception("Base function initialise() has not been implemented")

    def to_string(self, padding=4):
        """

        :param padding:
        :return:
        """
        dst = 'Dataset length: {}'.format(len(self))
        for transform in self.img_transform.transforms:
            dst += '  \n' + '  ' * padding + str(transform)
        return dst

    def initialise_transforms(self, image_transform_params: Union[ImageTransformParams, None], normalize: bool = True,
                             augment_data: bool = False):
        self.img_transform =  get_image_transforms(image_transform_params, normalize=normalize, augment_data=augment_data,
                                    use_bboxes=self.use_bboxes, use_keypoints=self.use_keypoints)

    @staticmethod
    def collate_fn(batch):
        raise Exception("collate_fn should be overwritten in class inherinting BaseDataset")

    def clip_dataset(self, length: int):
        raise Exception("clip_dataset should be overwritten in class inherinting BaseDataset")

    def get_raw_image(self, idx: int):
        raise Exception("get_raw_image should be overwritten in class inherinting BaseDataset")

    def __getitem__(self, idx):
        raise Exception("__getitem__ should be overwritten in class inherinting BaseDataset")

    def __len__(self):
        raise Exception("__len__ should be overwritten in class inherinting BaseDataeset")
