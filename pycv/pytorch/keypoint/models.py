from __future__ import annotations
import torch.nn as nn
import ssl
import ssl
import torch
import numpy as np
import torchvision
from collections import OrderedDict
from typing import Dict, List, Union
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
from ..base.base_model import BaseModel


class KeypointModel(BaseModel):
    def __init__(self, name: str, n_keypoints):
        super().__init__(name)
        self.n_keypoints = n_keypoints

    def forward(self, x: torch.FloatTensor):
        raise Exception("Base class forward(x) not implemented")

    def process_predictions(self, preds: torch.FloatTensor):
        raise Exception("Base class process_predictions(x) not implemented")


class ResNetKeypointModel(KeypointModel):
    def __init__(self, num_keypoints, n_dense_layers=1):
        super().__init__("Resnet Keypoint model", num_keypoints)
        ssl._create_default_https_context = ssl._create_unverified_context()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        fc = []
        for layer_index in range(1, n_dense_layers):
            fc.append(nn.Linear(2048, 2048))
            fc.append(nn.ReLU(inplace=True))
        fc.append(nn.Linear(2048, num_keypoints*2))
        self.model.fc = nn.Sequential(*fc)
        ssl._create_default_https_context = ssl.create_default_context()



    def forward(self, x: torch.FloatTensor):
        """
        :param x:
        :return:
        """
        return self.model(x)

    def process_predictions(self, x: torch.FloatTensor):
        return x
    