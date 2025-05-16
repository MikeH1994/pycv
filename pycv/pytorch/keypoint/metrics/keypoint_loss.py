from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict
from ...base.base_metrics import BaseLossFn, check_assertions_keypoints


class L1Loss(BaseLossFn):
    criterion: nn.Module

    def __init__(self):
        super().__init__("L1 Loss", check_assertions_keypoints)
        self.criterion = nn.L1Loss(reduction="mean")

    def metric(self, output: torch.FloatTensor, targets: torch.FloatTensor) -> torch.Tensor:
        """
        Returns the L1 loss
        :param output: the predicted coordinates for each pixel. Shape is (B, num_keypoints*2).
        :param targets: the ground truth coordinates for each pixel. Shape is (B, num_keypoints*2).
        :return: A single element tensor containing the loss
        """

        return self.criterion(output, targets)


class SmoothL1Loss(BaseLossFn):
    criterion: nn.Module

    def __init__(self):
        super().__init__("Smooth L1 Loss", check_assertions_keypoints)
        self.criterion = nn.SmoothL1Loss(reduction="mean")

    def metric(self, output: torch.FloatTensor, targets: torch.FloatTensor) -> torch.Tensor:
        """
        Returns the smooth L1 loss
        :param output: the predicted coordinates for each pixel. Shape is (B, num_keypoints*2).
        :param targets: the ground truth coordinates for each pixel. Shape is (B, num_keypoints*2).
        :return: A single element tensor containing the loss
        """

        return self.criterion(output, targets)


class L2Loss(BaseLossFn):
    def __init__(self):
        super().__init__("L2 Loss", check_assertions_keypoints)
        self.criterion = nn.MSELoss(reduction="mean")

    def metric(self, output: torch.FloatTensor, targets: torch.FloatTensor) -> torch.Tensor:
        """
        Returns the combined loss for all the different components returned by the KeypointRCNN
        :param output: the predicted logits for each pixel. Shape is (B, num_classes, h, w).
        :param targets: the ground truth values. Shape is (B, h, w). Values represent the class index,
                        and are integers in the range [0, num_classes-1]
        :return: A single element tensor containing the loss
        """

        return self.criterion(output, targets)
