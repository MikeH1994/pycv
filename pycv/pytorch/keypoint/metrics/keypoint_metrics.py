from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, Tuple, List
from ...base.base_metrics import BaseScoreFn, check_assertions_keypoints


class BaseKeypointScoreFn(BaseScoreFn):
    criterion: nn.Module

    def __init__(self, name: str):
        super().__init__(name, check_assertions_keypoints)

    def metric(self, output: torch.FloatTensor, targets: torch.FloatTensor) -> Tuple[float, List[float]]:
        """
        Returns the L1 score
        :param output: the predicted coordinates for each pixel. Shape is (B, num_keypoints*2).
        :param targets: the ground truth coordinates for each pixel. Shape is (B, num_keypoints*2).
        :return: A tuple containing:
        """
        loss_per_elem = torch.mean(self.criterion(output, targets), dim=0)
        loss = torch.mean(loss_per_elem).item()
        loss_per_point = (loss_per_elem[0::2] + loss_per_elem[1::2])/2.0

        return loss, loss_per_point.tolist()


class L1Score(BaseKeypointScoreFn):
    def __init__(self):
        super().__init__("L1 Loss")
        self.criterion = nn.L1Loss(reduction="none")


class SmoothL1Score(BaseKeypointScoreFn):
    def __init__(self):
        super().__init__("Smooth L1 Loss")
        self.criterion = nn.SmoothL1Loss(reduction="none")


class L2Score(BaseKeypointScoreFn):
    def __init__(self):
        super().__init__("L2 Loss")
        self.criterion = nn.MSELoss(reduction="none")

