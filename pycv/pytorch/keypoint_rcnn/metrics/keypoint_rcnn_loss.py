from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict
from ...base.base_metrics import BaseLossFn, check_assertions_keypoints


class KeypointRCNNLoss(BaseLossFn):
    def __init__(self):
        super().__init__("Keypoint RCNN Loss", check_assertions_keypoints)

    def metric(self, output: Dict, targets: torch.FloatTensor) -> torch.Tensor:
        """
        Returns the combined loss for all the different components returned by the KeypointRCNN
        :param pred_logits: the predicted logits for each pixel. Shape is (B, num_classes, h, w).
        :param targets: the ground truth values. Shape is (B, h, w). Values represent the class index,
                        and are integers in the range [0, num_classes-1]
        :return:
        """
        loss = [loss for loss in output.values()]
        loss = torch.stack(loss, dim=0)
        loss = torch.sum(loss, dim=0)
        return loss
