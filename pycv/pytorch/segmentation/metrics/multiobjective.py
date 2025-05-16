from __future__ import annotations
import torch.nn as nn
import torch
from .jaccard import JaccardLoss
from .focal import FocalLoss
from ...base.base_metrics import BaseLossFn, BaseScoreFn, check_assertions_segmentation


class MultiObjectiveLoss(BaseLossFn):
    def __init__(self, alpha: float = 0.9, gamma: float = 2.0, binary: bool = False):
        """
        :param alpha: weight of the Jaccard loss
        """
        super().__init__("Mulit-objective loss", check_assertions_segmentation)
        self.alpha = alpha
        self.binary = binary
        self.jaccard_loss_fn = JaccardLoss(soft_loss=True)
        self.focal_loss_fn = FocalLoss(gamma=gamma, binary=self.binary)

    def metric(self, pred_logits: torch.FloatTensor, targets: torch.FloatTensor) -> torch.FloatTensor:
        """
        Returns the multi-objective loss.
        :param pred_logits: the predicted logits for each pixel.
                            Shape is (B, num_classes, h, w).
        :param targets: the ground truth values. Shape is (B, h, w).
                        Values represent the class index,
                        and are integers in the range [0, num_classes-1]
        :return:
        """
        jaccard_loss = self.alpha * self.jaccard_loss_fn(pred_logits, targets)
        focal_loss = self.focal_loss_fn(pred_logits, targets)
        loss = jaccard_loss + focal_loss
        return loss
