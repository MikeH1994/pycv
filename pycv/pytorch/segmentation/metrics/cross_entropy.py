from __future__ import annotations
import torch.nn as nn
from ...base.base_metrics import BaseLossFn, check_assertions_segmentation
import torch


class CrossEntropyLoss(BaseLossFn):
    def __init__(self, ignore_indices=-1, binary=False, reduce=True):
        super().__init__("Cross entropy loss", check_assertions_segmentation)
        self.ignore_indices = ignore_indices
        self.binary = binary
        reduction = 'mean' if reduce else 'none'
        if self.binary:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction=reduction)
        else:
            self.loss_fn = nn.CrossEntropyLoss(reduction=reduction, ignore_index=self.ignore_indices)

    def metric(self, pred_logits: torch.FloatTensor, targets: torch.FloatTensor) -> torch.FloatTensor:
        """
        Returns the cross entropy loss.
        :param pred_logits: the predicted logits for each pixel. Shape is (B, num_classes, h, w).
        :param targets: the ground truth values. Shape is (B, h, w). Values represent the class index,
                        and are integers in the range [0, num_classes-1]
        :return: A single element tensor containing the loss
        """
        if self.binary:
            return self.loss_fn(pred_logits.squeeze(1), targets)
        else:
            return self.loss_fn(pred_logits, targets.long())
