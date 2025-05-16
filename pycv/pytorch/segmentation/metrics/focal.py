from __future__ import annotations
import torch.nn as nn
import torch
from ...base.base_metrics import BaseLossFn, BaseScoreFn, check_assertions_segmentation


class FocalLoss(BaseLossFn):
    def __init__(self, gamma: float = 2.0, ignore_indices: int = -1, binary: bool = False):
        """
        :param gamma: parameter which control slope of loss function.
        :param ignore_indices: indices to ignore
        """
        super().__init__("Focal loss", check_assertions_segmentation)
        self.gamma = gamma
        self.ignore_indices = ignore_indices
        self.binary = binary
        if self.binary:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
            self.ignore_indices = -1
        else:
            self.loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=self.ignore_indices)

    def metric(self, pred_logits: torch.FloatTensor, targets: torch.FloatTensor) -> torch.FloatTensor:
        """
        Returns the focal loss.
        :param pred_logits: the predicted logits for each pixel. Shape is (B, num_classes, h, w).
        :param targets: the ground truth values. Shape is (B, h, w). Values represent the class index,
                        and are integers in the range [0, num_classes-1]
        :return: A single element tensor containing the loss
        """
        if self.binary:
            assert(pred_logits.shape[1] == 1), "Binary parameter passed, but more than 1 class"
            loss_ce = self.loss_fn(pred_logits.squeeze(1), targets.float())
        else:
            assert(pred_logits.shape[1] > 1), "Binary parameter not pass, but only 1 class found"
            loss_ce = self.loss_fn(pred_logits, targets.long())
        loss_focal = (1.0 - loss_ce.mul(-1).exp()).pow(self.gamma) * loss_ce
        if self.binary:
            return loss_focal.mean()
        else:
            return loss_focal[targets != self.ignore_indices].mean()
