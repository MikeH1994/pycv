from __future__ import annotations
import torch.nn as nn
import torch
from typing import Tuple, List, Callable, Union


class BaseLossFn(nn.Module):
    """
    Base class for loss functions.
    """
    def __init__(self, name: str, check_assertions: Callable[[torch.FloatTensor, torch.FloatTensor], None] = None):
        super().__init__()
        self.name = name
        self.check_assertions = check_assertions

    def forward(self, pred_logits: torch.FloatTensor, targets: torch.FloatTensor) -> torch.FloatTensor:
        if self.check_assertions is not None:
            self.check_assertions(pred_logits, targets)
        return self.metric(pred_logits, targets)

    def metric(self, pred_logits: torch.FloatTensor, targets: torch.FloatTensor) -> torch.FloatTensor:
        raise Exception("Base Function called")


class BaseScoreFn(nn.Module):
    """
    Base class for other metric scores that returns a total score and a per-class score
    """
    def __init__(self, name: str, check_assertions: Callable[[torch.FloatTensor, torch.FloatTensor], None] = None):
        super().__init__()
        self.name = name
        self.check_assertions = check_assertions

    def forward(self, pred_logits: torch.FloatTensor, targets: torch.FloatTensor) -> Tuple[float,
                                                                                           Union[List[float], None]]:
        if self.check_assertions is not None:
            self.check_assertions(pred_logits, targets)
        return self.metric(pred_logits, targets)

    def metric(self, pred_logits: torch.FloatTensor, targets: torch.FloatTensor) -> Tuple[float,
                                                                                          Union[List[float], None]]:
        raise Exception("Base Function called")


def check_assertions_segmentation(pred_logits: torch.FloatTensor, targets: torch.FloatTensor):
    """
    Checks the inputs fpr pred_logits and targets are correct.
    :param pred_logits: the predicted logits for each pixel.
                        Shape is (B, num_classes, h, w).
    :param targets: the ground truth values. Shape is (B, h, w).
                    Values represent the class index,
                    and are integers in the range [0, num_classes-1]
    :return:
    """
    error_string = """Invalid input shapes. Expected pred_logits in the form (B, num_classes, h, w) and targets in 
    the form (B, h, w). Got {} and {} instead""".format(pred_logits.shape, targets.shape)
    assert (len(pred_logits.shape) == 4), error_string
    assert (len(targets.shape) == 3), error_string
    assert (pred_logits.shape[0] == targets.shape[0]), error_string
    assert (pred_logits.shape[2] == targets.shape[1]), error_string
    assert (pred_logits.shape[3] == targets.shape[2]), error_string
    assert (torch.is_floating_point(pred_logits)), "Supplied params must be float"
    assert (torch.is_floating_point(targets)), "Supplied params must be float"
    return


def check_assertions_keypoints(pred_logits: torch.FloatTensor, targets: torch.FloatTensor):
    pass