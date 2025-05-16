from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Union
from ...base.base_metrics import BaseLossFn, BaseScoreFn, check_assertions_keypoints


class KeypointRCNNClassifierLoss(BaseLossFn):
    def __init__(self):
        super().__init__("Keypoint RCNN loss_classifier", check_assertions_keypoints)

    def metric(self, output: Dict, targets: torch.FloatTensor) -> Tuple[torch.FloatTensor, Union[List[float], None]]:
        """
        Returns the combined loss for all the different components returned by the KeypointRCNN
        :param pred_logits: the predicted logits for each pixel. Shape is (B, num_classes, h, w).
        :param targets: the ground truth values. Shape is (B, h, w). Values represent the class index,
                        and are integers in the range [0, num_classes-1]
        :return:
        """
        return output["loss_classifier"].detach()


class KeypointRCNNBoxRegLoss(BaseLossFn):
    def __init__(self):
        super().__init__("Keypoint RCNN loss_box_reg", check_assertions_keypoints)

    def metric(self, output: Dict, targets: torch.FloatTensor) -> Tuple[torch.FloatTensor, Union[List[float], None]]:
        """
        Returns the combined loss for all the different components returned by the KeypointRCNN
        :param pred_logits: the predicted logits for each pixel. Shape is (B, num_classes, h, w).
        :param targets: the ground truth values. Shape is (B, h, w). Values represent the class index,
                        and are integers in the range [0, num_classes-1]
        :return:
        """
        return output["loss_box_reg"].detach()


class KeypointRCNNKeypointLoss(BaseLossFn):
    def __init__(self):
        super().__init__("Keypoint RCNN loss_keypoint", check_assertions_keypoints)

    def metric(self, output: Dict, targets: torch.FloatTensor) -> Tuple[torch.FloatTensor, Union[List[float], None]]:
        """
        Returns the combined loss for all the different components returned by the KeypointRCNN
        :param pred_logits: the predicted logits for each pixel. Shape is (B, num_classes, h, w).
        :param targets: the ground truth values. Shape is (B, h, w). Values represent the class index,
                        and are integers in the range [0, num_classes-1]
        :return:
        """
        return output["loss_keypoint"].detach()


class KeypointRCNNObjectnessLoss(BaseLossFn):
    def __init__(self):
        super().__init__("Keypoint RCNN loss_objectness", check_assertions_keypoints)

    def metric(self, output: Dict, targets: torch.FloatTensor) -> Tuple[torch.FloatTensor, Union[List[float], None]]:
        """
        Returns the combined loss for all the different components returned by the KeypointRCNN
        :param pred_logits: the predicted logits for each pixel. Shape is (B, num_classes, h, w).
        :param targets: the ground truth values. Shape is (B, h, w). Values represent the class index,
                        and are integers in the range [0, num_classes-1]
        :return:
        """
        return output["loss_objectness"].detach()


class KeypointRCNNRPNBoxRegLoss(BaseLossFn):
    def __init__(self):
        super().__init__("Keypoint RCNN loss_objectness", check_assertions_keypoints)

    def metric(self, output: Dict, targets: torch.FloatTensor) -> Tuple[torch.FloatTensor, Union[List[float], None]]:
        """
        Returns the combined loss for all the different components returned by the KeypointRCNN
        :param pred_logits: the predicted logits for each pixel. Shape is (B, num_classes, h, w).
        :param targets: the ground truth values. Shape is (B, h, w). Values represent the class index,
                        and are integers in the range [0, num_classes-1]
        :return:
        """
        return output["loss_box_reg"].detach()
