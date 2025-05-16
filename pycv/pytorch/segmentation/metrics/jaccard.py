from __future__ import annotations
import torch
from ...base.base_metrics import BaseLossFn, BaseScoreFn, check_assertions_segmentation
from ..utils import get_pixel_predictions_from_output
from typing import Tuple, List


class JaccardScore(BaseScoreFn):
    def __init__(self, epsilon=1e-5, use_logs=False, reduce=False):
        """
        :param epsilon: parameter to avoid divide by zero errors.
        :param use_logs: apply logs to score before return mean and mean per class
        :param reduce: reduce the pixel probabilities per class (B,n,h,w) to predicted class (B,h,w)
        """
        super().__init__("Jaccard score", check_assertions_segmentation)
        self.epsilon = epsilon
        self.use_logs = use_logs
        self.reduce = reduce

    def metric(self, pred_logits: torch.FloatTensor, targets: torch.FloatTensor) -> Tuple[torch.FloatTensor,
                                                                                          List[float]]:
        """
        Returns the Jaccard score.
        :param pred_logits: the predicted logits for each pixel.
                            Shape is (B, num_classes, h, w).
        :param targets: the ground truth values. Shape is (B, h, w).
                        Values represent the class index,
                        and are integers in the range [0, num_classes-1]
        :return:
        """
        num_classes = pred_logits.shape[1]
        mean_score = torch.FloatTensor([0.0])
        score_per_class: List[float] = []
        if self.reduce:
            preds = get_pixel_predictions_from_output(pred_logits)
        else:
            preds = torch.softmax(pred_logits, dim=1)
        for class_index in range(num_classes):
            if self.reduce:
                pred = (preds == class_index).float()
            else:
                pred = preds[:, class_index]
            target = (targets == class_index).float()
            score = JaccardScore.calc_jaccard_score(pred, target, epsilon=self.epsilon)
            if self.use_logs:
                score = torch.log(score)
            score_per_class.append(score.item())
            mean_score += score
        mean_score /= num_classes
        return mean_score, score_per_class

    @staticmethod
    def calc_jaccard_score(pred, target, epsilon=1e-5) -> torch.FloatTensor:
        """
        Calculates the Jaccard score, also known as Intersection Over Union (IOU),
        for a single class.
        :param pred: the probability each pixel belongs to this class class.
                     Shape is (B, h, w).
                     Values are probabilities in range [0, 1]
        :param target: the ground truth values, indicating if the pixel belongs to this class.
                       Shape is (B, h, w). Values are 0 or 1.
        :param epsilon: parameter to avoid divide by zero errors.
        :return:
        """
        pred = pred.flatten()
        target = target.flatten()
        intersection = (pred * target).sum()
        return (intersection + epsilon) / (pred.sum() + target.sum() - intersection + epsilon)


class JaccardLoss(BaseLossFn):
    def __init__(self, epsilon=1e-5, soft_loss=True):
        """
        :param epsilon: parameter to avoid divide by zero errors.
        """
        super().__init__("Jaccard loss", check_assertions_segmentation)
        self.epsilon = epsilon
        self.soft_loss = soft_loss
        use_logs = True if self.soft_loss else False
        self.jaccard_score = JaccardScore(self.epsilon, use_logs=use_logs, reduce=False)

    def metric(self, pred_logits, targets) -> torch.FloatTensor:
        """
        Returns the Jaccard loss.
        :param pred_logits: the predicted logits for each pixel.
                            Shape is (B, num_classes, h, w).
        :param targets: the ground truth values. Shape is (B, h, w).
                        Values represent the class index,
                        and are integers in the range [0, num_classes-1]
        :return: A single element tensor containing the loss
        """
        jaccard_score, _ = self.jaccard_score(pred_logits, targets)
        if self.soft_loss:
            return -jaccard_score
        else:
            return 1.0 - jaccard_score
