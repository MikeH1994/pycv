from __future__ import annotations
from .jaccard import *
from ...base.base_metrics import BaseLossFn, BaseScoreFn, check_assertions_segmentation


class IOU(BaseScoreFn):
    def __init__(self, epsilon=1e-5):
        """
        :param epsilon: parameter to avoid divide by zero errors.
        """
        super(IOU, self).__init__("IOU", check_assertions_segmentation)
        self.epsilon = epsilon
        self.jaccard_score = JaccardScore(self.epsilon, use_logs=False, reduce=True)

    def metric(self, pred_logits, targets) -> Tuple[float, List[float]]:
        """
        Returns the IOU, calculated through the Jaccard score.
        :param pred_logits: the predicted logits for each pixel.
                            Shape is (B, num_classes, h, w).
        :param targets: the ground truth values. Shape is (B, h, w).
                        Values represent the class index,
                        and are integers in the range [0, num_classes-1]
        :return:
        """
        return self.jaccard_score(pred_logits, targets)
