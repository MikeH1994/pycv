from __future__ import annotations
from typing import List

class AverageMeter:
    """
    Class that computes the running average of some value
    """
    def __init__(self):
        self.val: float = 0.0  # the last value added
        self.sum: float = 0.0  # the sum of all values
        self.avg: float = 0.0  # the average of all values
        self.count: int = 0  # the number of values added

    def reset(self):
        """
        Resets the AverageMeter
        :return:
        """
        self.val = 0.0
        self.sum = 0.0
        self.avg = 0.0
        self.count = 0

    def update(self, val: float, count: int = 1):
        """
        Add more values to the AverageMeter
        :param val: the current value to be added - this could be a single value, or the mean of some set
        :param count: if passing the mean of some set, the number of samples used to generate the mean- else 1
        :return:
        """
        self.val = val
        self.sum += val * count
        self.count += count
        self.avg = self.sum / self.count


class MultiClassAverageMeter:
    """

    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.val: List[float] = [0.0 for _ in range(n_classes)]
        self.sum: List[float] = [0.0 for _ in range(n_classes)]  # the sum of all values
        self.avg: List[float] = [0.0 for _ in range(n_classes)]  # the average of all values
        self.count: int = 0  # the number of values added

    def reset(self):
        """
        Resets the AverageMeter
        :return:
        """
        self.val = [0.0 for _ in range(self.n_classes)]
        self.sum = [0.0 for _ in range(self.n_classes)]
        self.avg = [0.0 for _ in range(self.n_classes)]
        self.count = 0

    def update(self, val: List[float], count: int = 1):
        """
        Add more values to the AverageMeter
        :param val: the current value to be added - this could be a single value, or the mean of some set
        :param count: if passing the mean of some set, the number of samples used to generate the mean- else 1
        :return:
        """
        assert(len(val) == self.n_classes)
        self.val = val
        self.count += count
        for i in range(self.n_classes):
            self.sum[i] += val[i] * count
            self.avg[i] = self.sum[i] / self.count
