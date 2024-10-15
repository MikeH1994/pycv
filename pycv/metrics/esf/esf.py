from __future__ import annotations
from numpy.typing import NDArray
from typing import Union
import matplotlib.pyplot as plt
from ...metrics.lsf.lsf import LSF
from ...metrics.mtf import MTF
from pycv.metrics.metric import Metric


class ESF(Metric):
    def __init__(self, x_data: NDArray, f_data: NDArray, normalise=False):
        self.x_data = x_data
        self.f_data = f_data
        self.default_x_label = "Distance to edge (px)"
        self.default_y_label = "ESF"
        if normalise:
            self.x_data, self.f_data = self.normalise_data(self.x_data, self.f_data)

    def normalise_data(self, x_data, f_data):
        raise Exception("Base function ESF.lsf() called")

    def f(self, x: Union[NDArray, float]) -> Union[NDArray, float]:
        raise Exception("Base function ESF.f() called")

    def lsf(self, **kwargs) -> LSF:
        raise Exception("Base function ESF.lsf() called")

    def mtf(self, **kwargs) -> MTF:
        return self.lsf().mtf(**kwargs)

    def plot_elem(self, **kwargs):
        label = kwargs["label"] if "label" in kwargs else "ESF"
        plt.scatter(self.x_data, self.f_data, label=label)

