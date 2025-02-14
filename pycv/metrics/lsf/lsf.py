from __future__ import annotations
from numpy.typing import NDArray
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.integrate import simpson
from ...metrics.mtf import MTF
from ...metrics.metric import Metric
from ...utils.windows import tukey_fn, ahamming_fn


class LSF(Metric):
    def __init__(self, x_data: NDArray, f_data: NDArray, **kwargs):
        wflag = kwargs["wflag"] if "wflag" in kwargs else 0
        normalise = kwargs["normalise"] if "normalise" in kwargs else True

        self.x_data = x_data
        self.f_data = f_data
        self.default_x_label = "Distance to edge (px)"
        self.default_y_label = "LSF"
        self.params = None

        width = math.ceil(np.max(x_data) - np.min(x_data))
        if wflag == 0:
            alpha = kwargs["alpha"] if wflag in kwargs else 1.0
            smoothing_fn = tukey_fn(width, alpha, 0.0)
            self.f_data *= 0.95*smoothing_fn(self.x_data) + 0.05
        elif wflag == 1:
            smoothing_fn = ahamming_fn(width, 0.0)
            self.f_data *= smoothing_fn(self.x_data)
        if normalise:
            integral = simpson(self.f_data, self.x_data)
            self.f_data /= integral

    def plot_elem(self, **kwargs):
        label = kwargs["label"] if "label" in kwargs else "LSF"
        plt.scatter(self.x_data, self.f_data, label=label)

    def f(self, x: Union[NDArray, float]) -> Union[NDArray, float]:
        raise Exception("Base function LSF.f() called")

    def esf(self):
        raise Exception("Base function LSF.lsf() called")

    def psf(self):
        raise Exception("Base function LSF.lsf() called")

    def mtf(self, **kwargs) -> MTF:
        raise Exception("Base function LSF.lsf() called")
