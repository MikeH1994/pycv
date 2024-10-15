from __future__ import annotations
from pycv.metrics.esf.esf import ESF
from scipy.optimize import curve_fit
from scipy.special import erf
from numpy.typing import NDArray
from typing import Union
from .utils import bin_data
import numpy as np
import scipy.interpolate
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
from ...metrics.lsf.binnedlsf import BinnedLSF


class BinnedESF(ESF):
    def __init__(self, x_data: NDArray, f_data: NDArray, roi_size=None, bins_per_pixel=4, zero_centered=True, **kwargs):
        super().__init__(x_data, f_data, **kwargs)
        self.roi_size = roi_size
        self.bins_per_pixel = bins_per_pixel
        self.data = bin_data(x_data, f_data, bins_per_pixel=bins_per_pixel, zero_centered=zero_centered)
        self.bin_centres = self.data["x"]
        self.bin_values = self.data["val"]
        self.bin_std = self.data["std"]
        self.bin_range = self.data["range"]
        self.bin_width = self.data["bin_width"]
        self.interpolation_fn = InterpolatedUnivariateSpline(self.bin_centres, self.bin_values, k=1, ext=1)

    def f(self, x: Union[NDArray, float]) -> Union[NDArray, float]:
        return self.interpolation_fn(x)

    def lsf(self, **kwargs):
        return BinnedLSF(self.data, **kwargs)

    def normalise_data(self):
        raise Exception("Base function ESF.lsf() called")

    def plot_elem(self, **kwargs):
        plt.errorbar(self.bin_centres, self.bin_values, yerr=self.bin_std, ls="none", marker='o', capsize=5)


