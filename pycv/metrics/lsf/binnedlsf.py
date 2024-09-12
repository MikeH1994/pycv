import numpy as np
from numpy.typing import NDArray
from typing import Union, Dict
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
from ...utils.matlab import matlab_round
from .lsf import LSF
from ..mtf.core import compute_mtf


class BinnedLSF(LSF):
    def __init__(self, data: Dict):
        super().__init__(None, None)
        self.data = data
        self.bin_centres = self.data["x"]
        self.bin_values = np.gradient(self.data["val"], self.bin_centres)
        self.bin_std = self.data["std"]
        self.bin_range = self.data["range"]
        self.bin_width = self.data["bin_width"]
        self.bins_per_pixel = self.data["bins_per_pixel"]
        self.interpolation_fn = InterpolatedUnivariateSpline(self.bin_centres, self.bin_values, k=1, ext=1)

    def plot_elem(self, **kwargs):
        plt.errorbar(self.bin_centres, self.bin_values, yerr=self.bin_std, ls="none", marker='o', capsize=5)

    def f(self, x: Union[NDArray, float]) -> Union[NDArray, float]:
        return self.interpolation_fn(x)

    def psf(self):
        raise Exception("Base function LSF.lsf() called")

    def mtf(self):
        # calculate mtf
        mtf = compute_mtf(self.bin_values, self.bin_width)
        # limit the number of frequencies, as in ISO 12233
        freq_lim = 2 if self.bins_per_pixel == 1 else 1
        n_outputted_frequencies = matlab_round(mtf.shape[0] * freq_lim / 2)
        mtf = mtf[:n_outputted_frequencies]
        return mtf
