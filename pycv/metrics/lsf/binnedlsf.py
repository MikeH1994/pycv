import numpy as np
from numpy.typing import NDArray
from typing import Union, Dict
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
from ...utils.matlab import matlab_round
from .lsf import LSF
from .fittedlsf import GaussianLSF, FittedLSF
from ..mtf.core import compute_mtf, fir2fix
from ...metrics.mtf import MTF
from pycv.utils.settings import FittingParams


class BinnedLSF(LSF):
    def __init__(self, data: Dict, **kwargs):
        super().__init__(data["x"], np.gradient(data["val"], data["x"]), **kwargs)
        self.data = data
        self.data["val"] = self.f_data
        self.bin_std = self.data["std"]
        self.bin_range = self.data["range"]
        self.bin_width = self.data["bin_width"]
        self.bins_per_pixel = self.data["bins_per_pixel"]
        self.interpolation_fn = InterpolatedUnivariateSpline(self.x_data, self.f_data, k=1, ext=1)

    def plot_elem(self, **kwargs):
        # plt.errorbar(self.x_data, self.f_data, yerr=self.bin_std, ls="none", marker='o', capsize=5)
        label = kwargs["label"] if "label" in kwargs else "LSF"
        plt.scatter(self.x_data, self.f_data, label=label)

    def f(self, x: Union[NDArray, float]) -> Union[NDArray, float]:
        return self.interpolation_fn(x)

    def fit(self, fitting_mode="gaussian", fitting_params: FittingParams = FittingParams()) -> FittedLSF:
        if fitting_mode == "gaussian":
            return GaussianLSF(self.data["x"], self.data["val"], fitting_params=fitting_params)
        else:
            raise Exception("Unknown ")

    def psf(self):
        raise Exception("Base function BinnedLSF.psf() called")

    def mtf(self, apply_fir_correction=True):
        # calculate mtf
        mtf = compute_mtf(self.f_data, self.bin_width)
        # correct the mtf for the impact that the discrete derivative has on the frequencies
        if apply_fir_correction:
            dcorr = fir2fix(mtf.shape[0], 3)
            mtf[:, 1] *= dcorr
        # limit the number of frequencies, as in ISO 12233
        freq_lim = 2 if self.bins_per_pixel == 1 else 1
        n_outputted_frequencies = matlab_round(mtf.shape[0] * freq_lim / 2)
        mtf = mtf[:n_outputted_frequencies]
        return MTF(mtf)
