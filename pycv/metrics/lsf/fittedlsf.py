from __future__ import annotations
from .lsf import LSF
from ..psf.psf import PSF
from ..psf.fittedpsf import GaussianPSF
from scipy.optimize import curve_fit
from scipy.special import erf
from numpy.typing import NDArray
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from pycv.utils.settings import FittingParams
from pycv.utils.maths import calculate_fwhm


class FittedLSF(LSF):
    def __init__(self, x_data: NDArray, f_data: NDArray, params: NDArray = None, fitting_params: FittingParams = None, **kwargs):
        super().__init__(x_data, f_data, **kwargs)
        self.params = params if params is not None else self.fit(self.x_data, self.f_data, fitting_params, **kwargs)

    def f(self, x: Union[NDArray, float]) -> Union[NDArray, float]:
        return self.fn(x, *self.params)

    def get_n_parameters(self):
        raise Exception("Base function FittedLSF.get_n_parameters() called")

    def get_p0(self):
        return np.ones(self.get_n_parameters())

    def get_bounds(self):
        return [(-np.inf, np.inf) for _ in range(self.get_n_parameters())]

    def fit(self, x_data, f_data, fitting_params: FittingParams = None, **kwargs) -> NDArray:
        if fitting_params is None:
            fitting_params = FittingParams()
        p0 = self.get_p0() if fitting_params.p0 is None else fitting_params.p0
        bounds = self.get_bounds() if fitting_params.bounds is None else fitting_params.bounds
        if fitting_params.clip_x_range:
            if fitting_params.x_range is None:
                cx, fwhm = calculate_fwhm(x_data, f_data)
                x_range = cx - 2*fwhm, cx + 2*fwhm
                fitting_params.x_range = x_range if x_range is not None else (np.min(x_data), np.max(x_data))
        x0, x1 = fitting_params.x_range
        f_data = f_data[(x_data >= x0) & (x_data <= x1)]
        x_data = x_data[(x_data >= x0) & (x_data <= x1)]

        try:
            params, _ = curve_fit(self.fn, x_data, f_data, maxfev=40000, p0=p0)
        except RuntimeError:
            params = p0
        return params

    def plot_elem(self, **kwargs):
        plt.scatter(self.x_data, self.f_data, label="LSF (Measured)")
        x_fitted = np.linspace(np.min(self.x_data), np.max(self.x_data), 1000)
        f_fitted = self.f(x_fitted)
        plt.plot(x_fitted, f_fitted, label="LSF (Fitted)")

    def fwhm(self):
        x_fitted = np.linspace(np.min(self.x_data), np.max(self.x_data), 1000)
        y_fitted = self.f(x_fitted)
        _, fwhm = calculate_fwhm(x_fitted, y_fitted)
        return fwhm

    @staticmethod
    def fn(x, *params) -> NDArray:
        raise Exception("Base function FittedLSF.fn() called")


class GaussianLSF(FittedLSF):
    def __init__(self,  x_data: NDArray, f_data: NDArray, params: NDArray = None, fitting_params: FittingParams = None, **kwargs):
        fitting_params = FittingParams() if fitting_params is None else fitting_params
        self.n_terms = fitting_params.n_terms if fitting_params is not None else 3
        super().__init__(x_data, f_data, params, fitting_params, **kwargs)

    def get_n_parameters(self):
        return self.n_terms*2

    @staticmethod
    def fn(x, *args):
        if len(args) % 2 != 0:
            raise ValueError("In LSF.fn(): the number of terms in the function must be even")
        f = 0.0
        for i in range(len(args)//2):
            a_i = args[i*2]
            b_i = args[i*2 + 1]
            f += 2.0 / np.sqrt(np.pi) * a_i / b_i * np.exp(-x**2 / b_i**2)
        return f

    """def esf(self) -> GaussianESF:
        return GaussianESF()"""

    def psf(self, **kwargs) -> GaussianPSF:
        height = 21 if "height" not in kwargs else kwargs["height"]
        width = 21 if "width" not in kwargs else kwargs["width"]
        return GaussianPSF(self, width, height)
