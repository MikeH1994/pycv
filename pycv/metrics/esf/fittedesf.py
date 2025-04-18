from __future__ import annotations
from dataclasses import dataclass
from .esf import ESF
from scipy.optimize import curve_fit
from scipy.special import erf
from numpy.typing import NDArray
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import pycv.metrics.lsf
from pycv.metrics.mtf import MTF
from pycv.utils.settings import FittingParams
from pycv.utils.maths import calculate_bounds_based_on_fwhm


class FittedESF(ESF):
    params: NDArray = None

    def __init__(self, x_data: NDArray, f_data: NDArray, params=None, fitting_params: FittingParams = None, **kwargs):
        super().__init__(x_data, f_data)
        self.params = self.fit(x_data, f_data, fitting_params) if params is None else params

    def plot_elem(self, **kwargs):
        plt.scatter(self.x_data, self.f_data, label="ESF (Measured)")
        #x_fitted = np.linspace(np.min(self.x_data), np.max(self.x_data), 1000)
        #f_fitted = self.f(x_fitted)
        #plt.plot(x_fitted, f_fitted, label="Fitted")
        plt.plot()

    def get_p0(self):
        raise Exception("Base function FittedESF.get_p0() called")

    def get_bounds(self):
        return [(-np.inf, np.inf) for _ in range(self.get_p0().shape[0])]

    def fit(self, x_data, f_data, fitting_params: FittingParams = None, **kwargs) -> NDArray:
        if fitting_params is None:
            fitting_params = FittingParams()
        p0 = self.get_p0() if fitting_params.p0 is None else fitting_params.p0
        bounds = self.get_bounds() if fitting_params.bounds is None else fitting_params.bounds
        if fitting_params.clip_x_range:
            if fitting_params.x_range is None:
                x_range = calculate_bounds_based_on_fwhm(x_data, f_data, fitting_params.clip_x_range_k)
                fitting_params.x_range = x_range if x_range is not None else (np.min(x_data), np.max(x_data))
        x0, x1 = fitting_params.x_range
        f_data = f_data[(x_data >= x0) & (x_data <= x1)]
        x_data = x_data[(x_data >= x0) & (x_data <= x1)]

        try:
            params, _ = curve_fit(self.fn, x_data, f_data, maxfev=40000, p0=p0)
        except RuntimeError:
            params = p0
        return params

    def f(self, x: Union[NDArray, float]) -> Union[NDArray, float]:
        return self.fn(x, *self.params)

    @staticmethod
    def calc_n_params(self):
        raise Exception("Base function FittedESF.calc_n_params() called")

    @staticmethod
    def fn(x, *params) -> NDArray:
        raise Exception("Base function FittedESF.fn() called")


class GaussianESF(FittedESF):
    def __init__(self,  x_data: NDArray, f_data: NDArray, params=None, fitting_params: FittingParams = None, **kwargs):
        fitting_params = FittingParams() if fitting_params is None else fitting_params
        self.n_terms = fitting_params.n_terms if params is None else self.calc_n_params(params)
        super().__init__(x_data, f_data, params=params)

    def get_p0(self):
        return np.zeros(self.n_terms*2)

    def lsf(self, **kwargs):
        return pycv.metrics.lsf.GaussianLSF(self.x_data, self.f_data, **kwargs)

    @staticmethod
    def fn(x, *args):
        if len(args) % 2 != 0:
            raise ValueError("In ESF.fn(): the number of terms in the function must be even")
        f = 0.5
        c = args[-2]
        d = args[-1]
        for i in range(len(args)//2 - 1):
            a_i = args[i*2]
            b_i = args[i*2 + 1]
            f += a_i*erf(x/b_i + c) + d
        return f
