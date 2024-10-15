from __future__ import annotations
from .lsf import LSF
from scipy.optimize import curve_fit
from scipy.special import erf
from numpy.typing import NDArray
from typing import Union
import numpy as np


class FittedLSF(LSF):
    def __init__(self, x_data: NDArray, f_data: NDArray, **kwargs):
        super().__init__(x_data, f_data, **kwargs)
        self.params = self.fit(self.x_data, self.f_data, **kwargs)

    def f(self, x: Union[NDArray, float]) -> Union[NDArray, float]:
        return self.fn(x, *self.params)

    def get_p0(self):
        raise Exception("Base function FittedLSF.get_p0() called")

    def get_bounds(self):
        return [(-np.inf, np.inf) for _ in range(self.get_p0().shape[0])]

    def fit(self, x_data, f_data, **kwargs) -> NDArray:
        p0 = self.get_p0()
        bounds = self.get_bounds()
        try:
            params, _ = curve_fit(self.fn, x_data, f_data, maxfev=40000, p0=p0, bounds=bounds)
        except RuntimeError:
            params = p0
        return params

    @staticmethod
    def fn(x, *params) -> NDArray:
        raise Exception("Base function FittedLSF.fn() called")


class GaussianLSF(LSF):
    def __init__(self,  x_data: NDArray, f_data: NDArray, **kwargs):
        super().__init__(x_data, f_data, **kwargs)

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
