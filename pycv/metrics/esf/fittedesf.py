from __future__ import annotations
from .esf import ESF
from scipy.optimize import curve_fit
from scipy.special import erf
from numpy.typing import NDArray
from typing import Union
import numpy as np

class FittedESF(ESF):
    def __init__(self, x_data: NDArray, f_data: NDArray, **kwargs):
        super().__init__(x_data, f_data)
        self.params = self.fit(x_data, f_data)

    def get_p0(self):
        raise Exception("Base function FittedESF.get_p0() called")

    def get_bounds(self):
        return None

    def fit(self, x_data, f_data, **kwargs) -> NDArray:
        p0 = self.get_p0()
        bounds = self.get_bounds()
        params, _ = curve_fit(self.fn, x_data, f_data, maxfev=40000, bounds=bounds, p0=p0)
        return params

    def f(self, x: Union[NDArray, float]) -> Union[NDArray, float]:
        return self.fn(x, *self.params)

    @staticmethod
    def fn(x, *params) -> NDArray:
        raise Exception("Base function FittedESF.fn() called")


class GaussianESF(ESF):
    def __init__(self,  x_data: NDArray, f_data: NDArray, **kwargs):
        self.n_terms = kwargs["n_terms"] if "n_terms" in kwargs else 4
        super().__init__(x_data, f_data)

    def get_p0(self):
        return np.ones(self.n_terms*2)

    @staticmethod
    def fn(x, *args):
        if len(args) % 2 != 0:
            raise ValueError("In ESF.fn(): the number of terms in the function must be even")
        f = 0.5
        for i in range(len(args)//2):
            a_i = args[i*2]
            b_i = args[i*2 + 1]
            f += a_i*erf(x/b_i)
        return f
