from __future__ import annotations
from .lsf import LSF
from scipy.optimize import curve_fit
from scipy.special import erf
from numpy.typing import NDArray
from typing import Union
import numpy as np


class FittedLSF(LSF):
    def __init__(self, x_data: NDArray, f_data: NDArray, params: NDArray):
        super().__init__(x_data, f_data)
        self.params = params

    def f(self, x: Union[NDArray, float]) -> Union[NDArray, float]:
        return self.fn(x, *self.params)

    @staticmethod
    def fn(x, *params) -> NDArray:
        raise Exception("Base function FittedESF.fn() called")


class GaussianLSF(LSF):
    def __init__(self,  x_data: NDArray, f_data: NDArray, params: NDArray, **kwargs):
        super().__init__(x_data, f_data)
        self.params = params
        assert(len(params.shape) == 1 and params.shape[0] % 2 == 0)

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
