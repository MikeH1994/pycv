from __future__ import annotations
from .esf import ESF
from scipy.optimize import curve_fit
from scipy.special import erf
from numpy.typing import NDArray
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import pycv.metrics.lsf
from pycv.metrics.mtf import MTF


class FittedESF(ESF):
    params: NDArray = None

    def __init__(self, x_data: NDArray, f_data: NDArray, **kwargs):
        super().__init__(x_data, f_data)
        self.params = self.fit(x_data, f_data)

    def get_p0(self):
        raise Exception("Base function FittedESF.get_p0() called")

    def plot_elem(self, **kwargs):
        plt.scatter(self.x_data, self.f_data, label="Measured")
        x_fitted = np.linspace(np.min(self.x_data), np.max(self.x_data), 1000)
        f_fitted = self.f(x_fitted)
        plt.plot(x_fitted, f_fitted, label="Fitted")
        plt.plot()

    def get_bounds(self):
        return [(-np.inf, np.inf) for _ in range(self.get_p0().shape[0])]

    def fit(self, x_data, f_data, **kwargs) -> NDArray:
        p0 = self.get_p0()
        bounds = self.get_bounds()
        try:
            params, _ = curve_fit(self.fn, x_data, f_data, maxfev=40000, p0=p0)
        except RuntimeError:
            params = p0
        return params

    def f(self, x: Union[NDArray, float]) -> Union[NDArray, float]:
        return self.fn(x, *self.params)

    @staticmethod
    def fn(x, *params) -> NDArray:
        raise Exception("Base function FittedESF.fn() called")


class GaussianESF(FittedESF):
    def __init__(self,  x_data: NDArray, f_data: NDArray, **kwargs):
        self.n_terms = kwargs["n_terms"] if "n_terms" in kwargs else 4
        super().__init__(x_data, f_data)

    def get_p0(self):
        return np.ones(self.n_terms*2)

    def lsf(self):
        return pycv.metrics.lsf.GaussianLSF(self.x_data, self.f_data, self.params)

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

    @staticmethod
    def create_from_slanted_edge(img: NDArray, bins_per_pixel=4, zero_centered=True, **kwargs):
        data = ESF.create_data_from_slanted_edge(img, **kwargs)
        x_data, f_data = data["edge_profiles"][0]
        edge_points = data["edge_points"][0]
        esf = GaussianESF(x_data, f_data, bins_per_pixel=bins_per_pixel, zero_centered=zero_centered)
        return esf, edge_points
