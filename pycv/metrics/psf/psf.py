from __future__ import annotations
from numpy.typing import NDArray
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.integrate import simps
import scipy.interpolate
from scipy.interpolate import RectBivariateSpline
from ...metrics.metric import Metric
from ..lsf.lsf import LSF
from typing import Tuple


class PSF(Metric):
    def __init__(self, interpolation_function: RectBivariateSpline):
        self.interp_fn = interpolation_function

    def create_kernel(self, psf_size: Tuple[int, int], sqrt_n_samples=30):
        h, w = psf_size
        x_offset = (w - 1) / 2
        y_offset = (h - 1) / 2
        psf_kernel = np.zeros(psf_size)
        for x in range(w):
            for y in range(h):
                x0 = x - x_offset - 0.5
                x1 = x - x_offset + 0.5
                y0 = y - y_offset - 0.5
                y1 = y - y_offset + 0.5
                x_samples = np.linspace(x0, x1, sqrt_n_samples)
                y_samples = np.linspace(y0, y1, sqrt_n_samples)
                xx, yy = np.meshgrid(x_samples, y_samples)
                f_values = self.interp_fn(yy, xx, grid=False)
                psf_kernel[y, x] = simps(simps(f_values, x_samples, axis=-1), y_samples)

        psf_kernel /= np.sum(psf_kernel)
        return psf_kernel

    def f(self, x: Union[float, NDArray], y: Union[float, NDArray]):
        return self.interp_fn(y, x, grid=False)

    def lsf(self, angle, x_min=-30.0, x_max=30.0, x_elems=1000, a = -100.0, b = 100.0, integration_elems=5000) -> LSF:
        # x0 and y0 are the points along the line going through the psf with the given angle
        r = np.linspace(x_min, x_max, x_elems)
        dl = np.linspace(a, b, integration_elems)
        x0 = r*np.sin(np.radians(angle))
        y0 = r*np.cos(np.radians(angle))

        # for each point along the line going through the psf, generate a set of coordinates
        # allowing us to integrate it in the perpendicular direction
        xx = np.zeros((x0.shape[0], integration_elems))
        yy = np.zeros((x0.shape[0], integration_elems))
        for i in range(xx.shape[0]):
            xx[i] = x0[i] + dl*np.sin(np.radians(angle + 90.0))
        for i in range(yy.shape[0]):
            yy[i] = y0[i] + dl*np.cos(np.radians(angle + 90.0))

        f = self.f(xx, yy)
        lsf_vals = scipy.integrate.simps(f, x=dl, axis=1)
        return LSF(r, lsf_vals)


