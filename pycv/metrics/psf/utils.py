from __future__ import annotations
from .psf import PSF
from ...utils.functions import gaussian_2d
from typing import Tuple
import numpy as np
from scipy.interpolate import RectBivariateSpline


def kernel_to_interpolation_fn(kernel, n_samples=100):
    h, w = kernel.shape[:2]
    x = np.linspace(-(w-1) // 2, (w-1) // 2, w)
    y = np.linspace(-(h-1) // 2, (h-1) // 2, h)
    return RectBivariateSpline(y, x, kernel)
