from .psf import PSF
from ...utils.functions import gaussian_2d
from typing import Tuple
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.integrate import simpson


def create_gaussian_psf(psf_size: Tuple[int, int], sigma_x=1.0, sigma_y=1.0, cx=0.0, cy=0.0, n_samples=100) -> PSF:
    h, w = psf_size
    x = np.linspace(-w / 2 - 1, w / 2 + 1, n_samples)
    y = np.linspace(-h / 2 - 1, h / 2 + 1, n_samples)
    xx, yy = np.meshgrid(x, y)
    f = gaussian_2d(xx, yy, sigma_x=sigma_x, sigma_y=sigma_y, cx=cx, cy=cy)
    interpolation_function = RectBivariateSpline(y, x, f)
    return PSF(interpolation_function)

