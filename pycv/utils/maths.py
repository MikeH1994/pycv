from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
from numpy.typing import NDArray


def calculate_fwhm(x: NDArray, y: NDArray):
    """

    :param x:
    :param y:
    :return:
    """
    intercept = (np.max(x) - np.min(x)) / 2.0
    spline = InterpolatedUnivariateSpline(x, y - intercept)
    roots = spline.roots()
    assert(len(roots) > 2 and len(roots) % 2 == 0)
    x0, x1 = roots[0], roots[-1]
    cx = (x0 + x1) / 2.0
    fwhm = np.abs(x1 - x0)
    return cx, fwhm


def calculate_bounds_based_on_fwhm(x: NDArray, y: NDArray, k = 2.5):
    """

    :param x:
    :param y:
    :param k:
    :return:
    """
    cx, fwhm = calculate_fwhm(x, y)
    dx = fwhm / 2.0
    x0 = cx - k*dx
    x1 = cx + k*dx
    return x0, x1