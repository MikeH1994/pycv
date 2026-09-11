from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
import numpy as np
from numpy.typing import NDArray
import scipy.signal

def calculate_fwhm(x: NDArray, y: NDArray):
    """

    :param x:
    :param y:
    :return:
    """
    # find_peaks and peak_widths give results in terms of indices rather than x pos
    peak_indices, _ = scipy.signal.find_peaks(y)

    if peak_indices.shape[0] > 1:
        print("Warning: more than 1 peak found")

    # we only want to take the highest peak, ignore everything else
    peak_index = [peak_indices[np.argmax(y[peak_indices])]]
    interp_fn = scipy.interpolate.interp1d(np.arange(x.shape[0]), x)
    cx = interp_fn(peak_index)
    roots = find_intercepts(x, y, (np.max(y) - np.min(y)) / 2)
    fwhm = np.max(roots) - np.min(roots)

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


def find_intercepts(x, y, threshold):
    spline = InterpolatedUnivariateSpline(x, y - threshold)
    roots = spline.roots()
    return roots
