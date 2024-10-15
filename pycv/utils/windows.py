import numpy as np
from numpy.typing import NDArray
import math
import scipy.interpolate

def ahamming(n, midpoint) -> NDArray:
    """
    Generates a general asymmetric Hamming-type window
    array. If mid = (n-1)/2 then the usual symmetric Hamming window is returned.

    :param n: length of array
    :param midpoint: midpoint (maximum) of window function
    :return:
    """
    midpoint += 1.5

    wid1 = midpoint - 1
    wid2 = n - midpoint
    wid = max(wid1, wid2)

    arg = (np.arange(n) + 1 - midpoint)
    data = np.cos(np.pi * arg / wid)
    data = 0.54 + 0.46 * data

    return data


def tukey2(n: int, alpha: float, mid: float):
    """
    Asymmetrical Tukey Tapered Cosine Window
    w = tukey2(n,alpha, mid) returns an n-point Tukey window with
    center at mid. Tukey window parameter is alpha

    :param n: length of window
    :param alpha: Tukey window parameter
    :param mid: Centre of window
    :return:
    """

    if n < 3:
        return np.ones(n)

    mid += 1.0
    m1 = n/2
    m2 = mid
    m3 = n-mid
    mm = max(m2, m3)
    n2 = round(2*mm)

    window = tukey(n2, alpha)

    if mid >= m1:
        window = window[:n]
    else:
        window = window[-n:]

    return window


def tukey(n, alpha):
    """
    Creates a symmetric n-point tukey window

    :param n: the length of the window
    :param alpha: the tukey window parameter
    :return:
    """
    if n == 1:
        return np.array([1.0])

    if alpha == 0:
        return np.ones(n)

    m = (n-1)/2
    window = np.zeros(n)
    k = np.arange(math.floor(m + 1))
    window[k[k > alpha * m]] = 1.0
    window[k[k <= alpha*m]] = 0.5*(1+np.cos(np.pi*(k[k <= alpha*m]/alpha/m - 1)))
    window[n - k - 1] = window[k]

    return window


def ahamming_fn(width, midpoint=0):
    halfwidth = width/2.0
    x0 = - halfwidth
    x1 = halfwidth
    x = np.linspace(x0, x1, 5000)
    y = 0.54 + 0.46 * np.cos(np.pi * x / halfwidth)
    x = np.array([x0-1, *x, x1+1])
    y = np.array([y[0], *y, y[-1]])
    fn = scipy.interpolate.interp1d(x + midpoint, y, fill_value="extrapolate")
    return fn


def tukey_fn(width, alpha, midpoint=0):
    x = np.arange(width)
    y = tukey2(width, alpha, (width - 1) / 2.0)
    x = np.array([x[0] - 1, *x, x[-1] + 1])
    y = np.array([y[0], *y, y[-1]])

    fn = scipy.interpolate.interp1d(x - (width - 1) / 2.0 + midpoint, y, fill_value="extrapolate")
    return fn
