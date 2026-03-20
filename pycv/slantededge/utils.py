import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import warnings
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from pycv.utils.windows import tukey2, ahamming
from pycv.utils.matlab import matlab_conv, matlab_round


def get_edge_points_from_centroid(img: NDArray, derivative_filter: NDArray, wflag, alpha = 1.0, npol=5):
    """

    :param img:
    :param derivative_filter:
    :param wflag:
    :param alpha:
    :param npol:
    :return:
    """
    assert(edge_is_vertical(img))
    assert(len(img.shape) == 2)
    height, width = img.shape

    # smoothing window for first part of edge location estimation
    win1 = get_window(width, (width-1)/2, wflag, alpha)

    # compute initial edge location and fitting
    lsf = deriv1(img[:, :], derivative_filter)

    edge_y = np.arange(height, dtype=np.float32)
    edge_x = []
    edge_x_refined = []
    for y in range(height):
        edge_x.append(find_centroid_in_row(lsf[y], win1))  # subtract 0.5 for FIR filter
    edge_x = np.array(edge_x)
    # fit polynomial of the form x = a + b*y + c*y**2 + d*y**3 + ...
    edge_fit = np.polynomial.polynomial.polyfit(edge_y, edge_x, npol)

    for y in range(height):
        edge_loc = np.polynomial.polynomial.polyval(y, edge_fit).item()
        win2 = get_window(width, edge_loc, wflag, alpha)
        edge_x_refined.append(find_centroid_in_row(lsf[y], win2))  # subtract 0.5 for FIR filter
    edge_x_refined = np.array(edge_x_refined)
    edge_y_refined = edge_y
    return edge_x_refined, edge_y_refined


def get_edge_points_from_esf_fit(img: NDArray, wflag, alpha = 1.0):
    """

    :param img:
    :return:
    """
    assert(edge_is_vertical(img))
    assert(len(img.shape) == 2)

    height, width = img.shape
    edge_y = []
    edge_x = []
    x_data = np.arange(width)
    derivative_filter = get_derivative_filters(img, size=2)
    lsf = deriv1(img[:, :], derivative_filter)
    win1 = get_window(width, (width-1)/2, wflag, alpha)

    for y in range(height):
        f_data = img[y]
        with warnings.catch_warnings():
            warnings.simplefilter("error", scipy.optimize.OptimizeWarning)
            try:
                init_guess = find_centroid_in_row(lsf[y], win1)
                popt, pcov = curve_fit(fermi_function, x_data, f_data, p0=(np.max(f_data), np.min(f_data), 1.0, init_guess))
                edge_x.append(popt[-1])
                edge_y.append(y)
            except scipy.optimize.OptimizeWarning:
                pass
    return np.array(edge_x), np.array(edge_y)


def get_edge_points_from_lsf_fit(img: NDArray, wflag, alpha = 1.0):
    assert(edge_is_vertical(img))
    assert(len(img.shape) == 2)
    height, width = img.shape

    derivative_filter = get_derivative_filters(img, size=2)

    # smoothing window for first part of edge location estimation
    win1 = get_window(width, (width-1)/2, wflag, alpha)
    # compute initial edge location and fitting
    lsf = deriv1(img[:, :], derivative_filter)

    edge_y = np.arange(height, dtype=np.float32)
    edge_x = []

    x_data = np.arange(width)
    for y in range(height):
        # make initial guess using centroid
        x = find_centroid_in_row(lsf[y], win1)
        # use this to create a window
        win2 = get_window(width, x, wflag, alpha)
        # then fit curve to windowed data
        popt, pcov = curve_fit(gaussian_function, x_data, lsf[y] * win2, p0=[0.4*np.max(lsf[y]), x, 1])

        edge_x.append(popt[1] - 0.5)
    edge_x = np.array(edge_x)
    return edge_x, edge_y


def gaussian_function(x, a, mu, sigma):
    """
    Gaussian function, used to fit the LSF of a row to find its edge location
    :param x:
    :param a:
    :param mu:
    :param sigma:
    :return:
    """
    return a * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))


def fermi_function(x, b, d, s, e):
    """
    Fermi function, use to fit the ESF of a row to find its edge location
    :param x:
    :param params:
    :return:
    """
    return d + (b - d)/(1 + np.exp(-s*(x-e)))


def edge_is_vertical(img: NDArray):
    """

    :param img:
    :return:
    """
    if len(img.shape) == 2:
        height, width = img.shape[:2]
        img = img.reshape((height, width, 1))
    nn = 3
    mm = 2 if img.shape[2] == 3 else 0

    vertical_diff = np.abs(np.mean(img[-nn, :, mm]) - np.mean(img[nn, :, mm]))
    horizontal_diff = np.abs(np.mean(img[:, -nn, mm]) - np.mean(img[:, nn, mm]))
    # if the vertical difference is high, that means the edge is horizontal and vice versa
    return vertical_diff < horizontal_diff


def rotate_image(img: NDArray):
    """

    :param img:
    :return:
    """
    assert(len(img.shape) == 2 or len(img.shape) == 3)
    if len(img.shape) == 2:
        img = np.transpose(img, (1, 0))
    else:
        img = np.transpose(img, (1, 0, 2))
    return img


def get_window(window_length, midpoint, wflag, alpha: float = 1.0):
    """

    :param window_length:
    :param midpoint:
    :param wflag:
    :param alpha:
    :return:
    """
    if wflag == 0:
        win1 = tukey2(window_length, alpha, midpoint)
        win1 = 0.95*win1 + 0.05
    else:
        win1 = ahamming(window_length, midpoint)
    return win1


def find_centroid_in_row(row, window):
    """

    :param row:
    :param window:
    :return:
    """
    return centroid(row * window) - 0.5


def centroid(vec: NDArray, x: NDArray = None) -> float:
    """
    Calculate the centroid of a vector

    :param vec: the vector
    :param x: coordinates corresponding to each point. If none, arange(vec.shape[0]) will be used
    :return:
    """
    assert(len(vec.shape) == 1)
    x = x if x is not None else np.arange(vec.shape[0])
    loc = np.sum(x*vec)/np.sum(vec)
    return loc


def deriv1(data: NDArray, fil: NDArray) -> NDArray:
    """
    Computes first derivative of an (mxn) vector using an FIR filter of shape (k, ).
    The filter is applied in axis 1 (i.e. horizontally)

    :param data: The data to calculate derivative of
    :param fil: The derivative filter to be used, e.g. [-0.5, 0.5]
    :return:
    """
    single_row_passed = len(data.shape) == 1
    if single_row_passed:
        data = data.reshape((1, data.shape[0]))
    assert (len(fil.shape) == 1)
    assert(len(data.shape) == 2)
    assert(data.dtype == np.float32 or data.dtype == np.float64)
    assert(fil.dtype == np.float32 or fil.dtype == np.float64)

    img_deriv = matlab_conv(data, fil)
    img_deriv[:, 0] = img_deriv[:, 1]
    img_deriv[:, -1] = img_deriv[:, -2]

    if single_row_passed:
        img_deriv = img_deriv.reshape(-1)
    return img_deriv


def findedge2(x: NDArray, y: NDArray, npol, mode: str = "f(x)"):
    """
    Fits polynomial equation to the centroids of the edge, of the form
    x = a + b*y + c*y**2 + ...

    :param x: the x data
    :param y: the y data
    :param npol: the order of the polynomial used.
    :return: the polynomial coefficients [a, b, c, d, ...], corresponding to the polynomial
             if mode == f(x): a + b*x + c*x**2 + d*x**3 + ...
             if mode == f(y): a + b*y + c*y**2 + d*y**3 + ...
    """
    assert(len(x.shape) == 1)
    assert(x.shape == y.shape)
    assert(mode == "f(x)" or mode == "f(y)")
    if mode == "f(x)":
        return np.polynomial.polynomial.polyfit(x, y, npol)
    else:
        return np.polynomial.polynomial.polyfit(y, x, npol)


def cent(data, center):
    """
    Centres an array, such that the previous center (cent) is now located at the midpoint
    of the array

    :param data:
    :param center:
    :return:
    """
    assert(len(data.shape) == 1)
    assert(data.dtype == np.float32 or data.dtype == np.float64)
    center += 1  # to match matlab indices
    n = data.shape[0]
    centred_data = np.zeros(data.shape)
    mid = matlab_round((n+1)/2)
    del_ = matlab_round(center - mid)

    if del_ > 0:
        i = np.arange(1, n - del_ + 1)
        centred_data[i - 1] = data[i + del_ - 1]
    elif del_ < 1:
        i = np.arange(-del_+1, n + 1)
        centred_data[i - 1] = data[i + del_ - 1]
    else:
        centred_data = np.copy(data)
    return centred_data


def get_derivative_filters(img: NDArray, size: int):
    """
    Returns the derivative filters used in the sfrmat5 algorithm

    :param img:
    :return: fil - the derivative filter requested
    """
    if size == 2:
        fil = np.array([0.5, -0.5])
    elif size == 3:
        fil = np.array([0.5, 0, -0.5])
    else:
        raise Exception("Invalid filter size")
    img = img.reshape((img.shape[0], img.shape[1], -1))
    # we need a positive edge
    t_left = np.sum(np.sum(img[:, :5, 0], axis=1))
    t_right = np.sum(np.sum(img[:, -5:, 0], axis=1))

    if t_left > t_right:
        fil = fil[::-1]

    return fil
