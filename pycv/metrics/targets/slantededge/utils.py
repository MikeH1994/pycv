import numpy as np
from numpy.typing import NDArray
import scipy.misc
import math
import scipy.special
import scipy.signal
import decimal
import sys
from typing import Union, List
from .windows import tukey2, ahamming


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
    if wflag == 0:
        win1 = tukey2(window_length, alpha, midpoint)
        win1 = 0.95*win1 + 0.05
    else:
        win1 = ahamming(window_length, midpoint)
    return win1


def centroid(vec: NDArray) -> float:
    """
    Calculate the centroid of a vector

    :param vec: the vector
    :return:
    """
    assert(len(vec.shape) == 1)
    loc = np.sum(np.arange(vec.shape[0])*vec)/np.sum(vec)
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


def findedge2(x: NDArray, y: NDArray, npol):
    """
    Fits polynomial equation to the centroids of the edge, of the form
    x = a + b*y + c*y**2 + ...

    :param x: the
    :param npol: the order of the polynomial used.
    :return: the polynomial coefficients [a, b, c, d, ...], corresponding to the polynomial
             a + b*y + c*y**2 + d*y**3 + ...
    """
    assert(len(x.shape) == 1)
    assert(x.shape == y.shape)
    p = np.polynomial.polynomial.polyfit(y, x, npol)
    return p


def fir2fix(n, m):
    """
    Correction for MTF of derivative (difference) filter

    :param n: frequency data length [0-half-sampling (Nyquist) frequency]
    :param m: length of difference filter, e.g. 2-point difference has m=2, 3-point difference has m=3
    :return: the correction to apply
    """
    m -= 1

    i = np.arange(1, n)
    correction = np.ones(n)
    correction[i] = np.abs((np.pi*(i+1)*m/(2*(n+1))) / np.sin(np.pi*(i+1)*m/(2*(n+1))))
    correction[correction > 10] = 10

    return correction


def project2(image, fitme, fac):
    """

    :param image:
    :param fitme:
    :param fac: binning factor
    :return: binned_data, a (2, x) array where the first row contains the number of elements
             in the bin, and the second row contains the mean value
    """
    assert(len(image.shape) == 2)
    assert(image.dtype == np.float32)
    height, width = image.shape
    slope = 1.0 / fitme[1]
    nn = math.floor(width*fac)
    offset = matlab_round(-fac*(height-1)/slope)

    del_ = abs(offset)
    if offset > 0:
        offset = 0.0

    bwidth = nn + del_ + 150
    binned_data = np.zeros((2, bwidth))
    poly_fn = np.polynomial.Polynomial(fitme)

    p2 = poly_fn(np.arange(height)) - fitme[0]

    n, m = np.meshgrid(np.arange(width), np.arange(height))
    bin_indices = (np.ceil((n-p2[m])*fac) - offset).astype(np.int32)
    bin_indices[bin_indices < 0] = 0
    bin_indices[bin_indices > bwidth - 1] = bwidth - 1

    bin_indices = bin_indices.reshape(-1)
    np.add.at(binned_data[0], bin_indices, 1)  # binned_data[0, bin_indices] += 1
    np.add.at(binned_data[1], bin_indices, image.reshape(-1))  # binned_data[1, bin_indices] += image

    start = matlab_round(0.5*del_)

    # checking for bins with no values in them
    zero_indices = np.arange(start, start + nn)[binned_data[0, np.arange(start, start + nn)] == 0]

    for i in zero_indices:
        if binned_data[0, i] == 0:
            if i == 0:
                binned_data[0, 0] = binned_data[0, 1]
                binned_data[1, 0] = binned_data[1, 1]
            if i == start + nn - 1:
                binned_data[0, start + nn - 1] = binned_data[0, start + nn - 2]
                binned_data[1, start + nn - 1] = binned_data[1, start + nn - 2]
            else:
                binned_data[0, i] = (binned_data[0, i-1] + binned_data[0, i+1])/2
                binned_data[1, i] = (binned_data[1, i-1] + binned_data[1, i+1])/2

    # calculate mean value of each bin
    point = np.zeros(nn)
    indices = np.arange(nn)[binned_data[0, np.arange(nn) + start] > 0]
    point[indices] = binned_data[1, indices+start]/binned_data[0, indices+start]

    return point


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


def sampeff(data: NDArray, val: Union[NDArray, List], del_: float, fflag: int = 0, pflag: int = 0):
    """
    Calculate the sampling efficiency from the SFR

    :param data: (n, 2), (n, 4), or (n,5) array. First col is frequency
    :param val: (n, ) vector of SFR frequency values we want the results of
    :param del_: sampling interval in mm (default = 1 pixel)
    :param fflag: 0 (default) no filter
                  1 = filter [1 1 1] applied to sfr
    :param pflag: 0 (default) no plots
                  1 plot results
    :return:
    """
    assert(len(data.shape) == 2)
    assert(data.shape[1] > 1)
    val = np.asarray(val)

    n, m = data.shape[:2]
    n_channels = m - 1
    imax = n - 1
    nval = val.shape[0]
    eff = np.zeros((nval, n_channels))
    freqval = np.zeros((nval, n_channels))
    sfrval = np.zeros((nval, n_channels))

    hs = 0.495 / del_
    x = np.where(data[:, 0] > hs)[0]
    if x.shape[0] == 0:
        print(" Missing SFR data, frequency up to half-sampling needed")

    for v in range(nval):
        freqval[v], sfrval[v] = findfreq(data, val[v], imax, fflag)
        freqval[v] = np.clip(freqval[v], 0.0, hs)

        for c in range(n_channels):
            eff[v, c] = min(matlab_round(100.0*freqval[v, c]/hs), 100)

    if pflag != 0:
        # TODO do plotting
        pass

    return eff, freqval, sfrval


def findfreq(data, val, imax, fflag: int = 0):
    """
    Find the frequency corresponding to a given SFR value

    :param data: (n, 2), (n, 4), or (n,5) array. First col is frequency
    :param val: (n, ) vector of SFR frequency values we want the results of
    :param imax: index of half-sampling frequency (normally n)
    :param fflag: 1 = filter [1 1 1] applied to sfr
                  0 (default) no filter
    :return:
    """
    assert(len(data.shape) == 2)
    n, m = data.shape
    nc = m - 1
    sfrval = np.zeros(nc)
    freqval = np.zeros(nc)

    maxf = data[imax, 0]
    fil = np.array([1.0, 1.0, 1.0])/3.0

    if fflag != 0:
        data = np.copy(data)

    for channel in range(nc):
        if fflag != 0:
            temp = matlab_conv(data[:, channel + 1], fil)
            data[1:-1, channel + 1] = temp[1:-1]
        test = data[:, channel + 1] - val
        x = np.where(test < 0)[0]

        if x.shape[0] == 0 or x[0] == 0:
            s = maxf
            sval = data[imax, channel + 1]
        else:
            x = x[0] - 1
            sval = data[x, channel + 1]
            s = data[x, 0]
            y = data[x, channel + 1]
            y2 = data[x+1, channel + 1]

            slope = (y2 - y) / data[1, 0]
            dely = test[x]
            s -= dely / slope
            sval -= dely

        if s > maxf:
            s = maxf
            sval = data[imax, channel + 1]

        freqval[channel] = s
        sfrval[channel] = sval
    return freqval, sfrval


def get_derivative_filters(img: NDArray):
    """
    Returns the derivative filters used in the sfrmat5 algorithm

    :param img:
    :return: fil1 - 1st derivative filter used for the image
             fil2 - 1st derivative filter used for the esf
    """
    fil1 = np.array([0.5, -0.5])
    fil2 = np.array([0.5, 0, -0.5])

    # we need a positive edge
    t_left = np.sum(np.sum(img[:, :5, 0], axis=1))
    t_right = np.sum(np.sum(img[:, -6:, 0], axis=1))

    if t_left > t_right:
        fil1 = np.array([-0.5, 0.5])
        fil2 = np.array([-0.5, 0, 0.5])

    # test for low contrast edge
    test = np.abs((t_left - t_right)/(t_left+t_right))
    if test < 0.2:
        print(" ** WARNING: Edge contrast is less that 20%, this can\n"
              "             lead to high error in the SFR measurement.")

    return fil1, fil2

def matlab_round(x):
    """
    Round a float to an integer, using the same rounding convention used in matlab
    (0.5->1.0 instead of 0.5->0.0)
    :param x:
    :return:
    """
    return int(decimal.Decimal(x).to_integral_value(rounding=decimal.ROUND_HALF_UP))


def matlab_conv(data, fil):
    """
    Convolves the input data with the specified convolution filter, taking care with
    padding to ensure the same numerical result is achieved as Matlab's conv(a,b,'same') operation.
    'data' and 'fil' should have the same number of dimensions. If 'fil' only has a single dimension,
    it is assumed that it will operate over the last axis.

    :return:
    """

    if len(data.shape) > len(fil.shape):
        n_dims_to_add = len(data.shape) - len(fil.shape)
        fil = np.expand_dims(fil, tuple(range(n_dims_to_add)))

    assert(len(data.shape) == len(fil.shape))

    padding = []
    for i in range(len(fil.shape)):
        npad = fil.shape[i] - 1
        padding.append((npad // 2, npad - npad // 2))
    data_padded = np.pad(data, padding)
    conv = scipy.signal.convolve(data_padded, fil, mode='valid')

    return conv
