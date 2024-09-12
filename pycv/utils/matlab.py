import numpy as np
from numpy.typing import NDArray
import scipy.misc
import math
import scipy.special
import scipy.signal
import decimal
import sys
from typing import Union, List
from scipy.optimize import curve_fit
from .windows import tukey2, ahamming



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