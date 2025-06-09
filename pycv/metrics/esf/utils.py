from numpy.typing import NDArray
import numpy as np
import scipy.integrate
from scipy.integrate import simpson
from scipy.optimize import curve_fit
from pycv.utils.matlab import matlab_round, matlab_conv
import math
from pycv.metrics.targets.slantededge.utils import fermi_function



def bin_data(esf_x: NDArray, esf_f: NDArray, bins_per_pixel: int = 4, zero_centered=True):
    """

    :param esf_x:
    :param esf_f:
    :param bins_per_pixel:
    :param zero_centered:
    :return:
    """
    bin_width = 1.0 / bins_per_pixel
    offset = 0.0 if zero_centered else bin_width/2.0

    x0 = np.round(np.min(esf_x) * bins_per_pixel) / float(bins_per_pixel) - bin_width - offset
    x1 = np.round(np.max(esf_x) * bins_per_pixel) / float(bins_per_pixel) + bin_width + offset
    n_bins = int((x1 - x0) * bins_per_pixel) + 1

    bin_centres = []
    bin_counts = []
    bin_values = []
    bin_std = []
    bin_range = []
    missing_data = []

    for x in np.linspace(x0, x1, n_bins):
        bin_lower = x - bin_width / 2.0
        bin_upper = x + bin_width / 2.0
        indices = np.where((esf_x >= bin_lower) & (esf_x < bin_upper))[0]

        bin_count = 0
        mean = 0.0
        std = 0.0
        half_range = 0.0
        missing = True

        if indices.shape[0] > 0:
            bin_count = np.count_nonzero(indices)
            mean = np.mean(esf_f[indices])
            std = 0.0 if bin_count < 2 else np.std(esf_f[indices])
            half_range = 0.0 if bin_count < 2 else (np.max(esf_f[indices]) - np.min(esf_f[indices])) / 2.0
            missing = False

        bin_centres.append(x)
        bin_values.append(mean)
        bin_std.append(std)
        bin_range.append(half_range)
        bin_counts.append(bin_count)
        missing_data.append(missing)

    for i in range(len(bin_centres)):
        if bin_counts[i] == 0:
            if i == 0:
                bin_values[i] = bin_values[i+1]
            elif i == n_bins - 1:
                bin_values[i] = bin_values[i-1]
            else:
                bin_values[i] = (bin_values[i-1] + bin_values[i+1])/2

    bin_centres = np.array(bin_centres)
    bin_values = np.array(bin_values)
    bin_std = np.array(bin_std)
    bin_range = np.array(bin_range)
    return {
        "x": bin_centres,
        "val": bin_values,
        "std": bin_std,
        "range": bin_range,
        "bin_width": bin_width,
        "bins_per_pixel": bins_per_pixel
    }

def normalise_esf_x(x_data, f_data):
    init_guess = 0.0
    popt, pcov = curve_fit(fermi_function, x_data, f_data, p0=(np.max(f_data), np.min(f_data), 1.0, init_guess))
    x_pos = popt[-1]
    x_data -= x_pos
    return x_data, f_data

def normalise_esf_data(x_data, esf_data):
    if np.mean(esf_data[x_data < 0]) > np.mean(esf_data[x_data > 0]):
        x_data *= -1

    sorted_indices = np.argsort(x_data)
    x_data = x_data[sorted_indices]
    esf_data = esf_data[sorted_indices]
    lsf_data = np.gradient(esf_data, x_data)

    integral = simpson(lsf_data, x=x_data)
    scale_factor = 1.0 / integral
    # x_offset = - centroid(esf_data, x_data)

