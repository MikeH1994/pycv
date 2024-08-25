import numpy as np
from numpy.typing import NDArray
from pycv.metrics.targets.slantededge.utils import get_derivative_filters, deriv1, centroid, findedge2, edge_is_vertical, get_window, rotate_image
from numpy.polynomial.polynomial import polyval
import matplotlib.pyplot as plt


def get_edge_points(img: NDArray, convert_to_grayscale=True, wflag = 0, alpha=1.0, npol=5):
    """

    :param img:
    :param convert_to_grayscale:
    :return:
    """
    height, width = img.shape[:2]
    img = img.reshape((height, width, -1))
    if convert_to_grayscale:
        img = np.mean(img, axis=-1).reshape((height, width, -1))

    is_vertical = edge_is_vertical(img)
    if not is_vertical:
        img = rotate_image(img)
    height, width, n_channels = img.shape

    # smoothing window for first part of edge location estimation
    win1 = get_window(width, (width-1)/2, wflag, alpha)
    # derivative filter
    fil1, _ = get_derivative_filters(img)

    returned_data = []

    for channel in range(n_channels):
        # compute initial edge location and fitting
        lsf = deriv1(img[:, :, channel], fil1)

        edge_y = np.arange(height, dtype=np.float32)
        edge_x = []
        edge_x_refined = []
        for y in range(height):
            edge_x.append(centroid(lsf[y]*win1) - 0.5)  # subtract 0.5 for FIR filter
        edge_x = np.array(edge_x)
        edge_fit = findedge2(edge_x, edge_y, npol)

        for y in range(height):
            edge_loc = polyval(y, edge_fit).item()
            win2 = get_window(width, edge_loc, wflag, alpha)
            edge_x_refined.append(centroid(lsf[y]*win2) - 0.5)  # subtract 0.5 for FIR filter
        edge_x_refined = np.array(edge_x_refined)
        edge_y_refined = edge_y

        if not is_vertical:
            edge_x_refined, edge_y_refined = edge_y_refined, edge_x_refined
        returned_data.append((edge_x_refined, edge_y_refined))
    return returned_data


def get_edge_values(img: NDArray, polynomial: NDArray):
    assert(edge_is_vertical(img))
    assert(polynomial.shape == (2, ))
    height, width = img.shape[:2]
    m = polynomial[1]

    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    edge_y = polyval(xx, polynomial)

    distance = ((yy - edge_y) / np.sqrt(m ** 2 + 1)).reshape(-1).tolist()
    vals = img.reshape(-1).tolist()

    sorted_xf = sorted(zip(distance, vals))

    esf_x = np.array([x for x, f in sorted_xf])
    esf_f = np.array([f for _, f in sorted_xf])
    data = np.zeros((esf_x.shape[0], 2))
    data[:, 0] = esf_x
    data[:, 1] = esf_f
    return data


def get_edge_profile_from_image(img: NDArray, convert_to_grayscale=True, wflag = 0, alpha=1.0, npol=5):
    height, width = img.shape[:2]
    img = img.reshape((height, width, -1))

    is_vertical = edge_is_vertical(img)
    if is_vertical:
        img = rotate_image(img)
        height, width = img.shape[:2]

    if convert_to_grayscale:
        img = np.mean(img, axis=-1).reshape((height, width, -1))

    edge_points_per_channel = get_edge_points(img, wflag=wflag, alpha=alpha, npol=npol)
    edge_profiles_per_channel = []
    linear_fits_per_channel = []

    for channel_index, edge_points in enumerate(edge_points_per_channel):
        poly = np.polynomial.polynomial.polyfit(edge_points[:, 0], edge_points[:, 1], 1)
        linear_fits_per_channel.append(poly)
        edge_profiles_per_channel.append(get_edge_values(img[:, :, channel_index], poly))

    return{
        "edge_profiles": edge_profiles_per_channel,
        "edge_points": edge_points_per_channel,
        "linear_fits": linear_fits_per_channel,
    }


