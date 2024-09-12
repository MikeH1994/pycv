import numpy as np
from numpy.typing import NDArray
from pycv.metrics.targets.slantededge.utils import get_derivative_filters, deriv1, centroid, findedge2, edge_is_vertical, get_window, rotate_image
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pycv.metrics.targets.slantededge.utils import get_edge_points_from_fit, get_edge_points_from_centroid


def get_edge_points(img: NDArray, convert_to_grayscale=True, edge_detection_mode="fit" ,**kwargs):
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

    # derivative filter
    fil1, _ = get_derivative_filters(img)

    returned_data = []

    for channel in range(n_channels):
        if edge_detection_mode == "fit":
            edge_x, edge_y = get_edge_points_from_fit(img[:, :, channel])
        elif edge_detection_mode == "centroid":
            wflag = kwargs["wflag"] if "wflag" in kwargs else 0
            alpha = kwargs["alpha"] if "alpha" in kwargs else 1.0
            npol = kwargs["npol"] if "npol" in kwargs else 5
            edge_x, edge_y = get_edge_points_from_centroid(img[:, :, channel], fil1, wflag, alpha, npol)
        else:
            raise ValueError("Invalid edge detection mode- {}".format(edge_detection_mode))
        if not is_vertical:
            edge_x, edge_y = edge_y, edge_x
        returned_data.append((edge_x, edge_y))
    return returned_data


def get_edge_profile_from_image(img: NDArray, convert_to_grayscale=True, edge_detection_mode="fit", **kwargs):

    height, width = img.shape[:2]
    img = img.reshape((height, width, -1))

    if convert_to_grayscale:
        img = np.mean(img, axis=-1).reshape((height, width, -1))

    edge_points_per_channel = get_edge_points(img, edge_detection_mode=edge_detection_mode, **kwargs)
    edge_profiles_per_channel = []
    linear_fits_per_channel = []

    for channel_index, edge_points in enumerate(edge_points_per_channel):
        x, y = edge_points
        poly = np.polynomial.polynomial.polyfit(x, y, 1)
        m = poly[1]

        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        edge_y = np.polynomial.polynomial.polyval(xx, poly)

        distance = ((yy - edge_y) / np.sqrt(m ** 2 + 1)).reshape(-1).tolist()
        vals = img.reshape(-1).tolist()

        sorted_xf = sorted(zip(distance, vals))

        esf_x = np.array([x for x, f in sorted_xf])
        esf_f = np.array([f for _, f in sorted_xf])

        if np.mean(esf_f[esf_x < 0]) > np.mean(esf_f[esf_x > 0]):
            esf_x *= -1.0
            esf_x = esf_x[::-1]
            esf_f = esf_f[::-1]

        linear_fits_per_channel.append(poly)
        edge_profiles_per_channel.append((esf_x, esf_f))

    return {
        "edge_profiles": edge_profiles_per_channel,
        "edge_points": edge_points_per_channel,
        "linear_fits": linear_fits_per_channel,
    }

