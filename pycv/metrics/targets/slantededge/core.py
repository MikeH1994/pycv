import numpy as np
from numpy.typing import NDArray
from pycv.metrics.targets.slantededge.utils import get_derivative_filters, deriv1, centroid, findedge2, edge_is_vertical, get_window, rotate_image
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from .utils import get_edge_points_from_esf_fit,  get_edge_points_from_lsf_fit, get_edge_points_from_centroid
from .edge import Edge


def get_edge_points(img: NDArray, convert_to_grayscale=True, edge_detection_mode="fit_lsf", **kwargs):
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
        img_output = rotate_image(img)
    else:
        img_output = img
    height, width, n_channels = img_output.shape

    returned_data = []
    wflag = kwargs["wflag"] if "wflag" in kwargs else 0
    alpha = kwargs["alpha"] if "alpha" in kwargs else 1.0
    npol = kwargs["npol"] if "npol" in kwargs else 5
    for channel in range(n_channels):
        if edge_detection_mode == "fit_esf":
            edge_x, edge_y = get_edge_points_from_esf_fit(img_output[:, :, channel], wflag)
        elif edge_detection_mode == "fit_lsf":
            edge_x, edge_y = get_edge_points_from_lsf_fit(img_output[:, :, channel], wflag, alpha)
        elif edge_detection_mode == "centroid":
            edge_x, edge_y = get_edge_points_from_centroid(img_output[:, :, channel], wflag, alpha, npol)
        elif edge_detection_mode == "ground_truth":
            assert("p0" in kwargs and "p1" in kwargs)
            p0 = kwargs["p0"]
            p1 = kwargs["p1"]
            assert(len(p0) == 2 and len(p1) == 2)
            edge_x = np.array([p0[0], p1[0]], dtype=np.float32)
            edge_y = np.array([p0[1], p1[1]], dtype=np.float32)
            if not is_vertical:
                edge_x, edge_y = edge_y, edge_x
        else:
            raise ValueError("Invalid edge detection mode- {}".format(edge_detection_mode))
        if not is_vertical:
            edge_x, edge_y = edge_y, edge_x
        returned_data.append(Edge(img[:, :, channel], edge_x, edge_y))
    return returned_data


def get_edge_profile_from_image(img: NDArray, convert_to_grayscale=True, edge_detection_mode="fit_esf", **kwargs):
    height, width = img.shape[:2]
    img = img.reshape((height, width, -1))

    if convert_to_grayscale:
        img = np.mean(img, axis=-1).reshape((height, width, -1))

    edge_per_channel = get_edge_points(img, edge_detection_mode=edge_detection_mode, **kwargs)
    edge_profiles_per_channel = []
    linear_fits_per_channel = []

    for channel_index, edge in enumerate(edge_per_channel):
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))

        distance = edge.get_distance_to_edge(xx, yy).reshape(-1).tolist()
        vals = img.reshape(-1).tolist()

        sorted_xf = sorted(zip(distance, vals))

        esf_x = np.array([x for x, f in sorted_xf])
        esf_f = np.array([f for _, f in sorted_xf])

        if np.mean(esf_f[esf_x < 0]) > np.mean(esf_f[esf_x > 0]):
            esf_x *= -1.0
            esf_x = esf_x[::-1]
            esf_f = esf_f[::-1]

        edge_profiles_per_channel.append((esf_x, esf_f))

    return {
        "edge_profiles": edge_profiles_per_channel,
        "edges": edge_per_channel,
    }
