import numpy as np
from typing import Tuple
from numpy.typing import NDArray
from pycv.slantededge import edge_is_vertical, rotate_image
from pycv.slantededge.utils import get_edge_points_from_esf_fit,  get_edge_points_from_lsf_fit, get_edge_points_from_centroid


def get_edge_points(img: NDArray, edge_detection_mode="fit_esf", **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """

    :param img:
    :param convert_to_grayscale:
    :return:
    """
    height, width = img.shape[:2]
    assert(len(img.shape) == 2)
    img = img.reshape((height, width))

    is_vertical = edge_is_vertical(img)
    if not is_vertical:
        img_output = rotate_image(img)
    else:
        img_output = img

    wflag = kwargs["wflag"] if "wflag" in kwargs else 0
    alpha = kwargs["alpha"] if "alpha" in kwargs else 1.0
    npol = kwargs["npol"] if "npol" in kwargs else 5
    if edge_detection_mode == "fit_esf":
        edge_x, edge_y = get_edge_points_from_esf_fit(img_output, wflag)
    elif edge_detection_mode == "fit_lsf":
        edge_x, edge_y = get_edge_points_from_lsf_fit(img_output, wflag, alpha)
    elif edge_detection_mode == "centroid":
        edge_x, edge_y = get_edge_points_from_centroid(img_output, wflag, alpha, npol)
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
    return edge_x, edge_y
