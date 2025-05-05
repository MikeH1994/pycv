import cv2
import numpy as np
from numpy.typing import NDArray
from skimage import morphology

def max_filter(img: NDArray, include_self=True):
    """
    Returns the max of neighbouring pixels
    :param include_self:
    :param img:
    :return:
    """
    # erosion is max filter
    kernel = np.ones((3,3), dtype=np.uint8)
    if not include_self:
        kernel[1, 1] = 0
    return morphology.erosion(img, kernel)


def min_filter(img: NDArray, include_self=True):
    """

    :param img:
    :param include_self:
    :return:
    """
    # dilation is min filter
    kernel = np.ones((3,3), dtype=np.uint8)
    if not include_self:
        kernel[1, 1] = 0
    return morphology.dilation(img, kernel)
