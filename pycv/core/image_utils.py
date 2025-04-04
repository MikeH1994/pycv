from typing import Tuple
from typing import Union, List
import cv2
import numpy as np
from numpy.typing import NDArray


def convert_to_8_bit(src: NDArray, min_val=None, max_val=None, return_as_rgb=False):
    # if a single image is supplied
    if src.dtype == np.uint8:
        if return_as_rgb and len(src.shape) == 1:
            return cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
        return src

    min_val = np.min(src) if min_val is None else min_val
    max_val = np.max(src) if max_val is None else max_val

    if min_val > max_val:
        raise Exception("Invalid min and max bounds found!")

    if min_val == max_val:
        scale_factor = 1.0
    else:
        scale_factor = 255.0 / (max_val - min_val)
    img = src.astype(np.float32)
    img[img < min_val] = min_val
    img[img > max_val] = max_val
    img -= min_val
    img *= scale_factor
    img = img.astype(np.uint8)
    if return_as_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def is_rgb(img: NDArray) -> bool:
    """

    :param img:
    :return:
    """
    return len(img.shape) == 3 and img.shape[2] == 3 and img.dtype == np.uint8


def is_grayscale(img: NDArray) -> bool:
    """

    :param img:
    :return:
    """
    return len(img.shape) == 2 and img.dtype == np.uint8


def image_is_valid(img: NDArray) -> bool:
    """

    :param img:
    :return:
    """
    return is_rgb(img) or is_grayscale(img)


def to_rgb(img: NDArray) -> NDArray:
    """

    :param img:
    :return:
    """
    assert(image_is_valid(img)), f"Image is not valid: Shape = {img.shape}, dtype={img.dtype}"
    if is_rgb(img):
        return img
    elif is_grayscale(img):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        raise Exception("Logic error")


def to_grayscale(img: NDArray) -> NDArray:
    """

    :param img:
    :return:
    """
    assert(image_is_valid(img)), f"Image is not valid: Shape = {img.shape}, dtype={img.dtype}"
    if is_grayscale(img):
        return img
    elif is_rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        raise Exception("Logic error")


def n_channels(img: NDArray):
    """

    :param img:
    :return:
    """
    assert(len(img.shape) == 2 or len(img.shape) == 3)
    if len(img.shape) == 2:
        return 1
    return img.shape[-1]
