from typing import Tuple
from typing import Union, List
import cv2
import numpy as np
from numpy.typing import NDArray
from pycv.imageutils.colour import get_colour


def pad_image(img: NDArray, dst_size: Tuple[int, int]):
    """

    :param img:
    :param dst_size:
    :return:
    """
    dst_width, dst_height = dst_size
    assert (img.shape[0] <= dst_height and img.shape[1] <= dst_width)
    top = (dst_height - img.shape[0]) // 2
    bottom = dst_height - img.shape[0] - top
    left = (dst_width - img.shape[1]) // 2
    right = dst_width - img.shape[1] - left
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT)
    assert (img.shape[0] == dst_height and img.shape[1] == dst_width)
    return img


def crop_image(img: NDArray, dst_size: Tuple[int, int]):
    dst_width, dst_height = dst_size
    assert (img.shape[0] >= dst_height and img.shape[1] >= dst_width)

    y0 = (img.shape[0] - dst_height) // 2
    x0 = (img.shape[1] - dst_width) // 2

    img = np.copy(img[y0:y0+dst_height, x0:x0+dst_width])
    assert (img.shape[0] == dst_height and img.shape[1] == dst_width)
    return img


def resize_image(img: NDArray, dst_size: Tuple[int, int],
                 exact_interpolation: bool = False, mode: str = 'pad'):
    """
    Resize an image_safe_zone to the given size, maintaining aspect ration by either cropping or padding the image_safe_zone based
    on the supplied arguments
    :param img:
    :param dst_size:
    :param exact_interpolation: if true (e.g if the exact value is important, such as in masks),
                                then use cv2.INTER_NEAREST
    :param mode: if 'pad', add a black border to maintain aspect ratio when resizing. If 'crop', trim the edges to
                 maintain aspect ratio
    :return:
    """
    assert(mode == 'pad' or mode == 'crop')
    src_height, src_width = img.shape[:2]
    dst_width, dst_height = dst_size
    k_x = dst_width / src_width
    k_y = dst_height / src_height
    interp_mode = cv2.INTER_NEAREST if exact_interpolation else cv2.INTER_CUBIC

    if mode == 'pad':
        if k_x < k_y:  # if the image_safe_zone needs to be resized more in the y direction
            # scale image_safe_zone so that width = dst_width and height < dst_height: then pad in y direction
            intermediate_size = (int(src_width * k_x), int(src_height * k_x))
        else:  # if the image_safe_zone needs to be resized more in the x direction
            # scale image_safe_zone so that height = dst_height and width < dst_width: then pad in x direction
            intermediate_size = (int(src_width * k_y), int(src_height * k_y))
        img = cv2.resize(img, intermediate_size, interpolation=interp_mode)
        img = pad_image(img, dst_size)
        return img
    elif mode == 'crop':
        if k_x < k_y:  # if the image_safe_zone needs to be resized more in the y direction
            # scale image_safe_zone so that height = dst_height and width > dst_width: then crop in x direction
            intermediate_size = (int(src_width * k_y), int(src_height * k_y))
        else:  # if the image_safe_zone needs to be resized more in the x direction
            # scale image_safe_zone so that width = dst_width and height > dst_height: then crop in y direction
            intermediate_size = (int(src_width * k_x), int(src_height * k_x))
        img = cv2.resize(img, intermediate_size, interpolation=interp_mode)
        img = crop_image(img, dst_size)
        return img
    else:
        raise Exception("Unknown mode -", mode)


def normalise_image(src, dark_img=None, img_min=None, img_max = None):
    """

    :param src:
    :param dark_img:
    :param img_min:
    :param img_max:
    :return:
    """
    dst = np.copy(src)
    if dark_img is None:
        dark_img = np.zeros(src.shape)
    dst -= dark_img
    if img_min is None:
        mid_val = (np.min(dst)+np.max(dst))/2.0
        img_min = np.mean(dst[dst < 0.7 * mid_val])
    if img_max is None:
        mid_val = (np.min(dst)+np.max(dst))/2.0
        img_max = np.mean(dst[dst > 1.3 * mid_val])
    dst -= img_min
    dst /= (img_max - img_min)
    return dst