from __future__ import annotations

import cv2
import numpy as np
import torch
from typing import Tuple, Dict, Union, List
from numpy.typing import NDArray
from ..core.colours import get_colour


def setup_namespace():
    return


def mask_to_rgb(mask):
    assert(len(mask.shape) == 2)
    mask = mask.astype(np.uint8)
    max_value = np.max(mask)
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for i in range(max_value+1):
        rgb[mask == i] = get_colour(i)

    return rgb


def get_pixel_predictions_from_output(preds: torch.FloatTensor):
    if len(preds.shape) == 3:
        dim = 0
    elif len(preds.shape) == 4:
        dim = 1
    else:
        raise Exception("Invalid shape passed- ", preds.shape)

    n_classes = preds.shape[dim]
    if n_classes == 1:
        return torch.sigmoid(preds).round().squeeze(dim=dim)
    else:
        return torch.softmax(preds, dim=dim).argmax(dim=dim)


def overlay_predictions_on_image(img: NDArray, prediction: NDArray, alpha=0.3):
    assert (isinstance(img, np.ndarray) and len(img.shape) == 3 and img.dtype == np.uint8)
    assert (isinstance(prediction, np.ndarray) and len(prediction.shape) == 2 and prediction.dtype == np.uint8)
    # img_dst = np.zeros(img.shape, dtype=np.uint8)
    # img_dst[:] = img
    img = np.copy(img)
    img = np.ascontiguousarray(img, dtype=np.uint8)

    mask_pred = np.zeros(img.shape, dtype=np.uint8)

    for i in range(np.max(prediction) + 1):
        mask_pred[prediction == i] = get_colour(i)

    img_mask = ((1.0-alpha)*img + alpha*mask_pred).astype(np.uint8)

    return img_mask


def get_unique_values(img: NDArray) -> List[Union[Tuple[int, int, int], int]]:
    if len(img.shape) == 2:
        unique_values_arr = np.unique(img.reshape(-1), axis=0).tolist()
    elif len(img.shape) == 3:
        unique_values_arr = np.unique(img.reshape(-1, img.shape[-1]), axis=0).tolist()
    else:
        raise Exception("Invalid shape: {}".format(img.shape))
    unique_values = []
    for val in unique_values_arr:
        if isinstance(val, list):
            unique_values.append(tuple(val))
        else:
            unique_values.append(val)
    return unique_values

def convert_color_mask_to_class_mask(mask_color: NDArray, classes: Dict[int, Tuple[int, int, int]],
                                     flip_channels=True) -> NDArray:
    """

    :param mask_color:
    :param classes:
    :param flip_channels: if the
    :return:
    """

    if len(mask_color.shape) == 3 and mask_color.shape[2] == 3 and flip_channels:
        mask_color = cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB)

    unique_values = get_unique_values(mask_color)
    class_colors = classes.values()

    for unique_value in unique_values:
        assert(unique_value in class_colors), "{} not found in specified classes. " \
                                              "supplied colors = {}".format(unique_value, class_colors)

    mask_class = np.zeros(mask_color.shape[:2], dtype=np.uint8)

    for class_index in classes:
        color = classes[class_index]
        mask_class[np.where(np.all(mask_color == color, axis=-1))] = class_index
    return mask_class


def convert_class_mask_to_color_mask(mask_class: NDArray, classes: Dict[int, Union[int, Tuple[int, int, int]]],
                                     flip_channels=True) -> NDArray:
    """

    :param mask_class:
    :param classes:
    :param flip_channels:
    :return:
    """
    unique_values = get_unique_values(mask_class)
    for unique_value in unique_values:
        assert(unique_value in classes), "{} not found in specified classes. " \
                                         "Values supplied = {}".format(unique_value, classes.keys())

    mask_color = np.zeros((mask_class.shape[0], mask_class.shape[1], 3), dtype=np.uint8)

    for class_index in classes:
        color = classes[class_index]
        mask_color[mask_class == class_index] = color

    if flip_channels:
        mask_color = cv2.cvtColor(mask_color, cv2.COLOR_RGB2BGR)

    return mask_color
