import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from typing import List, Tuple, Union, Dict
import mytorch.core as core
import torchvision.ops.boxes as bops
from albumentations.pytorch import ToTensorV2
import albumentations as alb
from albumentations.augmentations.crops.transforms import Crop


def crop_image_around_bbox(image, bbox: List[float], keypoints: List[List[float]] = None, dx: int = 0, dy: int = 0):
    """

    :param image:
    :param bbox:
    :param keypoints:
    :param dx:
    :param dy:
    :return:
    """
    x_min, y_min, x_max, y_max = [int (v) for v in bbox]
    x_min, x_max = core.clamp(x_min - dx, 0, image.shape[1] - 1), core.clamp(x_max + dx, 0, image.shape[1] - 1)
    y_min, y_max = core.clamp(y_min - dy, 0, image.shape[0] - 1), core.clamp(y_max + dy, 0, image.shape[0] - 1)

    keypoint_params = alb.KeypointParams(format='xy', remove_invisible=False) if keypoints is not None else None
    transform = alb.Compose(
        [Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, always_apply=True)],
        keypoint_params=keypoint_params)
    if keypoints is not None:
        f = transform(image=image, keypoints=keypoints)
    else:
        f = transform(image=image)
    image = f["image"]
    keypoints = f["keypoints"] if "keypoints" in f else None

    return {
        "image": image,
        "keypoints": keypoints,
        "corners": [x_min, y_min, x_max, y_max]
    }


def resize_image(image, dst_size: Tuple[int, int], masks: List[List] = None, keypoints: List[List[List]] = None,
                 mode: str = 'pad'):
    """

    :param image:
    :param dst_size:
    :param masks:
    :param keypoints:
    :param mode:
    :return:
    """
    assert(mode == 'pad' or mode == 'crop' or mode == 'resize')
    masks = [] if masks is None else masks
    keypoints = [] if keypoints is None else keypoints
    src_height, src_width = image.shape[:2]
    dst_width, dst_height = dst_size
    k_x = dst_width / src_width
    k_y = dst_height / src_height

    additional_targets = {}
    for i in range(len(masks)):
        additional_targets["mask{}".format(i)] = "mask"
    for i in range(len(keypoints)):
        additional_targets["keypoints{}".format(i)] = "keypoints"
    t = []
    if mode == "pad":
        if k_x < k_y:  # if the image needs to be resized more in the y direction
            # scale image so that width = dst_width and height < dst_height: then pad in y direction
            intermediate_size = (int(src_width * k_x), int(src_height * k_x))
        else:  # if the image needs to be resized more in the x direction
            # scale image so that height = dst_height and width < dst_width: then pad in x direction
            intermediate_size = (int(src_width * k_y), int(src_height * k_y))
        t = [alb.Resize(width = intermediate_size[0], height = intermediate_size[1]),
             alb.PadIfNeeded(min_height=dst_height, min_width=dst_width, border_mode=cv2.BORDER_CONSTANT)]
    elif mode == "crop":
        if k_x < k_y:  # if the image needs to be resized more in the y direction
            # scale image so that height = dst_height and width > dst_width: then crop in x direction
            intermediate_size = (int(src_width * k_y), int(src_height * k_y))
        else:  # if the image needs to be resized more in the x direction
            # scale image so that width = dst_width and height > dst_height: then crop in y direction
            intermediate_size = (int(src_width * k_x), int(src_height * k_x))
        t = [alb.Resize(width = intermediate_size[0], height = intermediate_size[1]),
             alb.CenterCrop(height=dst_height, width=dst_width)]
    else:
        t = [alb.Resize(width = dst_width, height = dst_height)]

    keypoint_params = alb.KeypointParams(format='xy', remove_invisible=False) if len(keypoints) > 0 else None
    transform = alb.Compose(t, keypoint_params=keypoint_params, additional_targets=additional_targets)

    additional_targets = {}
    for i in range(len(masks)):
        additional_targets["mask{}".format(i)] = masks[i]
    for i in range(len(keypoints)):
        additional_targets["keypoints{}".format(i)] = keypoints[i]

    if len(keypoints) > 0:
        f = transform(image=image, keypoints=keypoints[0], **additional_targets)
    else:
        f = transform(image=image, **additional_targets)
    dst_image = f["image"]
    dst_keypoints = []
    for i in range(len(keypoints)):
        dst_keypoints.append([list(v) for v in f["keypoints{}".format(i)]])
    dst_masks = []
    for i in range(len(masks)):
        dst_masks.append(f["mask{}".format(i)])

    assert(dst_image.shape[0] == dst_height and dst_image.shape[1] == dst_width)

    return {
        "image": dst_image,
        "keypoints": dst_keypoints,
        "masks": dst_masks
    }


def pad_image_old(img: NDArray, dst_size: Tuple[int, int]):
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


def crop_image_old(img: NDArray, dst_size: Tuple[int, int]):
    dst_width, dst_height = dst_size
    assert (img.shape[0] >= dst_height and img.shape[1] >= dst_width)

    y0 = (img.shape[0] - dst_height) // 2
    x0 = (img.shape[1] - dst_width) // 2

    img = np.copy(img[y0:y0+dst_height, x0:x0+dst_width])
    assert (img.shape[0] == dst_height and img.shape[1] == dst_width)
    return img


def resize_image_old(img: NDArray, dst_size: Tuple[int, int], exact_interpolation: bool = False,
                     mode: str = 'pad'):
    """
    Resize an image to the given size, maintaining aspect ration by either cropping or padding the image based
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
        if k_x < k_y: # if the image needs to be resized more in the y direction
            # scale image so that width = dst_width and height < dst_height: then pad in y direction
            intermediate_size = (int(src_width * k_x), int(src_height * k_x))
        else: # if the image needs to be resized more in the x direction
            # scale image so that height = dst_height and width < dst_width: then pad in x direction
            intermediate_size = (int(src_width * k_y), int(src_height * k_y))
        img = cv2.resize(img, intermediate_size, interpolation=interp_mode)
        img = pad_image_old(img, dst_size)
        return img
    elif mode == 'crop':
        if k_x < k_y: # if the image needs to be resized more in the y direction
            # scale image so that height = dst_height and width > dst_width: then crop in x direction
            intermediate_size = (int(src_width * k_y), int(src_height * k_y))
        else:  # if the image needs to be resized more in the x direction
            # scale image so that width = dst_width and height > dst_height: then crop in y direction
            intermediate_size = (int(src_width * k_x), int(src_height * k_x))
        img = cv2.resize(img, intermediate_size, interpolation=interp_mode)
        img = crop_image_old(img, dst_size)
        return img
    else:
        raise Exception("Unknown mode -", mode)


def get_bounding_boxes_around_mask(mask: NDArray, n_dilation=3) -> List[Tuple[int, List[float]]]:
    """

    :param mask: a mask of shape (height, width). Should be of type np.uint8 or np.
    :return:
    """
    assert(len(mask.shape) == 2)
    assert(mask.dtype in [np.uint8, np.uint16])

    bboxes = []

    for i in range(1, np.max(mask) + 1):
        binary_mask = np.zeros(mask.shape, dtype=np.uint8)
        binary_mask[mask == i] = 1

        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=n_dilation)
        binary_mask = cv2.erode(binary_mask, kernel, iterations=n_dilation)

        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # contours is a tuple containing each contour
        # each contour is an array of shape (n, 1, 2)
        # for each contour, find a bounding box surrounding it and add to bboxes
        for contour in contours:
            min_x = np.min(contour[:, :, 0])
            max_x = np.max(contour[:, :, 0])
            min_y = np.min(contour[:, :, 1])
            max_y = np.max(contour[:, :, 1])
            bboxes.append((i, [min_x, min_y, max_x, max_y]))

    return bboxes


def cropped_keypoints_to_uncropped_keypoints(cropped_image_size: Tuple[int, int], bounding_box: List[float],
                                             keypoints: List[float]):
    for i in range(len(keypoints)):
        x_min, y_min, x_max, y_max = keypoints[i]

