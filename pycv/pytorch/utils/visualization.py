import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from typing import List, Tuple, Union, Dict
import mytorch.core as core
import torchvision.ops.boxes as bops
import albumentations as alb
from albumentations.augmentations.crops.transforms import Crop

"""def overlay_points_on_image(image: NDArray, keypoints_list: List[NDArray] = None, bboxes: List[NDArray] = None,
                            classes: List = None,  points_normalised=False, radius=3, dx=5) -> NDArray:
    assert(len(image.shape) == 3 and image.shape[2] == 3), "Image suppled should be rgb- shape: {}".format(image.shape)
    assert(image.dtype==np.uint8), "Image should be uint8"
    image = np.copy(image)
    image = np.ascontiguousarray(image, dtype=np.uint8) # I don't know why but cv2.rectangle fails otherwise
    height, width = image.shape[:2]

    # draw bounding boxes
    if bboxes is not None:
        n_objects = len(bboxes)
        for i in range(n_objects):
            bbox = bboxes[i]
            start_point = (int(bbox[0]), int(bbox[1]))
            end_point = (int(bbox[2]+1), int(bbox[3]+1))
            if classes is not None:
                color = core.get_colour(i+1)
            else:
                color = (255, 0, 0)
            image = cv2.rectangle(image, start_point, end_point, color, 2)

    # draw keypoints
    if keypoints_list is not None:
        n_objects = len(keypoints_list)
        for i in range(n_objects):
            keypoints = keypoints_list[i]
            n_points = keypoints.shape[0]
            for n in range(n_points):
                point = keypoints[n]
                if points_normalised:
                    centre = (int(point[0]*width), int(point[1]*height))
                else:
                    centre = (int(point[0]), int(point[1]))
                if classes is not None:
                    color = core.get_colour(i + 1)
                else:
                    color = (255, 0, 0)
                image = cv2.circle(image, centre, radius, color, -1)
                cv2.putText(image, '{}'.format(n+1), org=(centre[0] + dx, centre[1] + dx), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=color, thickness=1, lineType=2)
    return image
"""

def overlay_points_on_image(image: NDArray, keypoints_list: Union[List[NDArray], NDArray, torch.Tensor],
                            points_normalised=False, radius=3, dx=5, label=True, color=None) -> NDArray:
    assert(len(image.shape) == 3 and image.shape[2] == 3), "Image suppled should be rgb- shape: {}".format(image.shape)
    assert(image.dtype==np.uint8), "Image should be uint8"
    image = np.copy(image)
    image = np.ascontiguousarray(image, dtype=np.uint8) # I don't know why but cv2.rectangle fails otherwise
    height, width = image.shape[:2]
    keypoints_list = torch.tensor(keypoints_list)
    if len(keypoints_list.shape) == 2:
        keypoints_list = keypoints_list.reshape((1, keypoints_list.shape[0], keypoints_list.shape[1]))
    assert(len(keypoints_list.shape) == 3)
    assert(keypoints_list.shape[-1] == 2 or keypoints_list.shape[-1] == 3)

    # draw keypoints
    n_objects = len(keypoints_list)
    for i in range(n_objects):
        keypoints = keypoints_list[i]
        n_points = keypoints.shape[0]
        for n in range(n_points):
            point = keypoints[n]
            if points_normalised:
                centre = (int(point[0]*width), int(point[1]*height))
            else:
                centre = (int(point[0]), int(point[1]))

            point_color = core.get_colour(i + 1) if color is None else color

            image = cv2.circle(image, centre, radius, point_color, -1)
            if label:
                cv2.putText(image, '{}'.format(n+1), org=(centre[0] + dx, centre[1] + dx), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=color, thickness=1, lineType=2)
    return image


def overlay_bboxes_on_image(image: NDArray, bboxes: List[List[float]]):
    """

    :param image:
    :param bboxes:
    :return:
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    assert(len(image.shape) == 3 and image.shape[2] == 3), "Image suppled should be rgb- shape: {}".format(image.shape)
    assert(image.dtype==np.uint8), "Image should be uint8"
    image = np.copy(image)
    image = np.ascontiguousarray(image, dtype=np.uint8) # I don't know why but cv2.rectangle fails otherwise


    # draw bounding boxes
    n_objects = len(bboxes)
    for i in range(n_objects):
        bbox = bboxes[i]
        start_point = (int(bbox[0]), int(bbox[1]))
        end_point = (int(bbox[2]+1), int(bbox[3]+1))
        color = core.get_colour(i+1)
        image = cv2.rectangle(image, start_point, end_point, color, 2)
    return image