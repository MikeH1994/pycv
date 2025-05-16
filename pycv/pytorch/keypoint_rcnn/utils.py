import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from typing import List, Tuple, Union, Dict
import mytorch.core as core
import torchvision.ops.boxes as bops
import albumentations as alb
from albumentations.augmentations.crops.transforms import Crop


def overlay_points_on_image(image: NDArray, keypoints_list: List[NDArray] = None, bboxes: List[NDArray] = None,
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


def albumentations_keypoints_to_rcnn_format(keypoints: List[List[float]], keypoints_visibility: List[int],
                                            n_keypoints_per_object: int, bboxes: List[List[float]],
                                            bboxes_indices: List[int], bbox_classes: List[int], width, height):
    """
    Given the keypoints and bounding boxes returned from albumentations, convert these to a format usable to the RCNN
    network.
    The output is a dict of numpy arrays:
        "boxes": shape: (n_bboxes, 4), dtype: float64
        "keypoints": shape: (n_bboxes, n_keypoints_per_class, 3), dtype: float64
        "labels": shape: (n_bboxes, ), dtype: int64
        "area": shape: (n_bboxes, ), dtype: float64
        "iscrowd": shape: (n_bboxes, ), dtype: bool

    :param keypoints: a list of keypoints, of the form [[x0, y0], [x1, y1], [x2, y2] ... ]. Shape (n_tot_keypoints, 2).
        Here n_tot_keypoints = n_keypoints_per_class*n_input_bboxes
    :param keypoints_visibility: a list describing the visibility of each point. 1 indicates point is visible, 0 is not
        visible. Non-visible points do not contibute to the localisation loss. Shape (n_tot_keypoints, )
    :param bboxes: a list of bboxes in the form [[x0, y0, x1, y1], [x0, y0, x1, y1]]
    :param bboxes_indices: a list of the indices of each bbox. If an index is missing, this indicates that the
        bounding box passed to the albumentations transformations was not returned (e.g. because it was now outside the
        image). Of the form [0, 1, 2, 4]
    :param bbox_classes: the indices
    :return:
    """
    assert(len(bboxes) == len(bboxes_indices) == len(bbox_classes)), "bboxes, bbox_indices and bbox_classes should all" \
                                                                     "have the same length - ({}vs{}vs{})".format(
        len(bboxes),len(bboxes_indices),len(bbox_classes))
    assert(len(keypoints) == len(keypoints_visibility))
    assert(len(keypoints) % n_keypoints_per_object == 0), "The number of keypoints passed is not a multiple of the " \
                                                          "number of keypoints per object- {} keypoints and {} " \
                                                          "keypoints per object".format(len(keypoints),
                                                                                        n_keypoints_per_object)
    n_src_objects = len(keypoints) // n_keypoints_per_object
    n_dst_objects = len(bboxes)
    for i, index in enumerate(bboxes_indices):
        assert(isinstance(index, int)), "bbox index {} is not an integer- (element {})".format(index, i)
        assert(0 <= index < n_src_objects), "bbox index {} should be in the range 0<=x<n_input_objects, where " \
                                            "n_input_objects is {}".format(index, i)
        assert(bboxes_indices.count(index) == 1), "bbox index {} appeared more than once in list".format(index)

    keypoints = np.array(keypoints).reshape((-1, n_keypoints_per_object, 2))
    keypoints_visibility = np.array(keypoints_visibility, dtype=bool).reshape((-1, n_keypoints_per_object))
    dst_keypoints = np.zeros((n_dst_objects, n_keypoints_per_object, 3), dtype=np.float64)
    dst_bboxes = np.array(bboxes)
    dst_labels = np.array(bbox_classes, dtype=np.int64)
    dst_area = np.zeros(n_dst_objects, dtype=np.float64)
    dst_iscrowd = np.full(n_dst_objects, fill_value=False,dtype=np.int64)
    for i, bbox_index in enumerate(bboxes_indices):
        dst_keypoints[i][:, :2] = keypoints[bbox_index]
        visible = keypoints_visibility[bbox_index]
        visible[(keypoints[bbox_index][:, 0] < 0) | (keypoints[bbox_index][:, 0] > width-1)] = False
        visible[(keypoints[bbox_index][:, 1] < 0) | (keypoints[bbox_index][:, 1] > height-1)] = False
        dst_keypoints[i][:, 2] = visible
    for i in range(n_dst_objects):
        bbox = dst_bboxes[i]
        area = (bbox[2] - bbox[0])*(bbox[3] - bbox[1])
        dst_area[i] = area

    return {
        "boxes": dst_bboxes,
        "keypoints": dst_keypoints,
        "labels": dst_labels,
        "area": dst_area,
        "iscrowd": dst_iscrowd
    }


def albumentations_bboxes_to_numpy_bboxes(bboxes: List[List]):
    return np.array(bboxes)


def numpy_bboxes_to_albumentations_bboxes(bboxes: NDArray):
    raise Exception("TODO")


def create_bbox_from_keypoints(keypoints: Union[NDArray, List], k_x: float = 0.0, k_y: float = 0.0, img_size=None) -> List[float]:
    if isinstance(keypoints, list):
        keypoints = np.array(keypoints, dtype=np.float32)
    assert(len(keypoints.shape) == 2)
    assert(keypoints.shape[1] == 2 or keypoints.shape[1] == 3)
    min_x = np.min(keypoints[:, 0])
    max_x = np.max(keypoints[:, 0])
    min_y = np.min(keypoints[:, 1])
    max_y = np.max(keypoints[:, 1])
    dx = max_x - min_x
    dy = max_y - min_y
    min_x -= k_x / 2.0 * dx
    max_x += k_x / 2.0 * dx
    min_y -= k_y / 2.0 * dy
    max_y += k_y / 2.0 * dy

    if min_x <= 0.0:
        min_x = 0.0
    if min_y <= 0.0:
        min_y = 0.0

    if img_size is not None:
        width, height = img_size
        if max_x > width - 1.0:
            max_x = width - 1.0
        if max_y > height - 1.0:
            max_y = height - 1.0

    bbox = [min_x, min_y, max_x, max_y]
    return bbox


def find_skeletons_corresponding_to_bbox(bboxes, bbox_labels, skeletons, skeleton_labels, label_ids: Dict[str, int] = None):
    assert(len(bboxes) == len(bbox_labels) == len(skeletons) == len(skeleton_labels))

    skeletons_dst = [None for _ in range(len(bboxes))]
    for i in range(len(skeletons)):
        skeleton = np.array(skeletons[i])
        skeleton_label = label_ids[skeleton_labels[i]] if label_ids is not None else skeleton_labels[i]
        skeleton_bbox = create_bbox_from_keypoints(skeleton)

        best_index = None
        best_iou = -np.inf
        for j in range(len(bboxes)):
            bbox = bboxes[j]
            bbox_label = label_ids[bbox_labels[j]] if label_ids is not None else bbox_labels[j]

            if skeleton_label != bbox_label:
                continue

            iou = bops.box_iou(torch.tensor([bbox], dtype=torch.float),
                               torch.tensor([skeleton_bbox], dtype=torch.float)).item()

            if iou > best_iou:
                best_iou = iou
                best_index = j

        assert(best_index is not None), "Could not find a bbox label corresponding to skeleton label {}".format(skeleton_label)
        assert(skeletons_dst[best_index] is None)
        skeletons_dst[best_index] = skeleton.tolist()

    return {
        "skeletons": skeletons_dst,
        "bboxes": bboxes,
        "labels": bbox_labels
    }


