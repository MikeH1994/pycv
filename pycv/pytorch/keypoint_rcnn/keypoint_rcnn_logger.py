from __future__ import annotations
import torch
import numpy as np
from typing import Dict, List, Union
from ..base.base_epoch_logger import BaseEpochLogger
from ..base.base_metrics import BaseLossFn, BaseScoreFn
from ..core.configuration import Configuration, ImageTransformParams
from ..core.utils import tensor_img_to_numpy
from ..keypoint_rcnn.utils import overlay_points_on_image
import cv2


class KeypointRCNNLogger(BaseEpochLogger):
    def __init__(self, loss_fn: BaseLossFn, num_classes,
                 additional_metrics: Dict[str, Union[BaseScoreFn, BaseScoreFn]] = None,
                 class_names: List[str] = None):
        super().__init__(loss_fn, num_classes, additional_metrics, class_names)

    def batch_to_images(self, imgs: torch.FloatTensor, preds: List[Dict], config: Configuration):
        """
        :param imgs: shape: (B, h, w)
        :param preds: shape: (B, num_classes, h, w)
        :param targets: (B, h, w)
        :param config: config
        :return:
        """
        batch_size = imgs.shape[0]
        imgs = imgs.to('cpu').detach()

        images = []
        for batch_idx in range(batch_size):
            keypoints = preds[batch_idx]['keypoints']
            boxes = preds[batch_idx]['boxes']
            classes = preds[batch_idx]['classes']
            img = tensor_img_to_numpy(imgs[batch_idx], convert_to_8_bit=True,
                                      scale_factor=config.image_transform_params.std,
                                      offset=config.image_transform_params.mean)
            img = overlay_points_on_image(img, keypoints, boxes, classes)
            images.append(img)
        return images