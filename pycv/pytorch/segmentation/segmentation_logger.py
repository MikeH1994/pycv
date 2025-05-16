from __future__ import annotations
import torch
import numpy as np
from typing import Dict, List, Union
from ..base.base_epoch_logger import BaseEpochLogger
from ..base.base_metrics import BaseLossFn, BaseScoreFn
from ..core.configuration import Configuration, ImageTransformParams
from ..core.utils import tensor_img_to_numpy
from .utils import get_pixel_predictions_from_output, overlay_predictions_on_image


class SegmentationEpochLogger(BaseEpochLogger):
    def __init__(self, loss_fn: BaseLossFn, num_classes,
                 additional_metrics: Dict[str, Union[BaseScoreFn, BaseScoreFn]] = None,
                 class_names: List[str] = None):
        super().__init__(loss_fn, num_classes, additional_metrics, class_names)

    def batch_to_images(self, imgs: torch.FloatTensor, preds: torch.FloatTensor, config: Configuration):
        """
        :param imgs: shape: (B, h, w)
        :param preds: shape: (B, num_classes, h, w)
        :param targets: (B, h, w)
        :param config: config
        :return:
        """
        batch_size = imgs.shape[0]
        imgs = imgs.to('cpu').detach()
        preds = preds.to('cpu').detach()

        images = []
        for batch_idx in range(batch_size):
            img = tensor_img_to_numpy(imgs[batch_idx], convert_to_8_bit=True,
                                      scale_factor=config.image_transform_params.std, offset=config.image_transform_params.mean)
            pred = preds[batch_idx].numpy().astype(np.uint8)
            img = overlay_predictions_on_image(img, pred)
            images.append(img)
        return images
