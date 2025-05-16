from __future__ import annotations
import albumentations as alb
import cv2
from albumentations.pytorch import ToTensorV2
from albumentations.core.composition import Compose
from typing import Dict
from .configuration import Configuration, ImageTransformParams


def get_image_transforms(image_params: ImageTransformParams, normalize: bool = True, augment_data: bool = False,
                         use_keypoints: bool = False, use_bboxes: bool = False, to_tensor: bool = True,
                         additional_targets: Dict[str, str] = None) -> Compose:

    t = []
    if image_params is not None:
        if image_params.always_clahe:
            t += [alb.CLAHE(p=1.0)]
        if augment_data:
            if image_params.p_hor_flip > 0.0:
                t += [alb.HorizontalFlip(p=image_params.p_hor_flip)]
            if image_params.p_vert_flip > 0.0:
                t += [alb.VerticalFlip(p=image_params.p_vert_flip)]

            p_clahe = image_params.p_clahe if image_params.always_clahe is False else 0.0
            alpha_min = image_params.sharpen_alpha_min
            alpha_max = image_params.sharpen_alpha_max

            if image_params.p_sharpen > 0.0:
                t += [alb.Sharpen(alpha=(alpha_min, alpha_max), lightness=(1.0, 1.0), p=image_params.p_sharpen)]
            if p_clahe > 0.0:
                t += [alb.CLAHE(p=image_params.p_clahe)]
            if image_params.p_rbc > 0.0:
                t += [alb.RandomBrightnessContrast(brightness_limit=image_params.rbc_contrast_limit,
                                                   contrast_limit=image_params.rbc_contrast_limit,
                                                   p=image_params.p_rbc)]
            if image_params.p_affine > 0.0:
                tx = image_params.affine_transl
                sx = image_params.affine_scale
                scale = sx if image_params.affine_keep_aspect_ratio else {"x": sx, "y": sx}
                rotate = (int(image_params.affine_rotate[0]), int(image_params.affine_rotate[1]))
                shear = (int(image_params.affine_shear[0]), int(image_params.affine_shear[1]))
                fit_output = image_params.affine_fit_output
                t += [alb.Affine(translate_percent=tx, scale=scale, p=image_params.p_affine,
                                 rotate=rotate, mode=cv2.BORDER_CONSTANT, fit_output=fit_output,
                                 shear=shear, keep_ratio=image_params.affine_keep_aspect_ratio)]

            if image_params.p_random_crop > 0:
                t += [alb.RandomCrop(width=image_params.image_width, height=image_params.image_height,
                                     p=image_params.p_random_crop)]
        if image_params.image_width is not None and image_params.image_height is not None:
            t += [alb.Resize(width=image_params.image_width, height=image_params.image_height)]

        if normalize:
            t += [alb.Normalize(mean=image_params.mean, std=image_params.std, max_pixel_value=1.0)]
    if to_tensor:
        t += [ToTensorV2()]

    additional_targets = {} if additional_targets is None else additional_targets
    keypoint_params = alb.KeypointParams(format='xy', remove_invisible=False) if use_keypoints else None #
    bbox_params = alb.BboxParams(format='pascal_voc', label_fields=['class_labels']) if use_bboxes else None
    return alb.Compose(t, keypoint_params=keypoint_params, bbox_params=bbox_params,
                       additional_targets=additional_targets)

