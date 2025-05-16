from __future__ import annotations
from dataclasses import dataclass
import os
from typing import Iterable
from dataclasses import dataclass
import torch
import random
from typing import Tuple, Union, List
from datetime import datetime


@dataclass
class BaseParams:
    def __str__(self):
        dst = "  \n"
        d = vars(self)
        for key in d:
            dst += "**{}**: {}  \n".format(key, d[key])
        return dst


@dataclass
class ImageTransformParams(BaseParams):
    image_width: Union[int, None] = None
    image_height: Union[int, None] = None
    mean: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    always_clahe: bool = False
    p_hor_flip: float = 0.0
    p_vert_flip: float = 0.0
    p_clahe: float = 0.0
    p_random_crop: float = 0.0

    p_rbc: float = 0.0
    rbc_contrast_brightness_limit: float = 0.15
    rbc_contrast_limit: float = 0.15

    """
    Affine transforms
    affine_transl_px - the translate the image by a fraction of the image- (-affine_transl_px, affine_transl_px)
    
    affine_fit_output: fit the image, so that black areas are not seen. Overwrites translation
    """
    p_affine: float = 0.0
    affine_transl: Tuple[float, float] = (0.0, 0.0)
    affine_rotate: Tuple[int, int] = (0, 0)
    affine_scale: Tuple[float, float] = (1.0, 1.0)
    affine_shear: Tuple[int, int] = (0, 0)
    affine_keep_aspect_ratio: bool = True
    affine_fit_output: bool = False
    """
    
    """
    p_sharpen: float = 0.0
    sharpen_alpha_min: float = 0.2
    sharpen_alpha_max: float = 0.5


@dataclass
class TrainingParams(BaseParams):
    n_epochs: int = 9999
    start_epoch: int = 0
    clip_dataset: int = None
    optimizer_type: str = "Adam"
    scheduler_type: str = "ExponentialLR"
    loss_fn: str = "default"
    model_type: str = "default"
    init_lr: float = 0.001
    lr_decay: float = 0.9
    momentum: float = 0.9
    weight_decay: float = 0.0001
    lr_step_milestones: Iterable = (30, 40)
    augment_training_data: bool = True
    display_images: bool = False


@dataclass
class Configuration(BaseParams):
    root: str = ""
    output_dir: str = os.path.join("output", datetime.now().strftime("%Y%m%d_%H%M%S"))
    load_model_path: str = None
    model_file_name: str = "model.pt"

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    cudnn_deterministic: int = True
    cudnn_benchmark_enabled: int = True
    seed: int = 1234
    num_workers: int = 8
    batch_size: int = 8

    training_params: TrainingParams = TrainingParams()
    image_transform_params: ImageTransformParams = ImageTransformParams()

    train_dataset_summary: str = ""
    val_dataset_summary: str = ""
    argv: str = ""
    comment: str = ""


@dataclass
class ClassificationConfiguration(Configuration):
    img_extension: str = ".jpg"
    n_classes: int = None


@dataclass
class SegmentationConfiguration(Configuration):
    img_extension: str = ".jpg"
    mask_extension: str = ".png"
    n_classes: int = None


@dataclass
class KeypointRCNNConfiguration(Configuration):
    img_extension: str = ".jpg"
    annotations_extension: str = ".txt"
    n_classes: int = None
    n_keypoints: int = None
    n_trainable_backbone_layers:int = 3
    all_layers_trainable: bool = False
    pretrained_backbone: bool = True
    anchor_generator_sizes: Tuple = None
    anchor_generator_aspect_ratios: Tuple = None
    score_boundary: float = 0.7
    nms_boundary: float = 0.3


@dataclass
class KeypointConfiguration(Configuration):
    n_keypoints: int = None
    n_dense_layers: int = 1
    all_layers_trainable: bool = False
    keypoint_names: List[str] = None
