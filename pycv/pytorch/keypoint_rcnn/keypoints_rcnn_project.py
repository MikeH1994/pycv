from typing import Type, Tuple
from torch.utils.data import DataLoader
from ..base import BaseProject, BaseLossFn, BaseScoreFn, BaseModel, BaseEpochLogger, BaseDataset, BaseTrainer, BaseDatasetViewer
from ..core import KeypointRCNNConfiguration, load_model, Configuration
from .metrics.keypoint_rcnn_metrics import (KeypointRCNNKeypointLoss, KeypointRCNNRPNBoxRegLoss,
                                            KeypointRCNNObjectnessLoss, KeypointRCNNClassifierLoss)
from .metrics.keypoint_rcnn_loss import KeypointRCNNLoss
from .keypoint_rcnn_logger import KeypointRCNNLogger
from .keypoint_rcnn_trainer import KeypointRCNNTrainer
from .keypoint_rcnn_dataset_viewer import KeypointRCNNDatasetViewer
from .models import KeypointRCNN


class KeypointRCNNProject(BaseProject):
    def __init__(self, train_dataset: BaseDataset, val_dataset: BaseDataset,
                 config: Configuration, calc_mean_and_std = True, test_dataset: BaseDataset = None):
        super().__init__(train_dataset, val_dataset, config, calc_mean_and_std=calc_mean_and_std,
                         test_dataset=test_dataset)

    def get_epoch_logger(self, config: KeypointRCNNConfiguration) -> BaseEpochLogger:
        additional_metrics = {
            "loss_classifier": KeypointRCNNClassifierLoss(),
            "loss_box_reg": KeypointRCNNRPNBoxRegLoss(),
            "loss_keypoint": KeypointRCNNKeypointLoss(),
            "loss_objectness": KeypointRCNNObjectnessLoss(),
            "loss_rpn_box_reg": KeypointRCNNRPNBoxRegLoss(),
        }
        loss_fn = self.get_loss_fn(config)
        return KeypointRCNNLogger(loss_fn, config.n_classes, additional_metrics) #

    def get_loss_fn(self, config: KeypointRCNNConfiguration) -> BaseLossFn:
        loss_fn_name = config.training_params.loss_fn
        loss_fns = {
            "keypoint-rcnn-loss": KeypointRCNNLoss(),
            "default": KeypointRCNNLoss(),
        }
        if loss_fn_name in loss_fns:
            return loss_fns[loss_fn_name]
        else:
            raise Exception("Unknown loss function- {}".format(loss_fn_name))

    def get_model(self, config: KeypointRCNNConfiguration) -> BaseModel:
        model_type = config.training_params.model_type

        if model_type == "keypoint-rcnn" or model_type == "default":
            model = KeypointRCNN(num_keypoints = config.n_keypoints, num_classes = config.n_classes,
                                 pretrained_backbone=config.pretrained_backbone,
                                 trainable_backbone_layers=config.n_trainable_backbone_layers,
                                 all_layers_trainable=config.all_layers_trainable,
                                 anchor_generator_sizes=config.anchor_generator_sizes,
                                 anchor_generator_aspect_ratios=config.anchor_generator_aspect_ratios)
        else:
            raise Exception("Unknown model name ", model_type)

        if config.load_model_path is not None:
            model = load_model(model, config.load_model_path)

        return model

    def get_trainer(self) -> BaseTrainer:
        return KeypointRCNNTrainer(self.model, self.config, self.loss_fn, self.optimizer, self.scheduler)

    def get_dataset_viewer(self, dataset: BaseDataset) -> BaseDatasetViewer:
        return KeypointRCNNDatasetViewer(dataset, self.config)
