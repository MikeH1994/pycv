from typing import Type, Tuple
from torch.utils.data import DataLoader
from ..base import BaseProject, BaseLossFn, BaseScoreFn, BaseModel, BaseEpochLogger, BaseDataset, BaseTrainer, BaseDatasetViewer
from ..core import KeypointRCNNConfiguration, load_model, KeypointConfiguration
from .metrics.keypoint_metrics import L1Score, L2Score, SmoothL1Score
from .metrics.keypoint_loss import L1Loss, SmoothL1Loss, L2Loss
from .models import ResNetKeypointModel
from .keypoint_dataset_viewer import KeypointDatasetViewer
from .keypoint_logger import KeypointLogger


class KeypointProject(BaseProject):
    def __init__(self, train_dataset: BaseDataset, val_dataset: BaseDataset,
                 config: KeypointConfiguration, calc_mean_and_std = True, test_dataset: BaseDataset = None):
        super().__init__(train_dataset, val_dataset, config, calc_mean_and_std=calc_mean_and_std,
                         test_dataset=test_dataset)

    def get_epoch_logger(self, config: KeypointConfiguration) -> BaseEpochLogger:
        additional_metrics = {
            "L1": L1Score(),
            "smooth_L1": SmoothL1Score(),
            "L2": L2Score(),
        }
        loss_fn = self.get_loss_fn(config)
        return KeypointLogger(loss_fn, config.n_keypoints, additional_metrics, config.keypoint_names) #

    def get_loss_fn(self, config: KeypointConfiguration) -> BaseLossFn:
        loss_fn_name = config.training_params.loss_fn
        loss_fns = {
            "L1": L1Loss(),
            "smooth_L1": SmoothL1Loss(),
            "L2": L2Loss(),
            "default": L1Loss(),
        }
        if loss_fn_name in loss_fns:
            return loss_fns[loss_fn_name]
        else:
            raise Exception("Unknown loss function- {}".format(loss_fn_name))

    def get_model(self, config: KeypointConfiguration) -> BaseModel:
        model_type = config.training_params.model_type

        if model_type == "resnet" or model_type == "default":
            model = ResNetKeypointModel(config.n_keypoints, n_dense_layers=config.n_dense_layers)
        else:
            raise Exception("Unknown model name ", model_type)

        if config.load_model_path is not None:
            model = load_model(model, config.load_model_path)

        return model

    def get_trainer(self) -> BaseTrainer:
        return BaseTrainer(self.model, self.config, self.loss_fn, self.optimizer, self.scheduler)

    def get_dataset_viewer(self, dataset: BaseDataset) -> BaseDatasetViewer:
        return KeypointDatasetViewer(dataset, self.config)
