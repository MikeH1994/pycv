from typing import Type, Tuple
from torch.utils.data import DataLoader
from ..base import BaseProject, BaseLossFn, BaseModel, BaseEpochLogger, BaseDataset, BaseDatasetViewer, BaseTrainer
from ..base import BaseEvaluationGui
from ..core import SegmentationConfiguration, load_model, Configuration
from .dataset import SegmentationDataset
from .metrics import CrossEntropyLoss, DiceLoss, JaccardLoss, FocalLoss, MultiObjectiveLoss, IOU
from .models import DeepLab, UNet, FCN
from .segmentation_logger import SegmentationEpochLogger
from .segmentation_dataset_viewer import SegmentationDatasetViewer
from .segementation_evaluation_gui import SegmentationEvaluationGui


class SegmentationProject(BaseProject):
    def __init__(self, train_dataset: BaseDataset, val_dataset: BaseDataset,
                 config: Configuration, calc_mean_and_std = True, test_dataset: BaseDataset = None):
        super().__init__(train_dataset, val_dataset, config, calc_mean_and_std=calc_mean_and_std,
                         test_dataset=test_dataset)

    def get_epoch_logger(self, config: SegmentationConfiguration) -> BaseEpochLogger:
        additional_metrics = {
            "IOU": IOU()
        }
        loss_fn = self.get_loss_fn(config)
        return SegmentationEpochLogger(loss_fn, config.n_classes, additional_metrics)

    def get_loss_fn(self, config: SegmentationConfiguration) -> BaseLossFn:
        binary = True if config.n_classes == 1 else False
        loss_fn_name = config.training_params.loss_fn

        loss_fns = {
            "default": FocalLoss(binary=binary),
            "cross-entropy": CrossEntropyLoss(binary=binary),
            "dice": DiceLoss(soft_loss=False),
            "soft-dice": DiceLoss(soft_loss=True),
            "jaccard": JaccardLoss(soft_loss=False),
            "soft-jaccard": JaccardLoss(soft_loss=False),
            "focal": FocalLoss(binary=binary),
            "multi-objective": MultiObjectiveLoss(binary=binary)
        }
        if loss_fn_name in loss_fns:
            return loss_fns[loss_fn_name]
        else:
            raise Exception("Unknown loss function- {}".format(loss_fn_name))

    def get_model(self, config: SegmentationConfiguration) -> BaseModel:
        n_classes = config.n_classes
        model_type = config.training_params.model_type
        if model_type == "deeplab" or model_type == "default":
            model = DeepLab(n_classes)
        elif model_type == "unet":
            model = UNet(n_classes)
        elif model_type == "fcn":
            model = FCN(n_classes)
        else:
            raise Exception("Unknown model name ", model_type)
        if config.load_model_path is not None:
            model = load_model(model, config.load_model_path)
        return model

    def get_dataset_viewer(self, dataset: BaseDataset) -> BaseDatasetViewer:
        return SegmentationDatasetViewer(dataset, self.config)

    def get_evaluation_gui(self, dataset: BaseDataset, model: BaseModel) -> BaseEvaluationGui:
        return SegmentationEvaluationGui(dataset, model, self.config)

    def get_trainer(self) -> BaseTrainer:
        return BaseTrainer(self.model, self.config, self.loss_fn, self.optimizer, self.scheduler)
