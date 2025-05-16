from ..base.base_trainer import BaseTrainer
from ..base.base_epoch_logger import BaseEpochLogger
from ..core.configuration import KeypointConfiguration
from .models import KeypointModel
from typing import Dict
import torch
from torch import Tensor


class KeypointTrainer(BaseTrainer):
    model: KeypointModel
    config: KeypointConfiguration

    def next_batch(self, sample: Dict, mode: str, epoch_logger: BaseEpochLogger, display_images=False):
        assert(mode == "train" or mode == "eval")
        images = sample["inputs"].float().to(self.config.device)
        targets = sample["targets"].float().to(self.config.device)

        preds = self.model(images)
        loss = self.loss_fn(preds, targets)

        epoch_logger.update(preds, targets, loss)

        if len(epoch_logger.images) < 16:
            epoch_logger.add_images(images, self.model.process_predictions(preds), self.config)
        return loss
