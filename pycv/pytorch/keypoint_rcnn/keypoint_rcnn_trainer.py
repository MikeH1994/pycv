from ..base.base_trainer import BaseTrainer
from ..core.configuration import KeypointRCNNConfiguration
from .models import KeypointRCNN
from typing import Dict
import torch
from torch import Tensor


class KeypointRCNNTrainer(BaseTrainer):
    model: KeypointRCNN
    config: KeypointRCNNConfiguration
    def next_batch(self, sample: Dict, mode: str, epoch_metrics, display_images=False):
        assert(mode == "train" or mode == "eval")
        images = sample["inputs"].float().to(self.config.device)
        targets = [{key: val.to(self.config.device) for key, val in sample["targets"][i].items()} for i in range(len(sample["targets"]))]
        if mode == "train":
            preds = self.model(images, targets)
        else:
            with torch.no_grad():
                self.model.train()
                preds = self.model(images, targets)
                self.model.eval()
        loss = self.loss_fn(preds, targets)

        epoch_metrics.update(preds, targets, loss)

        if len(epoch_metrics.images) < 16:
            if mode == "train":
                with torch.no_grad():
                    self.model.eval()
                    preds = self.model(images)
                    self.model.train()
            else:
                with torch.no_grad():
                    self.model.eval()
                    preds = self.model(images)
            epoch_metrics.add_images(images, self.model.process_predictions(preds,
                                                                            score_boundary=self.config.score_boundary,
                                                                            nms_boundary=self.config.nms_boundary),self.config)
        return loss
