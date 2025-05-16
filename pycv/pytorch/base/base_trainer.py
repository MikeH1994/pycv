from __future__ import annotations
from tqdm.auto import tqdm
import os
import torch
import torch.utils.data
import numpy as np
import copy
import matplotlib.pyplot as plt
from typing import List, Dict
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter
from ..base.base_model import BaseModel
from ..base.base_epoch_logger import BaseEpochLogger
from ..base.base_metrics import BaseLossFn
from ..core.utils import save_model, save_config, setup_system
from ..core import Configuration


class BaseTrainer:
    def __init__(self, model: BaseModel, config: Configuration, loss_fn: BaseLossFn, optimizer: Optimizer,
                 lr_scheduler: LRScheduler):
        """

        :param model:
        :param config:
        :param loss_fn:
        :param optimizer:
        :param lr_scheduler:
        """
        self.model = model
        self.config = config
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = lr_scheduler

    def train(self, train_loader: DataLoader, epoch_metrics: BaseEpochLogger, description: str = "",
              disable_progress_bar: bool = False, display_images=False):
        """

        :param train_loader:
        :param epoch_metrics:
        :param description:
        :param disable_progress_bar:
        :return:
        """
        epoch_metrics = copy.deepcopy(epoch_metrics)
        epoch_metrics.reset()
        progress_bar = tqdm(train_loader, disable=disable_progress_bar, dynamic_ncols=True, unit="batch")
        for batch_idx, sample in enumerate(progress_bar):
            self.model.train()
            self.optimizer.zero_grad()
            if sample["batch_size"] <= 1:
                continue
            loss = self.next_batch(sample, "train", epoch_metrics, display_images=display_images)
            loss.backward()
            self.optimizer.step()

            status = "[Train ({})][{}] Loss: {:.3E},  LR: {:.3E}".format(
                self.config.device,
                description, epoch_metrics["loss"],
                self.optimizer.param_groups[0]["lr"])
            progress_bar.set_description(status)
        return epoch_metrics

    def validate(self, val_loader: DataLoader, epoch_metrics: BaseEpochLogger,
                 description="", disable_progress_bar=False, display_images=False):
        """

        :param val_loader:
        :param epoch_metrics:
        :param description:
        :param disable_progress_bar:
        :return:
        """
        epoch_metrics = copy.deepcopy(epoch_metrics)
        epoch_metrics.reset()

        with torch.no_grad():
            progress_bar = tqdm(val_loader, disable=disable_progress_bar, dynamic_ncols=True, unit="batch")
            for batch_idx, sample in enumerate(progress_bar):
                self.model.eval()
                if sample["batch_size"] <= 1:
                    continue
                self.next_batch(sample, "eval", epoch_metrics, display_images=display_images)

                status = "[Val   ({})][{}] Loss: {:.3E}".format(self.config.device,
                                                                description,
                                                                epoch_metrics["loss"])
                progress_bar.set_description(status)
        return epoch_metrics

    def run(self, train_loader: DataLoader, val_loader: DataLoader, epoch_metrics: BaseEpochLogger,
            tb_writer: SummaryWriter = None, metrics_to_display: List[str] = None, display_images=False):
        """

        :param train_loader:
        :param val_loader:
        :param epoch_metrics:
        :param tb_writer:
        :param metrics_to_display:
        :return:
        """
        setup_system(self.config)
        self.model.to(self.config.device)
        save_config(self.config, os.path.join(self.config.output_dir, "config.pck"))
        if tb_writer is not None:
            tb_writer.add_text('misc/config', str(self.config))
            tb_writer.flush()

        loss_type = type(self.loss_fn).__name__
        init_val_metrics = self.validate(val_loader, epoch_metrics, description="Initial")
        print("    Initial resuls: Loss ({}): {:.3E}".format(loss_type, init_val_metrics["loss"]), flush=True)
        best_val_loss = init_val_metrics["loss"]
        best_train_loss = np.inf

        start_epoch = self.config.training_params.start_epoch
        end_epoch = self.config.training_params.start_epoch + self.config.training_params.n_epochs
        for epoch in range(start_epoch, end_epoch):
            description = "Epoch {:03d}/{:03d}".format(epoch + 1, end_epoch)
            train_metrics = self.train(train_loader, epoch_metrics, description=description)
            val_metrics = self.validate(val_loader, epoch_metrics, description=description)
            print("    Training: {}".format(train_metrics.to_string(metrics_to_display)), flush=True)
            print("    Validation: {}".format(val_metrics.to_string(metrics_to_display)), flush=True)

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                print('    Model Improved. Saving the Model...', flush=True)
                model_fpath = os.path.join(self.config.output_dir,
                                           self.config.model_file_name)
                save_model(self.model, device=self.config.device, model_fpath=model_fpath)

            if train_metrics["loss"] < best_train_loss:
                best_train_loss = train_metrics["loss"]

            model_fpath = os.path.join(self.config.output_dir,
                                       self.config.model_file_name.replace('.pt', '_checkpoint.pt'))
            save_model(self.model, device=self.config.device, model_fpath=model_fpath)

            if tb_writer is not None:
                train_metrics.write_to_tensorboard(tb_writer, "Training", epoch, write_images=True)
                val_metrics.write_to_tensorboard(tb_writer, "Validation", epoch, write_images=True)
                tb_writer.add_scalar("Validation/Best Loss", best_val_loss, epoch)
                tb_writer.add_scalar("Training/Best Loss", best_train_loss, epoch)
                tb_writer.add_scalar("Misc/Learning rate", self.optimizer.param_groups[0]["lr"], epoch)
                tb_writer.flush()

            if self.scheduler is not None:
                self.scheduler.step()

            if display_images:
                for i, img in enumerate(train_metrics.images[:8]):
                    plt.figure()
                    plt.title("Training- image {}".format(i))
                    plt.imshow(img)
                plt.show()

                for i, img in enumerate(val_metrics.images[:8]):
                    plt.figure()
                    plt.title("Validation- image {}".format(i))
                    plt.imshow(img)
                plt.show()
        if tb_writer is not None:
            tb_writer.close()

    def next_batch(self, sample: Dict, mode: str, epoch_logger: BaseEpochLogger, display_images=False):
        images = sample["inputs"].float().to(self.config.device)
        targets = sample["targets"].float().to(self.config.device)
        preds = self.model(images)
        loss = self.loss_fn(preds, targets)

        preds = self.model(images)
        epoch_logger.update(preds.detach().to("cpu"), targets.detach().to("cpu"), loss.detach().to("cpu"))
        if len(epoch_logger.images) < 8:
            epoch_logger.add_images(images, self.model.process_predictions(preds), self.config)

        if display_images:
            images = epoch_logger.batch_to_images(images, self.model.process_predictions(preds), self.config)
            for image in images:
                plt.imshow(image)
                plt.show()
        return loss

