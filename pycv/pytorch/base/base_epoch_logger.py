from __future__ import annotations
import numpy as np
import torch
from typing import Dict, Union, List
from ..core.average_meter import AverageMeter, MultiClassAverageMeter
from ..core.configuration import Configuration
from .base_metrics import BaseLossFn, BaseScoreFn


class BaseEpochLogger:
    def __init__(self, loss_fn: BaseLossFn, num_classes,
                 additional_metrics: Dict[str, Union[BaseScoreFn, BaseScoreFn]] = None,
                 class_names: List[str] = None):
        self.num_classes = num_classes
        self.metric_fns: Dict[str, Union[BaseLossFn, BaseScoreFn]] = {
            "loss": loss_fn,
            **additional_metrics
        }
        self.metrics: Dict[str, Union[AverageMeter, MultiClassAverageMeter]] = {
            "loss": AverageMeter(),
        }
        if additional_metrics is not None:
            for metric_name in additional_metrics:
                metric = additional_metrics[metric_name]
                if isinstance(metric, BaseLossFn):
                    self.metrics[metric_name] = AverageMeter()
                elif isinstance(metric, BaseScoreFn):
                    self.metrics[metric_name] = MultiClassAverageMeter(num_classes)
                else:
                    raise Exception("Logic error")

        if class_names is not None:
            assert(len(class_names) == num_classes)
            self.class_names = ["({}) {}".format(i, class_names[i]) for i in range(len(class_names))]
        else:
            self.class_names = ["class {}".format(i) for i in range(num_classes)]
        self.images = []

    def update(self, preds: torch.FloatTensor, targets: torch.FloatTensor, loss: torch.FloatTensor):
        batch_size = len(preds)
        for metric_name in self.metric_fns:
            metric_fn = self.metric_fns[metric_name]
            if isinstance(metric_fn, BaseLossFn):
                loss = metric_fn(preds, targets).item()
                self.metrics[metric_name].update(loss, batch_size)
            elif isinstance(metric_fn, BaseScoreFn):
                _, score_per_class = metric_fn(preds, targets)
                self.metrics[metric_name].update(score_per_class, batch_size)
            else:
                raise Exception("Logic error")

        return loss

    def add_images(self, imgs: torch.FloatTensor, preds: torch.FloatTensor, config: Configuration):
        images = self.batch_to_images(imgs, preds, config)
        self.images += images

    def batch_to_images(self, imgs: torch.FloatTensor, preds: torch.FloatTensor, config: Configuration):
        return []

    def reset(self):
        """

        :return:
        """
        for key in self.metrics:
            metric = self.metrics[key]
            if isinstance(metric, AverageMeter):
                metric.reset()
            elif isinstance(metric, list):
                for metric_elem in metric:
                    metric_elem.reset()
        self.images = []

    def write_to_tensorboard(self, tb_writer, prefix, epoch, write_images=False):
        """

        :param tb_writer:
        :param prefix:
        :param epoch:
        :param write_images:
        :return:
        """
        for metric_name in self.metrics:
            metric = self.metrics[metric_name]
            if isinstance(metric, AverageMeter):
                tb_writer.add_scalar('{}/{}'.format(prefix, metric_name), metric.avg, epoch)
            elif isinstance(metric, MultiClassAverageMeter):
                for i in range(self.num_classes):
                    name = '{}/{}/{}'.format(metric_name, prefix, self.class_names[i])
                    tb_writer.add_scalar(name, metric.avg[i], epoch)
        if write_images:
            for i, image in enumerate(self.images):
                tb_writer.add_image('{}/Image {}'.format(prefix, i), image.transpose(2, 0, 1), epoch)
        tb_writer.flush()

    def to_string(self, metrics_to_display: List[str] = None):
        """

        :param metrics_to_display:
        :return:
        """
        if metrics_to_display is None:
            metrics_to_display = ["loss"]
        msg = ""
        for metric_name in metrics_to_display:
            metric = self.metrics[metric_name]
            if isinstance(metric, AverageMeter):
                val = metric.avg
                msg += "{}: {:.5E}; ".format(metric_name, val)
            elif isinstance(metric, MultiClassAverageMeter):
                val = np.array(metric.avg).mean()
                msg += "Mean {}: {:.5E}; ".format(metric_name, val)
            else:
                raise Exception("Logic error")
        return msg

    def __getitem__(self, metric_name: str) -> Union[float, List[float]]:
        """

        :param metric_name:
        :return:
        """
        if metric_name not in self.metrics:
            raise Exception("Requested metric {} not found".format(metric_name))
        return self.metrics[metric_name].avg
