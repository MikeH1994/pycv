import sys
import os
import copy
import argparse
import torch
from torch.optim.lr_scheduler import LRScheduler, ExponentialLR
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Type, Dict, Tuple, List, Type, Union
from tkinter import Tk
from .base_model import BaseModel
from .base_dataset import BaseDataset
from .base_metrics import BaseLossFn
from .base_dataset_viewer import BaseDatasetViewer
from .base_evaluation_gui import BaseEvaluationGui
from .base_epoch_logger import BaseEpochLogger
from .base_trainer import BaseTrainer
from ..core.utils import calc_mean_std, load_model, setup_system
from ..core.configuration import Configuration, TrainingParams


class BaseProject:
    config: Configuration
    model: BaseModel
    loss_fn: BaseLossFn
    epoch_logger: BaseEpochLogger
    train_dataset: BaseDataset
    train_loader: DataLoader
    validation_dataset: BaseDataset
    validation_loader: DataLoader
    test_dataset: BaseDataset
    test_dataloader: DataLoader
    optimizer: Optimizer
    scheduler: LRScheduler

    def __init__(self, train_dataset: BaseDataset, val_dataset: BaseDataset,
                 config: Configuration, calc_mean_and_std = True, test_dataset: BaseDataset = None):
        self.train_dataset = train_dataset
        self.validation_dataset = val_dataset
        self.test_dataset = test_dataset
        self.config = self.parse_args(config)
        self.loss_fn = self.get_loss_fn(self.config)
        self.model = self.get_model(self.config)
        self.epoch_logger = self.get_epoch_logger(self.config)
        self.optimizer, self.scheduler = self.get_optimizer(self.model, self.config.training_params)

        if calc_mean_and_std:
            self.train_loader = self.get_loader(self.train_dataset, self.config, shuffle=True)
            mean, std = self.calc_mean_std()
            self.config.image_transform_params.mean = mean
            self.config.image_transform_params.std = std
        augment_data = self.config.training_params.augment_training_data

        self.train_dataset.initialise_transforms(self.config.image_transform_params, normalize=True,
                                                 augment_data=augment_data)
        self.train_loader = self.get_loader(self.train_dataset,self.config, shuffle=True)
        self.validation_dataset.initialise_transforms(self.config.image_transform_params, normalize=True,
                                                      augment_data=False)
        self.validation_loader = self.get_loader(self.validation_dataset, self.config, shuffle=False)
        if self.test_dataset is not None:
            self.test_dataset.initialise_transforms(self.config.image_transform_params, normalize=True,
                                                    augment_data=False)
            self.test_loader = self.get_loader(self.test_dataset, self.config, shuffle=False)
        else:
            self.test_loader = None
        config.train_dataset_summary = self.train_dataset.to_string()
        config.val_dataset_summary = self.validation_dataset.to_string()
        config.argv = ' '.join(sys.argv[1:])

    def calc_mean_std(self):
        return calc_mean_std(self.train_loader, show_loading_bar=True,
                                      description="Calculating mean and std of dataset...")

    def display_model(self):
        for name, module in self.model.named_modules():
            print(name, module)

    def parse_args(self, config: Configuration) -> Configuration:
        parser = argparse.ArgumentParser()
        parser.add_argument("--n_epochs", type=int, help="Number of epochs to train over")
        parser.add_argument("--data_root", type=str, default=None, help="Data root")
        parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
        parser.add_argument("--load_model", type=str, default=None, help="Load model")
        parser.add_argument("--model_type", type=str, default=None, help="Model used")
        parser.add_argument("--loss_fn", type=str, default=None, help="Loss function used")
        parser.add_argument("--init_lr", type=float, default=None, help="Initial learning rate. Default: 1E-3")
        parser.add_argument("--lr_decay", type=float, default=None, help="Learning rate decay. Default: 0.99")
        parser.add_argument("--scale_factor", type=float, default=None, help="Resize images")
        parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
        parser.add_argument("--clip_dataset", type=int, default=None, help="Batch size")
        parser.add_argument("--display_images", default=None, action="store_true", help="Display images each epoch")
        parser.add_argument("--always_clahe", default=None, action="store_true", help="Always apply CLAHE to images")
        parser.add_argument("--overwrite", action="store_true", help="Allow model overwriting")

        args = parser.parse_args()

        if args.n_epochs is not None:
            config.training_params.n_epochs = args.n_epochs
        if args.data_root is not None:
            config.root = args.data_root
        if args.output_dir is not None:
            config.output_dir = args.output_dir
        if args.load_model is not None:
            config.load_model_path = args.load_model
        if args.model_type is not None:
            config.training_params.model_type = args.model_type
        if args.loss_fn is not None:
            config.training_params.loss_fn = args.loss_fn
        if args.init_lr is not None:
            config.training_params.init_lr = args.init_lr
        if args.lr_decay is not None:
            config.training_params.lr_decay = args.lr_decay
        if args.scale_factor is not None:
            config.image_transform_params.image_width = int(args.scale_factor * config.image_transform_params.image_width)
            config.image_transform_params.image_height = int(args.scale_factor * config.image_transform_params.image_height)
        if args.batch_size is not None:
            config.batch_size = args.batch_size
        if args.clip_dataset is not None:
            config.clip_dataset = args.clip_dataset
        if args.display_images is not None:
            config.training_params.display_images = args.display_images
        if args.always_clahe is not None and args.always_clahe is True:
            config.image_transform_params.always_clahe = args.always_clahe

        if os.path.exists(os.path.join(config.output_dir, config.model_file_name)):
            if not args.overwrite:
                print("Data already exists in {}- pass --overwrite if you wish to continue"
                      .format(config.output_dir))
                sys.exit(1)
        return config

    def get_optimizer(self, model, training_params: TrainingParams) -> Tuple[Optimizer, LRScheduler]:
        optimizer = Adam(model.parameters(), lr=training_params.init_lr)
        scheduler = ExponentialLR(optimizer, gamma=training_params.lr_decay)
        return optimizer, scheduler

    def get_loader(self, dataset: BaseDataset, config: Configuration, shuffle: bool) -> DataLoader:
        return torch.utils.data.DataLoader(dataset, shuffle=shuffle, collate_fn=dataset.collate_fn,
                                           persistent_workers=True, batch_size=config.batch_size,
                                           num_workers=config.num_workers)

    def get_trainer(self) -> BaseTrainer:
        raise Exception("Base functio get_trainer() is not implemented")

    def get_epoch_logger(self, config: Configuration) -> BaseEpochLogger:
        raise Exception("Base function get_epoch_logger() is not implemented")

    def get_loss_fn(self, config: Configuration) -> BaseLossFn:
        raise Exception("Base function get_loss_fn(config) is not implemented")

    def get_model(self, config: Configuration) -> BaseModel:
        raise Exception("Base function get_model(config) is not implemented")

    def get_dataset_viewer(self, dataset: BaseDataset) -> BaseDatasetViewer:
        raise Exception("get_dataset_viewer is not implemented")

    def get_evaluation_gui(self, dataset: BaseDataset, model: BaseModel) -> BaseEvaluationGui:
        raise Exception("get_evaluation_gui is not implemented")

    def run_training(self, metrics_to_display=None, display_images = False):
        setup_system(self.config)
        self.model.to(self.config.device)

        tb_writer = SummaryWriter(log_dir=self.config.output_dir)
        trainer = self.get_trainer()
        trainer.run(self.train_loader, self.validation_loader, self.epoch_logger, tb_writer, metrics_to_display,
                    display_images=display_images)
        tb_writer.close()

    def run_evaluation(self, dataloader: Union[str, DataLoader], display_images=False, disable_progress_bar=False) -> BaseEpochLogger:
        setup_system(self.config)
        self.model.to(self.config.device)

        if isinstance(dataloader, str):
            dataloader = {
                "train": self.train_loader,
                "validation": self.validation_loader,
                "test": self.test_loader
            }[dataloader]
        assert(dataloader is not None)

        trainer = self.get_trainer()
        epoch_logger = self.get_epoch_logger(self.config)
        epoch_logger = trainer.validate(dataloader, epoch_logger, "Evaluating", display_images=display_images,
                                        disable_progress_bar=disable_progress_bar)
        return epoch_logger

    def run_dataset_viewer(self, mode: str):
        assert(mode == "train" or mode == "validation" or mode == "test")
        if mode == "train":
            dataset = self.train_dataset
        elif mode == "validation":
            dataset = self.validation_dataset
        else:
            dataset = self.test_dataset
        dataset_viewer = self.get_dataset_viewer(dataset)
        dataset_viewer.run()

    def run_evaluation_viewer(self, mode: str, load_model_fpath: str = None):
        assert(mode == "train" or mode == "validation" or mode == "test")
        if mode == "train":
            dataset = self.train_dataset
        elif mode == "validation":
            dataset = self.validation_dataset
        else:
            dataset = self.test_dataset
        if load_model_fpath is None:
            model = self.model
        else:
            model = self.get_model(self.config)
            model = load_model(model, load_model_fpath)

        evaluation_gui = self.get_evaluation_gui(dataset, model)
        evaluation_gui.run()
