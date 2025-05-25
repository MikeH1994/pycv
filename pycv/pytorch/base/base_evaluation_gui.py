import numpy as np
import cv2
from .base_model import BaseModel
from .base_dataset import BaseDataset
from pycv.pytorch.core import Configuration
from typing import Tuple


class BaseEvaluationGui:
    def __init__(self, dataset: BaseDataset, model: BaseModel,
                 config: Configuration, name: str = "[default]"):
        self.dataset = dataset
        self.config = config
        self.model = model
        if model is not None:
            self.model.eval()
            self.model.to(self.config.device)
        self.running = True
        self.current_index = 0
        self.change_index = True
        self.len = 0
        self.name = name
        self.offset = config.image_transform_params.mean
        self.scale_factor = config.image_transform_params.std
        self.img = np.zeros((config.image_transform_params.image_height,
                             config.image_transform_params.image_width, 3), dtype=np.uint8)
        if dataset is not None:
            self.len = len(dataset)

    def run(self):
        self.image_thread()

    def toggle(self):
        pass

    def load_image(self, index):
        return np.zeros((self.config.image_transform_params.image_height,
                         self.config.image_transform_params.image_width, 3), dtype=np.uint8)

    def process_image(self, current_index):
        return np.copy(self.img)

    def image_thread(self):
        image = np.zeros((self.config.image_transform_params.image_height,
                         self.config.image_transform_params.image_width, 3), dtype=np.uint8)

        while self.running:
            if self.change_index:
                self.load_image(self.current_index)
                image = self.process_image(self.current_index)
                self.change_index = False

            title = '{}'.format(self.name)

            cv2.imshow(title, image)
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, image.shape[1], image.shape[0])
            key_pressed = cv2.waitKeyEx(0)

            if key_pressed == 119:  # w
                self.toggle()
                image = self.process_image(self.current_index)

            if key_pressed == 114:  # r
                image = self.process_image(self.current_index)
                self.change_index = True

            if key_pressed == 113 or key_pressed == 101:  # q or e
                change_index = True

                if key_pressed == 113:
                    self.current_index -= 1

                if key_pressed == 101:
                    self.current_index += 1

                if self.current_index >= self.len:
                    self.current_index = self.len - 1
                    change_index = False

                if self.current_index < 0:
                    self.current_index = 0
                    change_index = False

                if change_index:
                    self.change_index = True
