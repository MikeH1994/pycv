import numpy as np
import cv2
from typing import Tuple, Dict, Callable
from mytorch.base import BaseDataset
from mytorch.core import Configuration


class BaseDViewer:
    def __init__(self, dataset: BaseDataset, config: Configuration, name: str = "[default]",
                 dst_size: Tuple[int, int] = None, button_callbacks: Dict[str, Callable] = None):
        self.current_index = 0
        self.running = True
        self.change_index = True
        self.dataset = dataset
        self.name = name
        self.offset = config.image_transform_params.mean
        self.scale_factor = config.image_transform_params.std
        self.img = np.zeros((config.image_transform_params.image_height,
                             config.image_transform_params.image_width, 3), dtype=np.uint8)
        self.dst_size = dst_size
        self.len = 0
        self.button_callbacks = button_callbacks if button_callbacks is not None else {}
        if dataset is not None:
            self.len = len(dataset)

    def run(self):
        self.image_thread()

    def load_image(self, index):
        return np.zeros((*self.dst_size, 3))

    def process_image(self, image, current_index):
        return image

    def handle_keypress(self, keypress: int):
        pass

    def image_thread(self):
        image = np.zeros((*self.dst_size, 3))

        while self.running:
            if self.change_index:
                image = self.load_image(self.current_index)
                image = self.process_image(image, self.current_index)
                if self.dst_size is not None:
                    image = cv2.resize(image, self.dst_size)
                self.change_index = False

            title = '{}'.format(self.name)

            cv2.imshow(title, image)
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, image.shape[1], image.shape[0])
            key_pressed = cv2.waitKeyEx(0)
            print(key_pressed)

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

