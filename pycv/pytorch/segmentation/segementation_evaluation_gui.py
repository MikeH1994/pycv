import numpy as np
import mytorch.core as core
import cv2
import matplotlib.pyplot as plt
import torch
from .utils import overlay_predictions_on_image
from ..base import BaseEvaluationGui, BaseDataset, BaseModel
from ..core import Configuration


class SegmentationEvaluationGui(BaseEvaluationGui):
    overlay_mask = True
    mask = None

    def __init__(self, dataset: BaseDataset, model: BaseModel,
                 config: Configuration, name: str = "[default]"):
        super().__init__(dataset, model, config, name)
        self.mask = np.zeros((config.image_transform_params.image_height,
                              config.image_transform_params.image_width, 3), dtype=np.uint8)

    def toggle(self):
        self.overlay_mask = not self.overlay_mask

    def load_image(self, index):
        if self.dataset is not None:
            if index >= len(self.dataset) or index < 0:
                raise Exception("Index out of range")

            samples = self.dataset[index]
            self.img = core.tensor_img_to_numpy(samples["input"], scale_factor=self.scale_factor, offset=self.offset,
                                                convert_to_8_bit=True)
            with torch.no_grad():
                pred = self.model(torch.stack([samples["input"]]).to(self.config.device))[0].to('cpu')
                self.mask = self.model.process_predictions(pred).numpy().astype(np.uint8)

    def process_image(self, index):
        if self.overlay_mask:
            img = overlay_predictions_on_image(self.img, self.mask)
        else:
            img = np.copy(self.img)

        text = "{}/{}".format(index, len(self.dataset))
        coordinates = (0, img.shape[0] - 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 0, 255)
        fontScale = 1
        thickness = 1
        img = cv2.putText(img, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
