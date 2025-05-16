import numpy as np
import mytorch.core as core
import cv2
import matplotlib.pyplot as plt
import torch
from .utils import overlay_points_on_image
from ..base import BaseEvaluationGui, BaseDataset, BaseModel
from ..core import Configuration, KeypointRCNNConfiguration
from .models import KeypointRCNN


class KeypointRCNNEvaluationGui(BaseEvaluationGui):
    model: KeypointRCNN
    overlay_bboxes = True
    nms_boundary = 0.3
    value_boundary = 0.7
    config: KeypointRCNNConfiguration
    preds = None

    def __init__(self, dataset: BaseDataset, model: BaseModel,
                 config: Configuration, name: str = "[default]"):
        super().__init__(dataset, model, config, name)

    def toggle(self):
        self.overlay_bboxes = not self.overlay_bboxes

    def load_image(self, index):
        if self.dataset is not None:
            if index >= len(self.dataset) or index < 0:
                raise Exception("Index out of range")

            sample = self.dataset[index]
            self.img = core.tensor_img_to_numpy(sample["input"], scale_factor=self.scale_factor, offset=self.offset,
                                                convert_to_8_bit=True)
            with torch.no_grad():
                self.model.eval()
                self.preds = self.model(torch.stack([sample["input"]]).to(self.config.device))[0].to('cpu')

    def process_image(self, index):
        if self.overlay_bboxes:
            preds = self.model.process_predictions(self.preds, score_boundary=self.config.score_boundary,
                                                   nms_boundary=self.config.nms_boundary)

            img = overlay_points_on_image(self.img, preds)
        else:
            img = np.copy(self.img)

        text = "{}/{} nms = {:.2f} sb = {:.2f}".format(index, len(self.dataset), self.config.nms_boundary,
                                                       self.config.score_boundary)
        coordinates = (0, img.shape[0] - 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 0, 255)
        fontScale = 1
        thickness = 1
        img = cv2.putText(img, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
