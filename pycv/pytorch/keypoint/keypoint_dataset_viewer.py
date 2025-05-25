from ..base.base_dataset_viewer import BaseDatasetViewer
from ..utils import overlay_points_on_image
import pycv.pytorch.core as core
import numpy as np
import cv2
import matplotlib.pyplot as plt


class KeypointDatasetViewer(BaseDatasetViewer):
    overlay_points = True
    points = None
    bboxes = None
    labels = None

    def toggle(self):
        self.overlay_points = not self.overlay_points

    def load_image(self, index):
        if self.dataset is not None:
            if index >= len(self.dataset) or index < 0:
                raise Exception("Index out of range")

            samples = self.dataset[index]
            self.img = core.tensor_img_to_numpy(samples["input"], scale_factor=self.scale_factor, offset=self.offset,
                                                convert_to_8_bit=True)
            self.points = samples["target"].numpy().astype(np.float32).reshape(-1, 2)
            print(samples["image_filepath"])


    def process_image(self, index):
        if self.overlay_points:
            img = overlay_points_on_image(self.img, keypoints_list=self.points)
        else:
            img = np.copy(self.img)

        text = "{}/{}".format(index + 1, len(self.dataset))
        coordinates = (0, img.shape[0] - 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 0, 255)
        fontScale = 1
        thickness = 1
        img = cv2.putText(img, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
