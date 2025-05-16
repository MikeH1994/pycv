import matplotlib.pyplot as plt
import cv2
import numpy as np
from ..base import BaseDataset
from ..core import tensor_img_to_numpy
from ..segmentation import overlay_predictions_on_image


def test_display_images(dataset, overlay_images: bool = True, min_index=0, max_index=None):
    if max_index is None:
        max_index = len(dataset)
    for i in range(min_index, min(len(dataset), max_index)):
        img_raw = dataset.get_raw_image(i)
        plt.title("Raw")
        plt.imshow(img_raw)

        for j in range(3):
            samples = dataset[i]
            img_tensor = samples["input"]
            mask_npy = samples["target"].numpy().astype(np.uint8)

            img_npy = tensor_img_to_numpy(img_tensor, convert_to_8_bit=True,
                                          scale_factor=dataset.config.image_transform_params.std,
                                          offset=dataset.config.image_transform_params.mean)
            if overlay_images:
                img_npy = overlay_predictions_on_image(img_npy, mask_npy)
            plt.figure()
            plt.title("Transformed {}".format(j))
            plt.imshow(img_npy)
        plt.show()
