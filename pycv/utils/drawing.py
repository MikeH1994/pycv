from typing import Tuple
from typing import Union, List
import cv2
import numpy as np
from numpy.typing import NDArray
from .colour import get_colour

def overlay_points_on_image(img: NDArray, points: List[Tuple], radius=1, color=(255, 0, 0), thickness = -1):
    assert(len(img.shape) == 3 and img.shape[2] == 3), "Image suppled should be rgb- shape: {}".format(img.shape)
    image = np.copy(img)
    image = np.ascontiguousarray(image)  # I don't know why but cv2.rectangle fails otherwise

    for p in points:
        cv2.circle(image, (int(p[0]), int(p[1])), radius, color, thickness)
    return image