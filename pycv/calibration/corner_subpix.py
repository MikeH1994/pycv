import numpy as np
from typing import Tuple

def get_rect_subpix():
    pass

def corner_subpix(image: np.ndarray, corners: np.ndarray, win_size: Tuple[int, int], zero_zone, criteria):
    #https://github.com/opencv/opencv/blob/e9bded6ff3e54f68d25f16e2e51ce1a61777c09a/modules/imgproc/src/cornersubpix.cpp#L44
    win_height, win_width = win_size

    mask = np.zeros()#

