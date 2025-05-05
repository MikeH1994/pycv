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


def draw_text(img, text, font=cv2.FONT_HERSHEY_PLAIN, pos=(0, 0), font_scale=3, font_thickness=2,
              text_color=(255, 255, 255), text_color_bg=(0, 0, 0), center=True, boundary=6, halign="left", valign="top"):
    assert(halign == "left" or halign == "right" or halign == "center")
    assert(valign == "top" or halign == "bottom" or halign == "center")

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size

    y += boundary
    x += boundary

    if halign=="center":
        x -= text_w//2
    elif halign=="right":
        x -= text_w
    if valign == "center":
        y -= text_h//2
    elif valign == "bottom":
        y -= text_h

    cv2.rectangle(img, (x-boundary,y-boundary), (x + text_w+boundary, y + text_h+boundary), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)
    return img, text_w + 2* boundary, text_h + 2*boundary