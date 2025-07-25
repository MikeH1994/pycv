import numpy as np
import matplotlib.pyplot as plt
import cv2

import pycv.core
from pycv.pinholecamera import PinholeCamera


def visualise_distortion(camera: PinholeCamera, new_fig=True, color='black', label="calibration"):
    xres = camera.xres
    yres = camera.yres
    label_set = False
    if new_fig:
        plt.figure()
        blank_img = np.full((yres, xres, 3), fill_value=255, dtype=np.uint8)
        plt.imshow(blank_img)
    for x in np.linspace(0, xres-1, 20):
        xarr = np.full(1000, fill_value=x)
        yarr = np.linspace(-100, yres+100, 1000)

        xdistorted, ydistorted = camera.distort_points(pycv.core.stack_coords((xarr, yarr)))
        if not label_set:
            plt.plot(xdistorted, ydistorted, color=color, label=label)
            label_set=True
        else:
            plt.plot(xdistorted, ydistorted, color=color)


    for y in np.linspace(0, xres-1, 20):
        yarr = np.full(1000, fill_value=y)
        xarr = np.linspace(-100, xres+100, 1000)

        xdistorted, ydistorted = camera.distort_points(pycv.core.stack_coords((xarr, yarr)))
        plt.plot(xdistorted, ydistorted, color=color)

    plt.xlim([0, xres-1])
    plt.ylim([yres-1, 0])