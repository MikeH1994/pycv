import os
import cv2
import matplotlib.pyplot as plt
from pycv.metrics.targets.slantededge.core import get_edge_points
from pycv.utils.drawing import overlay_points_on_image
import numpy as np


def run_get_edge_points():
    folderpath = "../../images/slanted_edge"
    for i in range(1, 63):
        for transpose in [True, False]:
            image_fpath = os.path.join(folderpath, "image_{}_gray.png".format(i))
            img = cv2.imread(image_fpath, -1)

            if transpose:
                img = np.transpose(img)

            x, y = get_edge_points(img)[0]
            print(x, y)
            plt.imshow(img)
            plt.scatter(x[::4], y[::4], color='r')
            plt.show()


if __name__ == "__main__":
    run_get_edge_points()
