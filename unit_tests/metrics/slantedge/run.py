import os
import cv2
import matplotlib.pyplot as plt
import pycv.metrics.targets.slantededge.core
from pycv.metrics.targets.slantededge.core import get_edge_profile_from_image
import pycv.metrics.esf as pyesf
import numpy as np


def run_get_edge_points():
    folderpath = "../../images/slanted_edge"
    for i in range(2, 63)[15:]:
        for transpose in [True, False]:
            image_fpath = os.path.join(folderpath, "image_{}_gray.png".format(i))
            img = cv2.imread(image_fpath, -1)

            if transpose:
                img = np.transpose(img)

            x_fit, y_fit = pycv.metrics.targets.slantededge.core.get_edge_points(img, edge_detection_mode="fit")[0]
            x_cent, y_cent = pycv.metrics.targets.slantededge.core.get_edge_points(img, edge_detection_mode="centroid")[0]

            plt.imshow(img)
            plt.scatter(x_fit[::4], y_fit[::4], color='r')
            plt.scatter(x_cent[::4], y_cent[::4], color='b')
            plt.figure()

            esf, _ = pyesf.create_binned_esf(img, edge_detection_mode="fit")
            lsf = esf.lsf()
            mtf = lsf.mtf()

            esf.plot()
            plt.figure()
            lsf.plot()
            plt.figure()
            mtf.plot()
            plt.show()


if __name__ == "__main__":
    run_get_edge_points()
