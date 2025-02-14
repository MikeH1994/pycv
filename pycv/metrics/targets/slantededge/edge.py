import numpy as np
from numpy.typing import NDArray
from typing import Union
from pycv.metrics.targets.slantededge.utils import get_derivative_filters, deriv1, centroid, findedge2, edge_is_vertical, get_window, rotate_image
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pycv.metrics.targets.slantededge.utils import get_edge_points_from_esf_fit,  get_edge_points_from_lsf_fit,\
    get_edge_points_from_centroid


class Edge:
    def __init__(self, img: NDArray, edge_x: NDArray, edge_y: NDArray):
        self.img = np.copy(img)
        self.edge_x = np.copy(edge_x)
        self.edge_y = np.copy(edge_y)
        self.poly = np.polynomial.polynomial.polyfit(self.edge_x, self.edge_y, 1)
        self.angle = np.degrees(np.arctan(self.poly[1]))
        self.m = self.poly[1]
        self.is_inf = False

    def plot(self, title="", show=True, new_figure=True, linestyle="-", marker=""):
        if new_figure:
            plt.figure()
        plt.title(title)
        plt.imshow(self.img)
        plt.plot(self.edge_x, self.edge_y, linestyle=linestyle, marker=marker)
        if show:
            plt.show()

    def get_distance_to_edge(self, x: Union[float, NDArray], y: Union[float, NDArray], mode="normal"):
        if mode == "normal":
            return self.distance_normal(x, y)
        raise Exception("Have not implemented other modes yet")

    def distance_normal(self, x: Union[float, NDArray], y: Union[float, NDArray]):
        if not self.is_inf:
            edge_y = np.polynomial.polynomial.polyval(x, self.poly)
            return (y - edge_y) / np.sqrt(self.m ** 2 + 1)
        else:
            raise Exception("Not implemented yet")