import numpy as np
from numpy.typing import NDArray
from typing import Union
import matplotlib.pyplot as plt
from .core import get_edge_points


class Edge:
    def __init__(self, img: NDArray):
        self.img = np.copy(img)

        self.edge_x_loc, self.edge_y_loc = get_edge_points(img)
        self.poly = np.polynomial.polynomial.polyfit(self.edge_x_loc, self.edge_y_loc, 1)
        self.angle = np.degrees(np.arctan(self.poly[1]))
        self.m = self.poly[1]
        self.is_inf = False

    def plot(self, title="", show=True, new_figure=True, linestyle="-", marker=""):
        if new_figure:
            plt.figure()
        plt.title(title)
        plt.imshow(self.img)
        plt.plot(self.edge_x_loc, self.edge_y_loc, linestyle=linestyle, marker=marker)
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

    def parse_edge(self, img: NDArray):
        assert(len(img.shape) == 2)
        height, width = img.shape[:2]

        xx, yy = np.meshgrid(np.arange(width), np.arange(height))

        distance = self.get_distance_to_edge(xx, yy).reshape(-1).tolist()
        vals = img.reshape(-1).tolist()

        sorted_xf = sorted(zip(distance, vals))

        esf_x = np.array([x for x, f in sorted_xf])
        esf_f = np.array([f for _, f in sorted_xf])

        if np.mean(esf_f[esf_x < 0]) > np.mean(esf_f[esf_x > 0]):
            esf_x *= -1.0
            esf_x = esf_x[::-1]
            esf_f = esf_f[::-1]
        esf_f -= np.min(esf_f)
        esf_f /= np.max(esf_f)
        return esf_x, esf_f