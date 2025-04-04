from numpy.typing import NDArray
import matplotlib.pyplot as plt
from ...metrics.metric import Metric
from pycv.utils.maths import find_intercepts

class MTF(Metric):
    def __init__(self, data: NDArray):
        self.data = data
        self.default_x_label = "cy/mm"
        self.default_y_label = "MTF"
        self.ifov_scale_factor = 1.0

    def mtf50(self):
        return find_intercepts(self.data[:, 0], self.data[:, 1], 0.5)[0]

    def plot_elem(self, **kwargs):
        mode = kwargs["mode"] if "mode" in kwargs else "cy/px"
        label = kwargs["label"] if "label" in kwargs else "MTF"
        assert(mode == "cy/px" or mode == "cy/rad")
        self.default_x_label = mode
        freq = self.data[:, 0]
        mtf = self.data[:, 1]
        if mode == "cy/rad":
            print("Scaling frequencies by {}".format(self.ifov_scale_factor))
            freq /= self.ifov_scale_factor
        plt.scatter(freq, mtf, label=label)
