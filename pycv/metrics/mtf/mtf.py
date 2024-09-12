from numpy.typing import NDArray
import matplotlib.pyplot as plt


class MTF:
    def __init__(self, data: NDArray):
        self.data = data

    def plot(self, mode="cy/px"):
        assert(mode == "cy/px" or mode == "cy/rad")
        plt.plot(self.data[:, 0], self.data[:, 1])
        plt.show()