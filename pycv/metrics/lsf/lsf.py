from numpy.typing import NDArray
from typing import Union
import matplotlib.pyplot as plt


class LSF:
    def __init__(self, x_data: NDArray, f_data: NDArray):
        self.x_data = x_data
        self.f_data = f_data

    def plot(self, **kwargs):
        new_figure = kwargs["new_figure"] if "new_figure" in kwargs else False
        show = kwargs["show"] if "show" in kwargs else False
        title = kwargs["title"] if "title" in kwargs else None
        xlabel = kwargs["xlabel"] if "xlabel" in kwargs else None
        ylabel = kwargs["ylabel"] if "ylabel" in kwargs else None

        if new_figure:
            plt.figure()
        if title is not None:
            plt.title(title)

        self.plot_elem()

        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        if show:
            plt.show()

    def plot_elem(self):
        plt.scatter(self.x_data, self.f_data)

    def f(self, x: Union[NDArray, float]) -> Union[NDArray, float]:
        raise Exception("Base function LSF.f() called")

    def psf(self):
        raise Exception("Base function LSF.lsf() called")

    def mtf(self):
        raise Exception("Base function LSF.lsf() called")
