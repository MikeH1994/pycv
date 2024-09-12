from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Union
import matplotlib.pyplot as plt
from ...metrics.targets.slantededge.core import get_edge_profile_from_image
from ...metrics.lsf.lsf import LSF
from ...metrics.mtf import MTF


class ESF:
    def __init__(self, x_data: NDArray, f_data: NDArray, normalise=False):
        self.x_data = x_data
        self.f_data = f_data
        if normalise:
            self.x_data, self.f_data = self.normalise_data(self.x_data, self.f_data)

    def normalise_data(self, x_data, f_data):
        raise Exception("Base function ESF.lsf() called")

    def f(self, x: Union[NDArray, float]) -> Union[NDArray, float]:
        raise Exception("Base function ESF.f() called")

    def lsf(self) -> LSF:
        raise Exception("Base function ESF.lsf() called")

    def mtf(self) -> MTF:
        return self.lsf().mtf()

    def plot(self, **kwargs):
        new_figure = kwargs["new_figure"] if "new_figure" in kwargs else False
        show = kwargs["show"] if "show" in kwargs else False
        legend = kwargs["legend"] if "legend" in kwargs else False
        title = kwargs["title"] if "title" in kwargs else None
        xlabel = kwargs["xlabel"] if "xlabel" in kwargs else None
        ylabel = kwargs["ylabel"] if "ylabel" in kwargs else None

        if new_figure:
            plt.figure()
        if title is not None:
            plt.title(title)

        self.plot_elem(**kwargs)

        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        if legend:
            plt.legend(loc=0)
        if show:
            plt.show()

    def plot_elem(self, **kwargs):
        plt.scatter(self.x_data, self.f_data)

    @staticmethod
    def create_data_from_slanted_edge(image: NDArray, edge_detection_mode="fit", **kwargs):
        return get_edge_profile_from_image(image, edge_detection_mode=edge_detection_mode, **kwargs)
