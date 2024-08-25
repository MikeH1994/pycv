from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Union
from ...metrics.targets.slantededge.core import get_edge_profile_from_image

class ESF:
    def __init__(self, x_data: NDArray, f_data: NDArray):
        self.x_data = x_data
        self.f_data = f_data

    def f(self, x: Union[NDArray, float]) -> Union[NDArray, float]:
        raise Exception("Base function ESF.f() called")

    def lsf(self):
        raise Exception("Base function ESF.lsf() called")

    @staticmethod
    def create_from_slanted_edge(image: NDArray, kind="binned", **kwargs):
        data = get_edge_profile_from_image(image)
        edge_data = data["edge_profiles"][0]
        x_data = edge_data[:, 0]
        f_data = edge_data[:, 1]
        if kind == "binned":
            pass
        elif kind == "gaussian":
            pass
        elif kind == "cauchy":
            pass