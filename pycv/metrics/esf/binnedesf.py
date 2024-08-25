from __future__ import annotations
from .esf import ESF
from scipy.optimize import curve_fit
from scipy.special import erf
from numpy.typing import NDArray
from typing import Union
import numpy as np

class BinnedESF(ESF):
    def __init__(self, x_data: NDArray, f_data: NDArray, bins_per_pixel=4, **kwargs):
        super().__init__(x_data, f_data)
        self.bins_per_pixel = bins_per_pixel



