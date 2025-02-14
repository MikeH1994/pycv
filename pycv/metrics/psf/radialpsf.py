from typing import Union
from numpy.typing import NDArray
import numpy as np
from scipy.integrate import dblquad
from .psf import PSF
from ..lsf import LSF
from .utils import kernel_to_interpolation_fn


class RadialPSF(PSF):
    params: NDArray
    angle: float

    def __init__(self, lsf: LSF, width: int, height: int):
        assert(width % 2 == 1 and height % 2 == 1), "Width and height must be odd"
        self.params = lsf.params
        self.kernel_width = width
        self.kernel_height = height
        self.psf_kernel = self.generate_kernel(width, height)
        super().__init__(kernel_to_interpolation_fn(self.psf_kernel))

    def generate_kernel(self, width: int, height: int):
        def integrand(x_, y_):
            r = np.sqrt(x_**2 + y_**2)
            return self.fn(r, *self.params)
        assert(width % 2 == 1 and height % 2 == 1), "Width and height must be odd"

        dst_kernel = np.zeros((height, width))
        for j in range(height):
            for i in range(width):
                x = i - width // 2
                y = j - height // 2
                value, _ = dblquad(integrand, x-0.5, x+0.5, y-0.5, y+0.5)
                dst_kernel[j, i] = value
        dst_kernel /= np.sum(dst_kernel)
        return dst_kernel

    @staticmethod
    def fn(r, *args):
        raise ValueError("In RSF.fn(): base function called")
