from typing import Union
from numpy.typing import NDArray
import numpy as np
from scipy.integrate import dblquad
from ssecorrection.lsf import LSF
from ssecorrection.psf.psf import PSF


class RadialPSF(PSF):
    params: NDArray
    angle: float

    def __init__(self, lsf: LSF, width: int, height: int):
        super().__init__()
        assert(width % 2 == 1 and height % 2 == 1), "Width and height must be odd"
        self.params = lsf.params
        self.kernel_width = width
        self.kernel_height = height
        self.psf_kernel = self.generate_kernel(width, height)
        self.angle = lsf.angle

    def f(self, r: Union[NDArray, float]):
        return self.fn(r, *self.params)

    def generate_kernel(self, width: int, height: int):
        def integrand(x_, y_):
            return self.f(np.sqrt(x_**2 + y_**2))
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
