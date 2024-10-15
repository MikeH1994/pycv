from ssecorrection.psf.radialpsf import RadialPSF
import numpy as np
from ssecorrection.lsf import GaussianLSF


class GaussianPSF(RadialPSF):
    def __init__(self, lsf: GaussianLSF, width: int, height: int):
        super().__init__(lsf, width, height)

    @staticmethod
    def fn(r, *params):
        if len(params) % 2 != 0:
            raise ValueError("In fn_PSF: the number of terms in the function must be even")
        f = 0.0
        for i in range(len(params) // 2):
            a_i = params[i * 2]
            b_i = params[i * 2 + 1]
            f += 2.0 / np.pi * a_i / b_i ** 2 * np.exp(-r**2 / b_i ** 2)
        return f
