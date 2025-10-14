from typing import List
from ssecorrection.psf.psf import PSF
from ssecorrection.psf.radialpsf import RadialPSF
from skimage.transform import iradon
from scipy.interpolate import interp1d
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


class NonRadialPSF(PSF):
    def __init__(self, psf_list: List[RadialPSF], width, height):
        super().__init__()
        self.psf_list = psf_list
        self.psf_kernel = self.generate_kernel(width, height)

    @staticmethod
    def create_radon_interpolation(slices: List[NDArray], angles: List[float]):
        interp_angles = []
        interp_data = []
        for i in range(len(slices)):
            data_slice = slices[i]
            angle = angles[i]
            interp_angles.append(angle)
            interp_data.append(data_slice)
            interp_angles.append(angle + 180)
            interp_data.append(data_slice)
            interp_angles.append(360 - angle)
            interp_data.append(data_slice)

        sorted_data = sorted(zip(interp_angles, interp_data))
        interp_data = np.array([d for _, d in sorted_data])
        interp_angles = np.array([theta for theta, _ in sorted_data])
        return interp1d(interp_angles, interp_data, axis=0, fill_value="extrapolate")

    def create_radon_image(self, interp_fn, angles):
        dst_img = np.zeros((len(interp_fn(0)), len(angles)))
        for i, angle in enumerate(angles):
            d = interp_fn(angle).reshape(-1)
            dst_img[:, i] = d
        return dst_img

    def generate_kernel(self, width, height):
        angles = []
        slices = []
        for psf in self.psf_list:
            d = psf.generate_kernel(1, height).reshape(-1) # psf.f(np.linspace(-height/2, height/2, height))
            assert(d.shape == (height, ))
            slices.append(d)
            angles.append(psf.angle)
        interp_fn = self.create_radon_interpolation(slices, angles)
        theta = np.linspace(0, 180, 100000)
        radon_image = self.create_radon_image(interp_fn, theta)

        kernel = iradon(radon_image, theta=theta, output_size=width)
        kernel /= np.sum(kernel)
        return kernel
