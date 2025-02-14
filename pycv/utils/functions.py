import numpy as np
from scipy.integrate import simps


def gaussian_2d(x, y, sigma_x=1.0, sigma_y=1.0, cx=0.0, cy=0.0):
    return 1/(2*np.pi*sigma_x*sigma_y)*np.exp(-((x-cx)**2)/(2*sigma_x**2) -((y-cy)**2)/(2*sigma_y**2))


def create_gaussian_kernel(psf_size, sigma_x=1.0, sigma_y=1.0, cx=0.0, cy=0.0, sqrt_n_samples=300):
    h, w = psf_size
    x_offset = (w-1)/2
    y_offset = (h-1)/2
    psf_kernel = np.zeros(psf_size)
    for x in range(w):
        for y in range(h):
            x0 = x - x_offset - 0.5
            x1 = x - x_offset + 0.5
            y0 = y - y_offset - 0.5
            y1 = y - y_offset + 0.5
            x_samples = np.linspace(x0, x1, sqrt_n_samples)
            y_samples = np.linspace(y0, y1, sqrt_n_samples)
            xx, yy = np.meshgrid(x_samples, y_samples)
            f_values = gaussian_2d(xx, yy, sigma_x, sigma_y, cx, cy)
            psf_kernel[y, x] = simps(simps(f_values, x_samples, axis=-1), y_samples)
    return psf_kernel
