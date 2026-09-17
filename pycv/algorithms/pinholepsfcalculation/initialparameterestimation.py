import numpy as np
import scipy
import matplotlib.pyplot as plt
from pycv.maths import calculate_fwhm, find_intercepts

def generate_initial_parameter_guess(pixels_x, pixels_y, pixels_radiance, aperture_radius, n_terms, slice_width = 0.02,
                                     k_sigma=1.5, k_w=0.75, sigma_0 = 0.7):

    mask = np.abs(pixels_y) < slice_width/2
    pixels_x = pixels_x[mask]
    pixels_y = pixels_y[mask]
    pixels_radiance = np.copy(pixels_radiance[mask])
    pixels_radiance /= np.max(pixels_radiance)

    init_guess = np.zeros(4*n_terms + 2)


    if aperture_radius <= 0:
        aperture_radius = 0.1
    params = [aperture_radius, 1]
    x = [0.0] * n_terms
    y = [0.0] * n_terms

    # create a set of gaussian widths and amplitudes that get progressively smaller and wider,
    # and integrate to 1 for energy conservation
    w = k_w ** np.arange(n_terms) / np.sum(k_w ** np.arange(n_terms))
    sigmas = sigma_0 * k_sigma ** np.arange(n_terms)
    amplitudes = w / (2 * np.pi * sigmas ** 2)

    approx_fwhm_required = approx_fwhm_for_convolved_aperture()

def sum_of_gaussian_fwhm(amplitudes, sigmas):
    amplitudes = np.array(amplitudes).astype(np.float32).flatten()
    sigmas = np.array(sigmas).astype(np.float32).flatten()
    max_sigma = np.max(sigmas)
    x = np.linspace(-5.0*max_sigma, 5.0*max_sigma, 1000)
    y = np.zeros_like(x)
    coords = np.vstack((x,y))
    f = np.zeros_like(x)
    for i in range(x.shape[0]):
        f += gaussian_2d(coords, 0, 0, amplitudes[i], sigmas[i], 0.0)
    _, fwhm = calculate_fwhm(x, f)
    return fwhm

def calculate_approx_gaussian_sigma_from_line_profile(x_values, radiance_values, plot=True):
    radiance_values = np.copy(radiance_values)
    upper_mask = x_values >= 0
    lower_mask = x_values <= 0


def run():
    pass


if __name__ == "__main__":
    run()