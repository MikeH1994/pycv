import numpy as np
from pycv import InterpolatedImage
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
from .psf import PSF
from .utils import create_subsamples
from scipy.optimize import minimize


def default_params(n_terms=1):
    """x = np.linspace(-1.5, 1.5, int(np.sqrt(n_terms)))
    y = np.linspace(-1.5, 1.5, int(np.sqrt(n_terms)))
    xx, yy = np.meshgrid(x, y)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)"""
    params = [1, 0.5]
    x = [0.0] * n_terms  # [0, -0.2, 0.2, 0, 0]
    y = [0.0] * n_terms  # [0, 0, 0, 0.2, 0.2]
    amplitudes = [1.0, 0.2, 0.1, 0.05]
    widths = [0.5, 1.0, 2.0, 4.0]
    for i in range(n_terms):
        params += [amplitudes[i], widths[i], x[i], y[i]] #
    return np.array(params)

def default_bounds(n_terms=1):
    bounds = [(1e-6, np.inf), (1e-6, np.inf)]
    for i in range(n_terms):
        bounds+= [(1e-6, np.inf), (1e-6, np.inf), (-1, 1), (-1, 1)] # ,
    return bounds

def calculate_brightness(xx, yy, psf, aperture_radius, aperture_brightness, background):
    integral_aperture = psf.integral_over_circle(xx, yy, aperture_radius)
    calculated_brightness = aperture_brightness * integral_aperture
    # add background contribution. Under the assumption that the background over an area of
    # a few square pixels (the main thickness of the PSF) is pretty uniform, we can treat
    # te background as a constant
    integral_non_aperture = psf.integral_over_infinity() - integral_aperture
    calculated_brightness += integral_non_aperture*background(xx, yy)
    return calculated_brightness

def loss_fn(X, measured_brightness, xx, yy, background: InterpolatedImage, pbar=None):
    aperture_radius, aperture_brightness, psf_params = X[0], X[1], X[2:]
    psf = PSF(psf_params)
    calculated_brightness = calculate_brightness(xx, yy, psf, aperture_radius, aperture_brightness, background)

    err = np.mean(np.abs(measured_brightness - calculated_brightness))
    if pbar is not None:
        pbar.update(1)
        pbar.set_description(f"    loss: {err:.4E}")
    return err

def calculate_psf(object_brightness: InterpolatedImage, background:InterpolatedImage,
                  xlim=(-2,2), ylim=(-2,2), n_samples_rough = 100, n_samples=1000, show_progress_bar=True, n_terms=3):
    params = default_params(n_terms=n_terms)
    bounds = default_bounds(n_terms=n_terms)
    for n_s in [n_samples_rough, n_samples]:
        x = np.linspace(xlim[0], xlim[1], n_s)
        y = np.linspace(ylim[0], ylim[1], n_s)
        xx, yy = np.meshgrid(x, y)

        target = object_brightness(xx, yy)
        k = np.max(target)
        target /= k
        background.scale_image(1.0/k)
        pbar = tqdm(disable=not show_progress_bar)
        params = minimize(loss_fn, params, args=(target, xx, yy, background, pbar), bounds=bounds, method='L-BFGS-B').x
        pbar.close()
        background.scale_image(k)
    print(params)
    aperture_radius, aperture_brightness, psf_params = params[0], params[1], params[2:]
    aperture_brightness *= k
    psf = PSF(psf_params)

    return psf, aperture_radius, aperture_brightness