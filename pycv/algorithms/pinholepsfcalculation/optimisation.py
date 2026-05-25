import numpy as np
from pycv import InterpolatedImage
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
from .psf import PSF
from .utils import create_subsamples
from scipy.optimize import minimize


class Aperture:
    def __init__(self, x_width, y_width, n, theta, background: InterpolatedImage = None, aperture_brightness=1.0):
        self.x0 = 0.0
        self.y0 = 0.0
        self.x_width = x_width
        self.y_width = y_width
        self.n = n
        self.theta = theta
        self.aperture_brightness = aperture_brightness
        self.background = background

    def __call__(self, xx, yy):
        a = self.x_width
        b = self.y_width
        z = np.zeros_like(xx)
        inside = point_inside_superellipse(xx, yy, a, b, self.n, self.x0, self.y0, self.theta)
        z[inside] = self.aperture_brightness
        if self.background is not None:
            z[~inside] = self.background(xx[~inside], yy[~inside])
        return z

def default_params(n_terms=1):
    """x = np.linspace(-1.5, 1.5, int(np.sqrt(n_terms)))
    y = np.linspace(-1.5, 1.5, int(np.sqrt(n_terms)))
    xx, yy = np.meshgrid(x, y)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)"""
    params = [0.5]
    for i in range(n_terms):
        params += [5, 5, 0.0, 0.0]
    return np.array(params)

def default_bounds(n_terms=1):
    bounds = [(0.1, 10.0)]
    for i in range(n_terms):
        bounds+= [(0.0, 10.0), (0.0001, 10.0), (-5.0, 5.0), (-5.0, 5.0)]
    return bounds

def calculate_brightness(xx, yy, psf, aperture_radius, aperture_brightness, background):
    integral_aperture = psf.integral_over_circle(xx, yy, aperture_radius)
    integral_non_aperture = psf.integral_over_infinity() - integral_aperture
    calculated_brightness = aperture_brightness * integral_aperture
    calculated_brightness += integral_non_aperture*background(xx, yy)
    return calculated_brightness



def loss_fn(X, measured_brightness, xx, yy, background: InterpolatedImage, aperture_brightness=1.0, pbar=None):
    aperture_radius, psf_params = X[0], X[1:]
    psf = PSF(psf_params)
    calculated_brightness = calculate_brightness(xx, yy, psf, aperture_radius, aperture_brightness, background)

    err = np.mean(np.abs(measured_brightness - calculated_brightness))
    if pbar is not None:
        pbar.update(1)
        pbar.set_description(f"    loss: {err:.4E}")
    return err

def calculate_psf(object_brightness: InterpolatedImage, background:InterpolatedImage,
                  xx=None, yy=None, source_radiance=1.0):
    if xx is None or yy is None:
        x = np.linspace(-1, 1, 30)
        y = np.linspace(-1, 1, 30)
        xx, yy = np.meshgrid(x, y)

    target = object_brightness(xx, yy)
    target /= source_radiance
    background.scale_image(1.0/source_radiance)

    x0 = default_params(n_terms=1)
    bounds = default_bounds(n_terms=1)
    pbar = tqdm()

    print("Init loss: {}".format(loss_fn(x0, target, xx, yy, background, 1.0)))
    res = minimize(loss_fn, x0, args=(target, xx, yy, background, 1.0, pbar), bounds=bounds, method='L-BFGS-B')
    print("End loss: {}".format(loss_fn(res.x, target, xx, yy, background, 1.0)))
    print(res.x[0])
    print(res.x[1:].reshape(-1, 4))
    background.scale_image(source_radiance)
    return res

def superellipse_s(x, y, a, b, n, x0=0.0, y0=0.0, theta=0.0):
    """
    Implicit superellipse value S(x,y) = |x'/a|^n + |y'/b|^n
    where (x',y') is (x,y) transformed into the shape's local frame
    via translation (x0,y0) and rotation theta (CCW).
    """
    # shift into local frame
    dx, dy = x - x0, y - y0
    c, s = np.cos(theta), np.sin(theta)
    xp =  dx * c + dy * s
    yp = -dx * s + dy * c

    # signed-power pattern to stay real for fractional n
    S = (np.abs(xp / a) ** n) + (np.abs(yp / b) ** n)
    return S

def point_inside_superellipse(x, y, a, b, n, x0=0.0, y0=0.0, theta=0.0):
    s = superellipse_s(x, y, a, b, n, x0, y0, theta)
    return s <= 1

