import cv2
from pycv import InterpolatedImage, fill_pixels_nearest
from pycv.radiometry import RadianceConverter
from .geometry import fraction_of_square_in_circle
from tqdm.auto import tqdm
from typing import Tuple
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import pycv
from skimage.restoration import wiener
from matplotlib.patches import Circle
import numpy as np
import scipy


class ApertureLocator:
    def __init__(self, background_image, radiance_model, roi_radius=1.5, mode="erf", max_iter=3):
        self.background_image = background_image
        self.radiance_model = radiance_model
        self.roi_radius = roi_radius
        self.mode = mode
        self.max_iter = max_iter

    def fit(self, xx, yy, roi, mode, fixed=None):
        fixed = dict() if fixed is None else fixed

        if mode == "erf":
            full_fn = self.erf
        elif mode == "gaussian":
            full_fn = self.gaussian
        else:
            raise ValueError(f"Invalid mode {mode}")

        all_params = {
            "x0": {"value": 0.0, "lower": -2, "upper": 2},
            "y0": {"value": 0.0, "lower": -2, "upper": 2},
            "A": {"value": np.max(roi) / (2*np.pi), "lower": 0, "upper": np.max(roi)},
            "B": {"value": 0.0, "lower": -np.inf, "upper": np.inf},
            "sigma": {"value": 1.0, "lower": 0.1, "upper": 20.0},
        }
        free_names = [p for p in all_params if p not in fixed]
        p0 = [all_params[p]["value"] for p in free_names]
        lower = [all_params[p]["lower"] for p in free_names]
        upper = [all_params[p]["upper"] for p in free_names]

        def wrapped(coords, *free_values):
            params = {}
            idx = 0
            for name in all_params:
                if name in fixed.keys():
                    params[name] = fixed[name]
                else:
                    params[name] = free_values[idx]
                    idx += 1
            return full_fn(coords, params["x0"], params["y0"], params["A"], params["B"], params["sigma"])

        popt, _ = curve_fit(
            wrapped,
            (xx, yy),
            roi,
            p0=p0,
            bounds=(lower, upper),
            maxfev=5000
        )
        fitted_params = {}
        idx = 0
        for name in all_params:
            if name in fixed:
                fitted_params[name] = fixed[name]
            else:
                fitted_params[name] = popt[idx]
                idx += 1

        return fitted_params

    def erf(self, coords, x0, y0, A,  B, sigma):
        xx, yy = coords
        dx = xx-x0
        dy = yy-y0
        s = np.sqrt(2.0) * sigma
        ex = (scipy.special.erf((dx + 0.5) / s) - scipy.special.erf((dx - 0.5) / s))
        ey = (scipy.special.erf((dy + 0.5) / s) - scipy.special.erf((dy - 0.5) / s))
        return A * (np.pi * sigma ** 2 / 2.0) * ex * ey + B

    def gaussian(self, coords, x0, y0, A, B, sigma):
        xx, yy = coords
        r2 = (xx - x0) ** 2 + (yy - y0) ** 2
        return (A * np.exp(-r2 / (2 * sigma ** 2)) + B).reshape(-1)

    def find_aperture(self, image: np.ndarray, aperture_radius_px, mask=None, fixed=None):
        fixed = dict() if fixed is None else fixed
        mask: np.ndarray = np.ones(image.shape[-2:], dtype=np.uint8) if mask is None else mask

        # Find approx location from the max value in the search region
        y_max, x_max = np.unravel_index(np.argmax(image), image.shape)

        # create an ROI around the max location
        xx, yy = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        xx = xx.astype(np.float32)
        yy = yy.astype(np.float32)
        d = np.sqrt((xx-x_max)**2 + (yy-y_max)**2)
        roi_orig = image[d<self.roi_radius*aperture_radius_px]
        mask_roi = mask[d<self.roi_radius*aperture_radius_px]
        xx = xx[d<self.roi_radius*aperture_radius_px]
        yy = yy[d<self.roi_radius*aperture_radius_px]

        # convert roi to radiance
        roi_orig = self.radiance_model.to_radiance(roi_orig, in_celsius=True)

        # Do an iterative estimation of the aperture location by fitting a gaussian
        # to the image. With each iteration, as the estimation of the aperture location
        # gets better, we are able to do a better job of removing the background, and
        # hence get closer to the correct location with each iteration.
        current_x = float(x_max)
        current_y = float(y_max)
        params = []

        for iteration_no in range(self.max_iter):
            roi = np.copy(roi_orig)

            # Aperture + background correction
            if self.background_image is not None and aperture_radius_px > 0:
                dx = xx - current_x
                dy = yy - current_y

                # based on our current estimation of where the aperture is,
                # calculate how much of each pixel is inside the aperture
                # the amount looking at the background = 1 - frac
                # we subtract the corresponding amount of radiance from pixels looking at the
                # background, so that all we have left (in theory) is radiance from the aperture
                frac = 1.0 - fraction_of_square_in_circle(dx.flatten(), dy.flatten(),
                                                    np.full(dx.size, aperture_radius_px)).reshape(xx.shape)
                roi -= frac * self.background_image(dx, dy)
                # any pixel less than zero = 0
                roi = np.clip(roi, 0, None)
            roi /= np.max(roi)

            # Next, we fit a 2D gaussian to our ROI. The gaussian has an x and y offset term,
            # (dx and dy) which is effectively the correction that we need to make such that
            # our estimate of the aperture location is correct.
            xx_local = xx - current_x
            yy_local = yy - current_y

            params = self.fit(xx_local[mask_roi > 0], yy_local[mask_roi > 0], roi[mask_roi>0], self.mode, fixed=fixed)
            dx_fit = params["x0"]
            dy_fit = params["y0"]

            current_x += dx_fit
            current_y += dy_fit

        params["x0"] = current_x
        params["y0"] = current_y
        return params

    def find_apertures(self, images: np.ndarray, aperture_radius_px, mask=None, show_pbar=False):
        A_vals = []
        sigma_vals = []
        B_vals = []
        n_images = images.shape[0]
        pbar = tqdm(range(n_images), desc="Computing aperture locations (1/2)", disable = not show_pbar)
        for i in pbar:
            image = images[i]
            params = self.find_aperture(image, aperture_radius_px, mask=mask)
            A_vals.append(params["A"])
            sigma_vals.append(params["sigma"])
            B_vals.append(params["B"])
        A = np.median(A_vals)
        sigma = np.mean(sigma_vals)
        B = np.median(B_vals)
        aperture_locs = []

        pbar = tqdm(range(n_images), desc="Computing aperture locations (2/2)", disable = not show_pbar)
        for i in pbar:
            image = images[i]
            params = self.find_aperture(image, aperture_radius_px, mask=mask, fixed={"A":A, "sigma":sigma})
            aperture_locs.append((params["x0"], params["y0"]))
        return aperture_locs
