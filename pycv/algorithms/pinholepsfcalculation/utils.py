import cv2
import numpy as np
import os
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

def get_radiance_model():
    return RadianceConverter(np.linspace(7000, 14000, 500))


def create_background(bkg_image: np.ndarray, centre: Tuple[float, float], rad: RadianceConverter,
                      aperture_radius_px, roi_width: int=21, interpolate_over_aperture=True):
    assert(len(bkg_image.shape) == 2)
    height, width = bkg_image.shape
    x, y = np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    cx, cy = centre
    x -= cx
    y -= cy
    w = (roi_width - 1) // 2
    roi = bkg_image[int(cy) - w: int(cy) + w + 1, int(cx) - w: int(cx) + w + 1]
    roi = rad.to_radiance(roi, in_celsius=True)

    x = x[int(cx) - w: int(cx) + w + 1]
    y = y[int(cy) - w: int(cy) + w + 1]

    if interpolate_over_aperture:
        xx, yy = np.meshgrid(x, y)
        frac = fraction_of_square_in_circle(xx.flatten(), yy.flatten(), np.full(xx.flatten().shape[0], aperture_radius_px)).reshape(roi.shape)
        # create mask of pixels that cross over into the aperture
        mask = (frac > 0).astype(np.uint8)
        # make expand the mask by 1 pixel just to be sure
        mask = cv2.dilate(mask, np.ones((3,3)))
        # replace these pixels with their nearest valid pixel
        roi = fill_pixels_nearest(roi.astype(np.float32), mask)

    return InterpolatedImage(roi, x, y)


def gaussian_2d(coords, x0, y0, A, sigma, B):
    xx, yy = coords
    r2 = (xx - x0)**2 + (yy - y0)**2
    return (A * np.exp(-r2 / (2 * sigma**2)) + B).reshape(-1)

class FittingFunction:
    def __init__(self, aperture_radius_px, background_image):
        self.aperture_radius_px = aperture_radius_px
        self.background_image = background_image

    def fn(self, coords, *params):
        xx, yy = coords
        x0, y0, k = params[:3]
        param_pairs = [(params[i], params[i+1]) for i in range(3, len(params), 2)]

        dx = xx-x0
        dy = yy-y0

        frac = fraction_of_square_in_circle(dx.flatten(), dy.flatten(),
                                                  np.full(dx.size, self.aperture_radius_px)).reshape(xx.shape)
        total = k #  + (1-frac) * self.background_image(dx, dy)

        for (A, sigma) in param_pairs:
            s = np.sqrt(2.0) * sigma
            ex = (scipy.special.erf((xx + 0.5 - x0) / s) - scipy.special.erf((xx - 0.5 - x0) / s))
            ey = (scipy.special.erf((yy + 0.5 - y0) / s) - scipy.special.erf((yy - 0.5 - y0) / s))
            A * (np.pi * sigma ** 2 / 2.0) * ex * ey
        return total.reshape(-1)


def find_aperture(image: np.ndarray, background_image, aperture_radius_px,
                  radiance_model, mask=None, roi_width=11, apply_blur=True, blur_sigma=0.7, n_terms=3,
                  max_iter=3):
    mask: np.ndarray = np.ones(image.shape[-2:], dtype=np.uint8) if mask is None else mask

    # Find approx location from the max value in the search region
    y_max, x_max = np.unravel_index(np.argmax(image), image.shape)

    # Apply blur if desired
    if apply_blur:
        image = gaussian_filter(np.copy(image), blur_sigma)

    # create an ROI around the max location
    w = (roi_width - 1) // 2
    roi_orig = image[y_max-w:y_max+w+1, x_max-w:x_max+w+1].astype(np.float32)
    xx, yy = np.meshgrid(np.arange(x_max - w, x_max + w + 1), np.arange(y_max - w, y_max + w + 1))
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)

    # create a corresponding max for this roi
    mask_roi = mask[y_max-w:y_max+w+1, x_max-w:x_max+w+1]

    # convert roi to radiance
    roi_orig = radiance_model.to_radiance(roi_orig, in_celsius=True)

    # Do an iterative estimation of the aperture location by fitting a gaussian
    # to the image. With each iteration, as the estimation of the aperture location
    # gets better, we are able to do a better job of removing the background, and
    # hence get closer to the correct location with each iteration.
    current_x = float(x_max)
    current_y = float(y_max)

    for iteration_no in range(max_iter):
        roi = np.copy(roi_orig)

        # Aperture + background correction
        if background_image is not None and aperture_radius_px > 0:
            dx = xx - current_x
            dy = yy - current_y

            # based on our current estimation of where the aperture is,
            # calculate how much of each pixel is inside the aperture
            # the amount looking at the background = 1 - frac
            # we subtract the corresponding amount of radiance from pixels looking at the
            # background, so that all we have left (in theory) is radiance from the aperture
            frac = 1.0 - fraction_of_square_in_circle(dx.flatten(), dy.flatten(),
                                                np.full(dx.size, aperture_radius_px)).reshape(xx.shape)
            roi -= frac * background_image(dx, dy)
            # any pixel less than zero = 0
            roi = np.clip(roi, 0, None)

        # Next, we fit a 2D gaussian to our ROI. The gaussian has an x and y offset term,
        # (dx and dy) which is effectively the correction that we need to make such that
        # our estimate of the aperture location is correct.
        xx_local = xx - current_x
        yy_local = yy - current_y

        initial_guess = [0.0, 0.0, 0.0]
        bounds_lower = [-np.inf, -np.inf, 0.0]
        bounds_upper = [np.inf, np.inf, np.inf]
        for i in range(n_terms):
            bounds_lower += [0, 1e-5]
            bounds_upper += [np.inf, np.inf]
            initial_guess += [1.0, 1.0]
        func = FittingFunction(aperture_radius_px, background_image)
        popt, _ = curve_fit(func.fn,(xx_local[mask_roi>0], yy_local[mask_roi>0]),
                            roi[mask_roi>0], p0=initial_guess, bounds=(bounds_lower, bounds_upper))
        dx_fit, dy_fit = popt[:2]

        current_x += dx_fit
        current_y += dy_fit

    return current_x, current_y



def create_roi(image, centre, roi_width):
    cx, cy = centre
    xmid = int(cx)
    ymid = int(cy)
    w = (roi_width-1)//2

    roi = image[ymid - w: ymid + w + 1, xmid - w: xmid + w + 1].astype(np.float32)
    cx -= (xmid - w)
    cy -= (ymid - w)

    return roi, (cx, cy)

def create_meshgrid(image_size, centre, roi_width):
    cx, cy = int(centre[0]), int(centre[1])
    w = (roi_width-1)//2
    width, height = image_size
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    xx = xx[cy - w:cy + w + 1, cx - w:cx + w + 1].astype(np.float32)
    yy = yy[cy - w:cy + w + 1, cx - w:cx + w + 1].astype(np.float32)
    xx -= centre[0]
    yy -= centre[1]
    return xx, yy

def generate_pixel_data(frames, centres, rad: RadianceConverter, roi_width, mask=None):
    mask: np.ndarray = np.ones(frames[0].shape, dtype=np.uint8) if mask is None else mask
    pixel_values = []
    pixel_x_coords = []
    pixel_y_coords = []
    for i, frame in enumerate(frames):
        #for each frame, crop image around the aperture location
        frame_centre = centres[i]
        roi, roi_centre = create_roi(frame, frame_centre, roi_width)
        mask_roi, _  = create_roi(mask, frame_centre, roi_width)
        # convert the image from temperature to radiance
        roi = rad.to_radiance(roi, in_celsius=True)
        # create pixel coordinates and subtract background radiance
        xx, yy = create_meshgrid((roi.shape[1], roi.shape[0]), roi_centre, roi_width)
        # store pixel values and coordinates
        pixel_values.append(roi[mask_roi>0].flatten().tolist())
        pixel_x_coords.append(xx[mask_roi>0].flatten().tolist())
        pixel_y_coords.append(yy[mask_roi>0].flatten().tolist())
    return np.array(pixel_x_coords), np.array(pixel_y_coords), np.array(pixel_values)

