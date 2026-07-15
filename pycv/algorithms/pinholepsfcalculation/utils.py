import cv2
import numpy as np
import os
from pycv import InterpolatedImage
from pycv.radiometry import RadianceModel
from .geometry import fraction_of_square_in_circle
from tqdm.auto import tqdm
from typing import Tuple
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import pycv
from skimage.restoration import wiener


def get_radiance_model():
    return RadianceModel(np.linspace(7000, 14000, 500))


def create_background(bkg_image: np.ndarray, centre: Tuple[float, float], radiance_model, roi_width: int=21):
    if len(bkg_image.shape) == 3:
        bkg_image = np.mean(bkg_image, axis=0)
    height, width = bkg_image.shape
    x, y = np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    cx, cy = centre
    x -= cx
    y -= cy
    w = (roi_width - 1) // 2
    roi = bkg_image[int(cy) - w: int(cy) + w + 1, int(cx) - w: int(cx) + w + 1]
    roi = radiance_model.temperature_to_radiance_fn(roi + 273.15)

    x = x[int(cx) - w: int(cx) + w + 1]
    y = y[int(cy) - w: int(cy) + w + 1]

    return InterpolatedImage(roi, x, y)



def gaussian_2d(coords, x0, y0, A, sigma, B):
    xx, yy = coords
    r2 = (xx - x0)**2 + (yy - y0)**2
    return (A * np.exp(-r2 / (2 * sigma**2)) + B).reshape(-1)





def find_aperture(
    image: np.ndarray,
    guess,
    background_image,
    aperture_radius_px,
    radiance_model=None,
    roi_width=7,
    search_window=100,
    apply_blur=True,
    blur_sigma=0.7,
    max_iter=5,
    tol=1e-3,
):
    height, width = image.shape
    guess_x, guess_y = guess

    # -----------------------------
    # 1. Coarse peak search
    # -----------------------------
    w = (search_window - 1) // 2

    search_region = image[
        guess_y - w : guess_y + w + 1,
        guess_x - w : guess_x + w + 1,
    ]

    dy_max, dx_max = np.unravel_index(np.argmax(search_region), search_region.shape)

    coarse_y = np.arange(height)[guess_y - w : guess_y + w + 1][dy_max]
    coarse_x = np.arange(width)[guess_x - w : guess_x + w + 1][dx_max]

    # -----------------------------
    # 2. Optional blur
    # -----------------------------
    if apply_blur:
        image = gaussian_filter(np.copy(image), blur_sigma)

    # -----------------------------
    # 3. Extract ROI
    # -----------------------------
    w = (roi_width - 1) // 2

    roi_orig = image[
        coarse_y - w : coarse_y + w + 1,
        coarse_x - w : coarse_x + w + 1,
    ].astype(np.float32)

    xx, yy = np.meshgrid(
        np.arange(coarse_x - w, coarse_x + w + 1),
        np.arange(coarse_y - w, coarse_y + w + 1),
    )

    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)

    # Radiance conversion
    if radiance_model is not None:
        roi_orig = radiance_model.temperature_to_radiance_fn(roi_orig + 273.15)

    # -----------------------------
    # 4. Iterative refinement
    # -----------------------------
    current_x = float(coarse_x)
    current_y = float(coarse_y)

    for iteration_no in range(max_iter):
        roi = np.copy(roi_orig)

        # ---- Aperture + background correction
        if background_image is not None and aperture_radius_px > 0:
            dx = xx - current_x
            dy = yy - current_y

            frac = fraction_of_square_in_circle(
                dx.flatten(),
                dy.flatten(),
                np.full(dx.size, aperture_radius_px),
            ).reshape(xx.shape)

            roi -= (1.0 - frac) * background_image(dx, dy)
            roi = np.clip(roi, 0, None)

        # ---- Fit Gaussian
        xx_local = xx - current_x
        yy_local = yy - current_y

        initial_guess = (0.0, 0.0, np.max(roi), 1.0, np.median(roi))

        bounds = (
            [-2, -2, 0, 0.3, -np.inf],
            [2, 2, np.inf, 5.0, np.inf],
        )

        try:
            popt, _ = curve_fit(
                gaussian_2d,
                (xx_local, yy_local),
                roi.reshape(-1),
                p0=initial_guess,
                bounds=bounds,
                maxfev=5000,
            )

            dx_fit, dy_fit, _, _, _ = popt

        except RuntimeError:
            # fallback: center of mass
            weights = roi
            total = np.sum(weights)

            if total > 0:
                dx_fit = np.sum(xx_local * weights) / total
                dy_fit = np.sum(yy_local * weights) / total
            else:
                dx_fit, dy_fit = 0.0, 0.0

        new_x = current_x + dx_fit
        new_y = current_y + dy_fit

        # ---- Convergence check
        if np.hypot(new_x - current_x, new_y - current_y) < tol:
            current_x, current_y = new_x, new_y
            break

        current_x, current_y = new_x, new_y

    # -----------------------------
    # 5. Final result
    # -----------------------------
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

def generate_pixel_data(frames, centres, radiance_model, roi_width):
    pixel_values = []
    pixel_x_coords = []
    pixel_y_coords = []
    for i, frame in enumerate(frames):
        #for each frame, crop image around the aperture location
        frame_centre = centres[i]
        roi, roi_centre = create_roi(frame, frame_centre, roi_width)
        # convert the image from temperature to radiance
        roi = radiance_model.temperature_to_radiance_fn(roi + 273.15)
        # create pixel coordinates and subtract background radiance
        xx, yy = create_meshgrid((roi.shape[1], roi.shape[0]), roi_centre, roi_width)
        # store pixel values and coordinates
        pixel_values.append(roi.flatten().tolist())
        pixel_x_coords.append(xx.flatten().tolist())
        pixel_y_coords.append(yy.flatten().tolist())
    return np.array(pixel_x_coords), np.array(pixel_y_coords), np.array(pixel_values)


def create_subsamples(min_val, max_val, n, p=3):
    u = np.linspace(-1, 1, n)
    y = np.sign(u) * np.abs(u)**p
    y = (y + 1)/2
    return y*(max_val-min_val) + min_val


def create_interpolated_image(pixels_x, pixels_y, pixels_vals, knn_sigma=0.2, knn_k=30, n_samples=15):
    xmin, xmax = np.min(pixels_x), np.max(pixels_x)
    ymin, ymax = np.min(pixels_y), np.max(pixels_y)
    x = np.linspace(xmin, xmax, int((xmax - xmin)*n_samples))
    y = np.linspace(ymin, ymax, int((ymax - ymin)*n_samples))
    xx, yy = np.meshgrid(x, y)
    pixels_x = np.array(pixels_x)
    pixels_y = np.array(pixels_y)
    pixels_vals = np.array(pixels_vals)
    zz = gaussian_knn(pixels_x, pixels_y, pixels_vals, xx, yy, knn_sigma, knn_k)
    return InterpolatedImage(zz, xx, yy)

def undo_pixelisation(interpolated_image, samples_per_pixel=10,  wiener_lambda=0.1):
    xmin, xmax = np.min(interpolated_image.x) + 0.5, np.max(interpolated_image.x) - 0.5
    ymin, ymax = np.min(interpolated_image.y) + 0.5, np.max(interpolated_image.y) - 0.5
    d = 1.0 / samples_per_pixel
    xmin = pycv.round_up_to(xmin, d)
    xmax = pycv.round_up_to(xmax, d)
    ymin = pycv.round_up_to(ymin, d)
    ymax = pycv.round_up_to(ymax, d)
    xx, yy = np.meshgrid(np.arange(xmin, xmax, d), np.arange(ymin, ymax, d))

    image_samples = interpolated_image(xx, yy)
    image_samples_deconv = wiener_deconv(image_samples, samples_per_pixel, samples_per_pixel, balance=wiener_lambda)
    dst_img = InterpolatedImage(image_samples_deconv, xx, yy)
    return dst_img


def gaussian_knn(x, y, vals, xx, yy, sigma, k=12):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    vals = np.asarray(vals).ravel()
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(vals))
    x, y, vals = x[mask], y[mask], vals[mask]
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    pts = np.column_stack([x, y])
    tree = cKDTree(pts)
    dist, idx = tree.query(grid, k=k)  # (M, k)

    w = np.exp(-(dist ** 2) / (2 * sigma * sigma))
    w_sum = np.sum(w, axis=1, keepdims=True)
    w_sum[w_sum == 0] = 1.0
    vals_i = np.sum(w * vals[idx], axis=1) / w_sum.ravel()
    vals_i = vals_i.reshape(xx.shape)
    return vals_i

def wiener_deconv(img, x_samples, y_samples, balance=1e-3):
    k = np.max(img)
    img /= k
    psf = np.ones((y_samples, x_samples), dtype=float) / (x_samples * y_samples)
    g_hat = wiener(img, psf, balance=balance) * k
    return np.asarray(g_hat, dtype=float)
