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

