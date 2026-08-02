import matplotlib
import numpy as np
import pycv
import matplotlib.pyplot as plt
from .optimisation import calculate_brightness
from .psf import PSF
from pycv.plt import plt_fig_to_rgb, set_colorbar, set_labels_and_legend
from pycv import InterpolatedImage
from scipy.stats import qmc

def show_video(images, centres, roi_size=80, delay_ms=50, cmap="gray", title="", output_fpath = "", show=True):
    w = (roi_size-1)//2
    if show:
        plt.ion()  # interactive mode
    fig, ax = plt.subplots()

    im = ax.imshow(images[0], cmap=cmap, animated=True)

    ax.axis("off")
    delay_s = delay_ms / 1000.0


    centre_loc_x, = ax.plot([0, 0], [0,0], "r-", linewidth=0.75, label="Aperture location")
    centre_loc_y, = ax.plot([0, 0], [0, 0], "r-", linewidth=0.75)
    plt.legend(loc=0)

    plt_images = []

    for i in range(len(images)):
        frame = images[i]
        cx, cy = centres[i]

        if i == 0:
            ax.set_xlim(cx - w, cx + w)
            ax.set_ylim(cy + w, cy - w)

        plt.title(title + f" {i+1} / {len(images)}")
        im.set_data(frame)
        centre_loc_x.set_data([cx-3, cx+3], [cy, cy])
        centre_loc_y.set_data([cx, cx], [cy-3, cy+3])
        if show:
            plt.pause(delay_s)
        plt_images.append(plt_fig_to_rgb())


    plt.ioff()
    #plt.show()
    plt.close("all")

    if output_fpath != "":
        pycv.write_avi(plt_images, output_fpath, int(1000/delay_ms))

def show_interpolated_image(interpolated_image, xlim = (-2, 2), ylim = (-2, 2), title="", show=True):
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 1000), np.linspace(ylim[0], ylim[1], 1000))
    img = interpolated_image(xx, yy)

    plt.figure()
    plt.pcolormesh(xx, yy, img, shading='auto', cmap='viridis')
    plt.colorbar(label='intensity')
    plt.title(title)
    if show:
        plt.show()

def display_error(x, y, pixel_vals, psf: PSF, background: InterpolatedImage, aperture_radius: float,
                  aperture_brightness: float, xlim=(-4, 4), ylim=(-0.01, 0.01), n_subsamples=100):
    mask = (x > xlim[0]) & (x < xlim[1]) & (y > ylim[0]) & (y < ylim[1])
    x = x[mask]
    y = y[mask]
    pixel_vals = pixel_vals[mask]

    rng = np.random.default_rng(1234)
    subsamples = qmc.Halton(d=2, scramble=True, rng=rng).random(n_subsamples)
    x_samples = np.zeros((x.shape[0], n_subsamples)) + x.reshape(-1, 1) + subsamples[:, 0]
    y_samples = np.zeros((y.shape[0], n_subsamples)) + y.reshape(-1, 1) + subsamples[:, 1]
    l_bkg = background(x_samples, y_samples)
    brightness = calculate_brightness(x_samples, y_samples, psf, aperture_radius, aperture_brightness, l_bkg)
    brightness = np.mean(brightness, axis=-1)

    plt.scatter(x, brightness, label="Calculated")
    plt.scatter(x, pixel_vals, label="Measured")
    plt.legend(loc=0)
    plt.show()

def create_simulated_image(psf: PSF, background: InterpolatedImage, aperture_radius: float, aperture_brightness: float,
                           xlim=(-4, 4), ylim=(-4, 4), n_samples=1000):
    x = np.linspace(xlim[0], xlim[1], n_samples)
    y = np.linspace(ylim[0], ylim[1], n_samples)
    xx, yy = np.meshgrid(x, y)
    vals = calculate_brightness(xx, yy, psf, aperture_radius, aperture_brightness, background)
    return InterpolatedImage(vals, xx, yy)