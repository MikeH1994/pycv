import matplotlib
import numpy as np

import pycv
matplotlib.use('Qt5Agg')  # or
import matplotlib.pyplot as plt
from pycv.plt import plt_fig_to_rgb
from pycv import InterpolatedImage
from .psf import PSF
from .optimisation import calculate_brightness

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

def plot_pixel_values(pixel_x, pixel_y, pixel_val, title="", show=True):
    plt.figure()
    scatter = plt.scatter(pixel_x, pixel_y, c=pixel_val, cmap='viridis')
    #plt.colorbar(scatter, label='vals')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    if show:
        plt.show()

def show_interpolated_image(interpolated_image, xlim = (-2, 2), ylim = (-2, 2), title="", show=True):
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 1000), np.linspace(ylim[0], ylim[1], 1000))
    img = interpolated_image(xx, yy)

    plt.figure()
    plt.pcolormesh(xx, yy, img, shading='auto', cmap='viridis')
    plt.colorbar(label='intensity')
    plt.title(title)
    if show:
        plt.show()

def create_simulated_image(psf: PSF, background: InterpolatedImage, aperture_radius: float, aperture_brightness: float,
                           xlim=(-4, 4), ylim=(-4, 4), n_samples=1000):
    x = np.linspace(xlim[0], xlim[1], n_samples)
    y = np.linspace(ylim[0], ylim[1], n_samples)
    xx, yy = np.meshgrid(x, y)
    vals = calculate_brightness(xx, yy, psf, aperture_radius, aperture_brightness, background)
    return InterpolatedImage(vals, xx, yy)