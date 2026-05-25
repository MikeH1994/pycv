import matplotlib
import numpy as np

matplotlib.use('Qt5Agg')  # or
import matplotlib.pyplot as plt

def show_video(images, centres, roi_size=80, delay_ms=50, cmap="gray", title=""):
    w = (roi_size-1)//2
    plt.ion()  # interactive mode
    fig, ax = plt.subplots()

    im = ax.imshow(images[0], cmap=cmap, animated=True)

    ax.axis("off")
    delay_s = delay_ms / 1000.0


    centre_loc_x, = ax.plot([0, 0], [0,0], "r-", linewidth=0.75, label="Aperture location")
    centre_loc_y, = ax.plot([0, 0], [0, 0], "r-", linewidth=0.75)
    plt.legend(loc=0)

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
        plt.pause(delay_s)

    plt.ioff()
    plt.show()
    plt.close("all")

def plot_pixel_values(pixel_x, pixel_y, pixel_val):
    plt.figure()
    scatter = plt.scatter(pixel_x, pixel_y, c=pixel_val, cmap='viridis')
    plt.colorbar(scatter, label='vals')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def show_interpolated_image(interpolated_image, xlim = (-2, 2), ylim = (-2, 2)):
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 1000), np.linspace(ylim[0], ylim[1], 1000))
    img = interpolated_image(xx, yy)

    plt.pcolormesh(xx, yy, img, shading='auto', cmap='viridis')
    plt.colorbar(label='intensity')
    plt.show()