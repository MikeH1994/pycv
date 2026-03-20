from pycv.utils.functions import gaussian_2d, create_gaussian_kernel
import numpy as np
import matplotlib.pyplot as plt


def test():
    x = np.linspace(-20, 20, 51)
    y = np.linspace(-20, 20, 51)
    xx, yy = np.meshgrid(x, y)
    plt.imshow(gaussian_2d(xx, yy, sigma_x=3.0, sigma_y=3.0))
    plt.figure()
    plt.imshow(gaussian_2d(xx, yy, sigma_x=3.0, sigma_y=1.0))
    plt.figure()
    plt.imshow(gaussian_2d(xx, yy, sigma_x=3.0, sigma_y=3.0, cx = -5, cy=-5))
    plt.show()

def generate_kernel():
    psf = create_gaussian_kernel((31, 31), sigma_x=3.0, sigma_y=3.0, cx=-5.0, cy=-5.0, sqrt_n_samples=30)
    plt.imshow(psf)
    plt.show()


if __name__ == "__main__":
    generate_kernel()
