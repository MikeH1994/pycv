import cv2
import numpy as np
import pycv
from pycv.pinholecamera import InterpolatedDistortionMap, invert_distortion_maps
from pycv import InterpolatedImage, stack, unstack
import matplotlib.pyplot as plt

def create_pinhole_camera():
    f = 806.436
    cx = 335.295
    cy = 244.375
    res = (640, 512)
    camera_matrix = pycv.create_camera_matrix(f, f, cx, cy)
    distortion_coeffs = pycv.create_distortion_coeffs(-0.06203, -1.88952, 4.48810, -0.00061, 0.00050)
    return pycv.PinholeCamera(camera_matrix, res, distortion_coeffs)


def interpolated_distortion_map():
    """
    InterpolatedDistortionMap instances enable the distortion and undistortion maps
    to be continuously sampled without having to know the camera matrix or
    distortion coefficients. This is useful in cases where the distortion
    maps are calculated by extrapolation, or as an average of several other maps.
    It also provides a convenient wrapper to distort and undistort points or images.


    """
    cam = create_pinhole_camera()
    xx, yy = cam.meshgrid()
    distortion_fn_u, distortion_fn_v = cam.create_distortion_map_functions(boundary=100)
    u_map_x, u_map_y = unstack(cam.create_undistortion_map())

    distortion_map = InterpolatedDistortionMap(distortion_fn_u, distortion_fn_v)
    #distortion_map.verify((0, cam.xres), (0, cam.yres))
    u_map_x_2, u_map_y_2 = unstack(distortion_map.undistort_points(stack(xx, yy)))
    pass

def distortion_map_fn():
    pass

def test_inverting_distortion_maps():
    cam = create_pinhole_camera()
    xx, yy = np.meshgrid(np.arange(cam.xres), np.arange(cam.yres))
    d_map_x, d_map_y = pycv.unstack(cam.create_distortion_map())
    u_map_x, u_map_y = pycv.unstack(cam.create_undistortion_map())
    u_map_x_calc, u_map_y_calc = invert_distortion_maps(d_map_x, d_map_y)
    plt.imshow(u_map_x_calc - u_map_x)
    plt.show()

if __name__ == "__main__":
    #test_inverting_distortion_maps()
    interpolated_distortion_map()
    #inverting_distortion_maps()
    #compare_to_opencv()
