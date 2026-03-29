from pycv.pinholecamera import invert_distortion_maps
import unittest
import numpy as np
import cv2
import pycv
from pycv import stack, unstack

def create_pinhole_camera():
    f = 806.436
    cx = 335.295
    cy = 244.375
    res = (640, 512)
    camera_matrix = pycv.create_camera_matrix(f, f, cx, cy)
    distortion_coeffs = pycv.create_distortion_coeffs(-0.06203, -1.88952, 4.48810, -0.00061, 0.00050)
    return pycv.PinholeCamera(camera_matrix, res, distortion_coeffs)


class TestPinholeCamera(unittest.TestCase):
    def test_distort_undistort(self):
        # check that when we
        cam = create_pinhole_camera()
        xx, yy = np.meshgrid(np.arange(cam.xres), np.arange(cam.yres))
        p_init = stack((xx, yy))
        p_distorted = cam.distort_points(p_init)
        p_undistorted = cam.undistort_points(p_distorted)
        np.testing.assert_allclose(p_init, p_undistorted, atol=1e-6)

    def test_undistort(self):
        cam = create_pinhole_camera()
        xx, yy = np.meshgrid(np.arange(cam.xres), np.arange(cam.yres))
        p_undistorted = cam.undistort_points(stack((xx, yy)))


    def test_distortion(self):
        pass


    def test_create_undistortion_map(self):
        pinhole_camera = create_pinhole_camera()
        undistort_map_x_expected, undistort_map_y_expected = cv2.initUndistortRectifyMap(pinhole_camera.camera_matrix, pinhole_camera.distortion_coeffs, None, None, pinhole_camera.res(), cv2.CV_32FC1)
        undistort_map_calc = pinhole_camera.create_undistortion_map()
        undistort_map_x_calc, undistort_map_y_calc = unstack(undistort_map_calc)
        #np.testing.assert_allclose(undistort_map_x_calc, undistort_map_x_expected, atol=1e-2)
        #np.testing.assert_allclose(undistort_map_y_calc, undistort_map_y_expected, atol=1e-2)


if __name__ == '__main__':
    unittest.main()


