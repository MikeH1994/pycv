from pycv.pinholecamera import invert_distortion_maps
import unittest
import numpy as np
import cv2
import pycv

def create_pinhole_camera():
    f = 806.436
    cx = 335.295
    cy = 244.375
    res = (640, 512)
    camera_matrix = pycv.create_camera_matrix(f, f, cx, cy)
    distortion_coeffs = pycv.create_distortion_coeffs(-0.06203, -1.88952, 4.48810, -0.00061, 0.00050)
    return pycv.PinholeCamera(camera_matrix, res, distortion_coeffs)

class TestInvertDistortionMaps(unittest.TestCase):
    def setUp(self):
        self.height = 10
        self.width = 10
        self.map_u, self.map_v = np.meshgrid(np.arange(self.width), np.arange(self.height))

    def test_identity_map(self):
        # Identity distortion maps
        inverted_u, inverted_v = invert_distortion_maps(self.map_u.astype(np.float32), self.map_v.astype(np.float32))
        np.testing.assert_allclose(inverted_u, self.map_u, atol=1e-2)
        np.testing.assert_allclose(inverted_v, self.map_v, atol=1e-2)

    def test_output_shape(self):
        distorted_u = self.map_u + 0.5
        distorted_v = self.map_v + 0.5
        inverted_u, inverted_v = invert_distortion_maps(distorted_u.astype(np.float32), distorted_v.astype(np.float32))
        self.assertEqual(inverted_u.shape, (self.height, self.width))
        self.assertEqual(inverted_v.shape, (self.height, self.width))

    def test_no_nan_or_inf(self):
        distorted_u = self.map_u + np.random.randn(*self.map_u.shape) * 0.1
        distorted_v = self.map_v + np.random.randn(*self.map_v.shape) * 0.1
        inverted_u, inverted_v = invert_distortion_maps(distorted_u.astype(np.float32), distorted_v.astype(np.float32))
        self.assertFalse(np.isnan(inverted_u).any())
        self.assertFalse(np.isnan(inverted_v).any())
        self.assertFalse(np.isinf(inverted_u).any())
        self.assertFalse(np.isinf(inverted_v).any())

    def test_small_translation_distortion(self):
        cam = create_pinhole_camera()
        xx, yy = np.meshgrid(np.arange(cam.xres), np.arange(cam.yres))
        d_map_x, d_map_y = pycv.unstack(cam.create_distortion_map())
        u_map_x, u_map_y = pycv.unstack(cam.create_undistortion_map())
        inverted_u, inverted_v = invert_distortion_maps(d_map_x, d_map_y)
        pass


        np.testing.assert_allclose(inverted_u, expected_u, atol=1.0)
        np.testing.assert_allclose(inverted_v, expected_v, atol=1.0)


    def test_lens_distortion(self):
        pinhole_camera = create_pinhole_camera()


if __name__ == '__main__':
    unittest.main()
