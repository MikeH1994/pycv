import pycv
from pycv.pinholecamera import InterpolatedDistortionMap

f = 806.436
cx = 335.295
cy = 244.375
camera_matrix = pycv.create_camera_matrix(f, f, cx, cy)
distortion_coeffs = pycv.create_distortion_coeffs(-0.06203, -1.88952, 4.48810, -0.00061, 0.00050)
pinhole_camera = pycv.PinholeCamera(camera_matrix, (640, 512), distortion_coeffs)
