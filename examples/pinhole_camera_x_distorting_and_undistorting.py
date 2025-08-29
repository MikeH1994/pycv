import pycv
import numpy as np
import matplotlib.pyplot as plt

fx = fy = 793.40766
cx = 333.77654
cy = 244.74982
k1 = -0.075530
k2 = -1.718930
k3 = 3.8330500

camera_matrix = pycv.create_camera_matrix(cx, cy, fx, fy)
distortion_coeffs = pycv.create_distortion_coeffs(k1, k2, k3, 0, 0)
camera = pycv.PinholeCamera(camera_matrix, (640, 512), distortion_coeffs)

d = 100
xx, yy = np.meshgrid(np.arange(0, 640, d), np.arange(0, 512, d))
points_distorted = camera.distort_points(pycv.stack_coords((xx, yy)))
xx_distorted, yy_distorted = pycv.unstack_coords(points_distorted)
xx_undistorted, yy_undistorted = pycv.unstack_coords(camera.undistort_points(pycv.stack_coords((xx_distorted, yy_distorted))))

dx = xx - xx_undistorted
dy = yy - yy_undistorted
print(np.mean(np.abs(dx)), np.mean(np.abs(dy)))

plt.scatter(xx, yy, label="init")
plt.scatter(xx_distorted, yy_distorted, label="distorted")
plt.scatter(xx_undistorted, yy_undistorted, label="undistorted")
plt.legend(loc=0)
plt.show()

