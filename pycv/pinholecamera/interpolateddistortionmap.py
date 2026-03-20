import pycv
from ..pinholecamera import invert_distortion_maps
from ..core import stack, unstack
from ..imageutils.interpolated_image import InterpolatedImage
import numpy as np
import scipy
import scipy.interpolate
import cv2
from typing import Tuple, Union


class InterpolatedDistortionMap:
    def __init__(self, distortion_u: InterpolatedImage, distortion_v: InterpolatedImage,
                 undistortion_u: InterpolatedImage = None, undistortion_v: InterpolatedImage = None, res: Tuple[int, int] = None):
        self.distortion_map_u: InterpolatedImage = distortion_u
        self.distortion_map_v: InterpolatedImage = distortion_v

        if undistortion_u is not None and undistortion_v is not None:
            self.undistortion_map_u: InterpolatedImage = undistortion_u
            self.undistortion_map_v: InterpolatedImage = undistortion_v
        else:
            self.undistortion_map_u, self.undistortion_map_v = self.invert_distortion_map()

        assert(np.array_equal(self.distortion_map_u.x, self.distortion_map_v.x))
        assert(np.array_equal(self.undistortion_map_u.x, self.undistortion_map_v.x))
        assert(np.array_equal(self.undistortion_map_u.x, self.distortion_map_u.x))
        assert(np.array_equal(self.distortion_map_u.y, self.distortion_map_v.y))
        assert(np.array_equal(self.undistortion_map_u.y, self.undistortion_map_v.y))
        assert(np.array_equal(self.distortion_map_u.y, self.undistortion_map_v.y))

        self.res = res

    def distort_points(self, p: np.ndarray) -> np.ndarray:
        assert(isinstance(p, np.ndarray))
        u, v = unstack(p)

        u_dist: np.ndarray = self.distortion_map_u(u, v)
        v_dist: np.ndarray = self.distortion_map_v(u, v)
        return stack((u_dist, v_dist))

    def undistort_points(self, p: np.ndarray):
        assert(isinstance(p, np.ndarray))
        u, v = unstack(p)

        u_dist: np.ndarray = self.undistortion_map_u(u, v)
        v_dist: np.ndarray = self.undistortion_map_v(u, v)
        return stack((u_dist, v_dist))

    def invert_distortion_map(self):
        assert(np.array_equal(self.distortion_map_u.x, self.distortion_map_v.x))
        assert(np.array_equal(self.distortion_map_u.y, self.distortion_map_v.y))
        distortion_map_u = self.distortion_map_u.img
        distortion_map_v = self.distortion_map_v.img
        undistortion_map_u, undistortion_map_v = invert_distortion_maps(distortion_map_u, distortion_map_v)

        x = self.distortion_map_u.x
        y = self.distortion_map_u.y
        undistorted_map_u = InterpolatedImage(undistortion_map_u, x, y)
        undistorted_map_v = InterpolatedImage(undistortion_map_v, x, y)
        return undistorted_map_u, undistorted_map_v

    def verify(self, x_range, y_range, n_samples = 1000, verbose=True):
        min_x, max_x = x_range
        min_y, max_y = y_range
        xx, yy = np.meshgrid(np.linspace(min_x, max_x, n_samples), np.linspace(min_y, max_y, n_samples))
        p_undistorted = self.undistort_points(self.distort_points(stack((xx, yy))))
        xx_calc, yy_calc = unstack(p_undistorted)
        x_err = np.abs(xx_calc - xx)
        y_err = np.abs(yy_calc - yy)
        err = np.sqrt(x_err**2 + y_err**2)

        max_err = np.max(err)
        mean_err = np.mean(err)

        if verbose:
            print(f"Verifying InterpolatedDistortionMap:")
            print(f"    max error: {max_err:.3f} mean error: {mean_err:.3f}")

        return mean_err, max_err

if __name__ == '__main__':
    pass
