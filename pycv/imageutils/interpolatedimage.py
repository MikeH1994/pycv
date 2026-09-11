import numpy as np
from numpy.typing import NDArray
from typing import Union
import scipy.interpolate


class InterpolatedImage:
    def __init__(self, img: NDArray, x: np.ndarray = None, y: np.ndarray = None, boundary_mode="extrapolate"):
        self.img = img
        self.boundary_mode = boundary_mode
        height, width = img.shape[:2]
        x = np.arange(width) if x is None else x
        y = np.arange(height) if y is None else y

        if len(x.shape) == 2 and len(y.shape) == 2:
            # in case xx and yy are passed
            assert(x.shape == y.shape)
            x = x[0]
            y = y[:, 0]

        assert(x.shape[0] == self.img.shape[1] and y.shape[0] == self.img.shape[0])

        self.x = x
        self.y = y
        self.interp_fn =self.create_interpolated_image()

    def create_interpolated_image(self):
        return scipy.interpolate.RectBivariateSpline(self.y, self.x, self.img)


    def __call__(self, x: Union[float, NDArray], y: Union[float, NDArray], return_as_int: bool = False,
                 boundary_mode = None) -> Union[int, float, NDArray]:
        return self.f(x, y, return_as_int=return_as_int, boundary_mode=boundary_mode)

    def f(self, x: Union[float, NDArray], y: Union[float, NDArray], return_as_int: bool = False,
          boundary_mode=None) -> Union[int, float, NDArray]:
        """

        :param x:
        :param y:
        :param return_as_int:
        :return:
        """
        assert isinstance(x, np.ndarray) == isinstance(y, np.ndarray)
        return_arr = isinstance(x, np.ndarray)
        x = np.array(x)
        y = np.array(y)
        boundary_mode = self.boundary_mode if boundary_mode is None else boundary_mode

        if boundary_mode == "reflect":
            x, y = self.coordinates_reflect(x, y)
        elif boundary_mode == "nearest":
            x, y = self.coordinates_nearest(x, y)
        elif boundary_mode == "extrapolate":
            pass
        else:
            raise Exception("Invalid boundary mode")

        ret = self.interp_fn(y, x, grid=False)
        if return_arr:
            if return_as_int:
                ret = ret.astype(np.uint8)
        else:
            if return_as_int:
                ret = int(ret)
        return ret

    def coordinates_reflect(self, x, y):
        x = self._reflect_coordinate(x, np.min(self.x), np.max(self.x))
        y = self._reflect_coordinate(y, np.min(self.y), np.max(self.y))
        return x, y

    def coordinates_nearest(self, x, y):
        x = np.clip(x, np.min(self.x), np.max(self.x))
        y = np.clip(y, np.min(self.y), np.max(self.y))
        return x, y

    def _reflect_coordinate(self, coord, min_val, max_val):
        coord = np.asarray(coord)

        length = max_val - min_val
        if length == 0:
            return np.full_like(coord, min_val)

        period = 2 * length
        c = (coord - min_val) % period
        reflected = np.where(c <= length, min_val + c, max_val - (c - length))

        return reflected

    def scale_image(self, scale_factor):
        self.img *= scale_factor
        self.interp_fn = self.create_interpolated_image()