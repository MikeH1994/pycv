import numpy as np
from numpy.typing import NDArray
from typing import Union
import scipy.interpolate


class InterpolatedImage:
    def __init__(self, img: NDArray, x: np.ndarray = None, y: np.ndarray = None):
        self.img = img
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


    def __call__(self, x: Union[float, NDArray], y: Union[float, NDArray], return_as_int: bool = False) -> Union[int, float, NDArray]:
        return self.f(x, y, return_as_int=return_as_int)

    def f(self, x: Union[float, NDArray], y: Union[float, NDArray], return_as_int: bool = False) -> Union[int, float, NDArray]:
        """

        :param x:
        :param y:
        :param return_as_int:
        :return:
        """
        ret = self.interp_fn(y, x, grid=False)
        if return_as_int:
            if isinstance(ret, np.ndarray):
                ret = ret.astype(np.uint8)
            else:
                ret = int(ret)
        return ret

    def scale_image(self, scale_factor):
        self.img *= scale_factor
        self.interp_fn = self.create_interpolated_image()