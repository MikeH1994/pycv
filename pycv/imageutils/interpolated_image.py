import numpy as np
from numpy.typing import NDArray
from typing import Union
import scipy.interpolate


class InterpolatedImage:
    def __init__(self, img: NDArray):
        self.img = img
        height, width = img.shape[:2]
        x = np.arange(width)
        y = np.arange(height)
        self.interp_fn = scipy.interpolate.RectBivariateSpline(y, x, self.img)

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
