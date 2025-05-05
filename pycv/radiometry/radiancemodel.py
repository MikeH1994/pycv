from typing import Union, Tuple
import numpy as np
from numpy.typing import NDArray
import scipy
import scipy.integrate
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid

class RadianceModel:
    """
    The RadianceModel is a class containing some functionality to generate a interpolation table to go from
    radiance to temperature and vice-versa
    """
    radiance_to_temperature_fn: interp1d
    temperature_to_radiance_fn: interp1d

    def __init__(self, wavelengths: NDArray, min_t=200.0, max_t=800.0,
                 response_fn: Union[NDArray, None] = None,
                 n_temp_elems: int = 50000):
        """
        :param wavelengths: the wavelengths, in nm, to sample over. Shape (n_wavelengths)
        :type wavelengths: np.ndarray
        :param min_t: the lower temperature bound, in Kelvin, to use for the temperature interpolation generation.
            Default: 200.0
        :type min_t: float, optional
        :param max_t: the upper temperature bound, in Kelvin, to use for the temperature interpolation table generation.
            Default: 800.0
        :type max_t: float
        :param response_fn: an array of shape (n_wavelengths) containing the spectral response at each wavelength. If
            None, no spectral response is used. Default: None
        :type response_fn: np.ndarray, optional
        :param n_temp_elems: the number of temperature elements to use in the interpolation table generation. Default:
            50000
        :type n_temp_elems: int, optional
        """

        self.wavelengths = wavelengths
        self.temperatures = np.linspace(min_t, max_t, n_temp_elems)
        self.response_fn = np.ones(wavelengths.shape[0]) if response_fn is None else response_fn

        self.radiances, self.radiance_to_temperature_fn, self.temperature_to_radiance_fn = self.build_table()

    def build_table(self) -> Tuple[NDArray, interp1d, interp1d]:
        """
        Build the radiance to temperature and temperature to radiance interpolation functions

        :return: a 3-tuple of (radiances, radiance_to_temperature_fn, temperature_to_radiance_fn).
            radiances is an array of shape (n_temp_elems), containingthe integrated radiance for each blackbody
            temperature. radiance_to_temperature_fn and temperature_to_radiance_fn are interpolation functions
            that allow the conversion of surface temperature to blackbody radiance and vice versa.
        :rtype: (np.ndarray, scipy.interpolate.interp1d, scipy.interpolate.interp1d)
        """
        radiances = np.zeros(self.temperatures.shape)
        for i in range(len(self.temperatures)):
            t = self.temperatures[i]
            l_spectral = self.planck_function(t, self.wavelengths)
            l_int = self.integrate_spectral_radiance(l_spectral, self.wavelengths)
            radiances[i] = l_int
        radiance_to_temperature_fn = interp1d(radiances, self.temperatures, kind='cubic', fill_value='extrapolate')
        temperature_to_radiance_fn = interp1d(self.temperatures, radiances, kind='cubic', fill_value='extrapolate')
        return radiances, radiance_to_temperature_fn, temperature_to_radiance_fn

    def integrate_spectral_radiance(self, l_spectral, wavelengths):
        return trapezoid(l_spectral, wavelengths)

    def planck_function(self, temp: NDArray|float, wl: NDArray) -> NDArray:
        """
        Compute the planck function, using the same form as the modified pbrt-v2.

        :param temp: the temperature, in Kelvin, to compute the Planck function for. This could be either a floating point
            value, in which case the returned array will be a numpy array of the shape (n_wavelengths), or this could be an
            arbitrary shaped array, in which case the returned array will be of the same shape but with an extra dimension
            corresponding to each wavelength- e.g. a 2D temperature image with the shape (512, 640) would return an array with
            the shape (512, 640, n_wavelengths)
        :type temp: float | np.ndarray
        :param wl: the sampled wavelengths, in nm. Shape: (n_wavelengths)
        :type wl: np.ndarray
        :param r: the spectral response at each wavelength. If not none, the output array will be multiplied by this before
            returning. Shape: (n_wavelegnths)
        :type r: np.ndarray, optional
        :return: a numpy array containing the blackbody spectral radiance at the given temperature(s)
        :rtype: np.ndarry
        """
        assert (len(wl.shape) == 1)
        c2 = 1.4388E7
        float_passed = False
        n_wavelengths = wl.shape[0]

        if isinstance(temp, float) or isinstance(temp, int):
            temp = np.array([temp])
            float_passed = True

        # reshape wavelengths so that it has the same number of dimensions as temp
        wl = wl.reshape(*[1 for _ in range(len(temp.shape) - len(wl.shape))], n_wavelengths)
        # calculate radiance for each element in temp
        radiance = np.zeros((*temp.shape, n_wavelengths))
        radiance[temp > 0] = 1E24 / (wl ** 5.0) / (np.exp(c2 / (wl * temp[temp > 0, None])) - 1.0)
        radiance *= self.response_fn

        if float_passed:
            radiance = radiance.reshape(n_wavelengths)

        return radiance
