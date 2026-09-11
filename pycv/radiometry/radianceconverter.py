from typing import Union, Tuple
import numpy as np
from numpy.typing import NDArray
import scipy
import scipy.integrate
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from scipy.optimize import minimize
from scipy.optimize import brentq

class RadianceConverter:
    """
    The RadianceModel is a class containing some functionality to generate a interpolation table to go from
    radiance to temperature and vice-versa
    """
    _radiance_to_temperature_fn: interp1d
    _temperature_to_radiance_fn: interp1d

    def __init__(self, wavelengths: NDArray, min_t=100.0, max_t=1500.0,
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
        self.min_t = min_t
        self.max_t = max_t
        self.temperatures = np.linspace(min_t, max_t, n_temp_elems)
        self.response_fn = np.ones(wavelengths.shape[0]) if response_fn is None else response_fn
        self.radiances, self._radiance_to_temperature_fn, self._temperature_to_radiance_fn = self.build_table()

    def to_radiance(self, temp, in_celsius=False):
        k = 273.15 if in_celsius else 0.0
        return self._temperature_to_radiance_fn(temp + k)

    def to_temperature(self, radiance, in_celsius=False):
        k = 273.15 if in_celsius else 0.0
        return self._radiance_to_temperature_fn(radiance) - k


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

    def integrate_spectral_radiance(self, l_spectral, wavelengths: NDArray | None = None):
        wavelengths = self.wavelengths if wavelengths is None else wavelengths
        return trapezoid(l_spectral, wavelengths)

    def planck_function(self, temp: NDArray|float, wl: NDArray | None = None) -> NDArray:
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
        wl = self.wavelengths if wl is None else wl
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

    def calc_apparent_temp(self, emissivity: Union[NDArray, float], t_surface: float, t_bkg: float) -> float:
        """
        Calculate the apparent temperature of a surface, given its emissivity, surface temperature and the
            background temperature

        :param emissivity: the emissivity of the surface- as either an ndarray, for spectral emissivity, or as a float
            for total emissivity
        :type emissivity: np.ndarray | flaot
        :param t_surface: the surface temperature, in Kelvin
        :type t_surface: float
        :param t_bkg: the background temperature, in Kelvin
        :type t_bkg: float
        :return: the apparent temperature, in Kelvin
        :rtype: float
        """
        if isinstance(emissivity, float) or isinstance(emissivity, int):
            emissivity = np.full(self.wavelengths.shape, fill_value=emissivity)
        assert(isinstance(emissivity, np.ndarray) and emissivity.shape == self.wavelengths.shape)
        assert(np.all(emissivity >= 0) and np.all(emissivity <= 1.0))
        l_emitted = self.planck_function(t_surface, self.wavelengths) * emissivity
        l_reflected = self.planck_function(t_bkg, self.wavelengths) * (1.0 - emissivity)
        l_total = self.integrate_spectral_radiance(l_emitted + l_reflected, self.wavelengths)
        return self._radiance_to_temperature_fn(l_total)



    def calc_surface_temperature(self, emissivity: Union[NDArray, float], t_app: float, t_bkg: float):
        """
        Calculate the surface temperature, given the emissivity, apparent temperature and background temperature
        :param emissivity: the emissivity of the surface, either as an ndarray for spectral emissivity, or as float
                           for total emissivity
        :param t_app: the apparent temperature, in Kelvin
        :param t_bkg: the background temperature, in Kelvin
        :return:
        """

        if isinstance(emissivity, float) or isinstance(emissivity, int):
            emissivity = np.full(self.wavelengths.shape, fill_value=emissivity)
        assert(isinstance(emissivity, np.ndarray) and emissivity.shape == self.wavelengths.shape)
        assert(np.all(emissivity >= 0) and np.all(emissivity <= 1.0))
        reflectivity = 1.0 - emissivity
        l_bkg = self.planck_function(t_bkg, self.wavelengths) * reflectivity
        l_bkg = self.integrate_spectral_radiance(l_bkg, self.wavelengths)
        l_app = self._temperature_to_radiance_fn(t_app)
        l_em = l_app - l_bkg

        def f(t):
            l = emissivity * self.planck_function(t,self.wavelengths)
            l = self.integrate_spectral_radiance(l, self.wavelengths)
            return l - l_em

        return brentq(f, self.min_t, self.max_t)

    def calc_surface_temperature_from_radiance(self, emissivity: Union[NDArray, float], l_meas: float, l_bkg: float, in_celsius=False):
        """
        Calculate the surface temperature, given the emissivity, apparent temperature and background temperature
        :param emissivity: the emissivity of the surface, either as an ndarray for spectral emissivity, or as float
                           for total emissivity
        :param t_app: the apparent temperature, in Kelvin
        :param t_bkg: the background temperature, in Kelvin
        :return:
        """

        if isinstance(emissivity, float) or isinstance(emissivity, int):
            emissivity = np.full(self.wavelengths.shape, fill_value=emissivity)
        assert(isinstance(emissivity, np.ndarray) and emissivity.shape == self.wavelengths.shape)
        assert(np.all(emissivity >= 0) and np.all(emissivity <= 1.0))
        l_em = l_meas - l_bkg

        def f(t):
            l = emissivity * self.planck_function(t,self.wavelengths)
            l = self.integrate_spectral_radiance(l, self.wavelengths)
            return l - l_em

        if l_em <= 0:
            print("Background is greater than measured radiance- check parameters used to calculate this")
            return np.nan

        if np.sign(f((self.min_t))) == np.sign(f(self.max_t)):
            print("Unable to find crossover point- check parameters (e.g. reflectivity, bkg temp) used to calculate"
                  " l_bkg, or try again with more wavelengths enabled / larger temperature range")
            return np.nan

        k = 273.15 if in_celsius else 0.0
        return brentq(f, self.min_t, self.max_t) - k

