from Tools.scripts.summarize_stats import emit_table
from scipy.interpolate import interp1d
import numpy as np
from typing import List, Union
from numpy.typing import NDArray
from .radiancemodel import RadianceModel


class OpticalPathway:
    def __init__(self, transmission_components: List[Union[NDArray, float]], wavelengths: NDArray, response_fn: NDArray):
        self.wavelengths = wavelengths
        self.transmission_components = transmission_components
        self.emissivity_components = [1 - t for t in self.transmission_components]
        self.response_fn = response_fn
        self.radiance_model = RadianceModel(wavelengths, response_fn = response_fn, max_t=2000.0)

    def compute(self, scene_temperature: float, component_temperatures: List[float]):
        l_tot_spectral = 0.0
        temperatures = [scene_temperature] + component_temperatures
        emissivity_components = [1.0] + self.emissivity_components
        transmission_components = self.transmission_components + [1.0]
        for i in range(len(emissivity_components)):
            temp = temperatures[i]
            em = emissivity_components[i]
            l_em = self.radiance_model.planck_function(temp, self.wavelengths)*em
            for t in transmission_components[i:]:
                l_em *= t
            l_tot_spectral += l_em
        l_tot = self.radiance_model.integrate_spectral_radiance(l_tot_spectral, self.wavelengths)
        brightness_temp = self.radiance_model.radiance_to_temperature_fn(l_tot)
        return brightness_temp
