from __future__ import annotations

from .radiancemodel import RadianceModel
from scipy.interpolate import interp1d
import numpy as np

class EmissivityCorrection:
    def __init__(self, radiance_model: RadianceModel):
        self.radiance_model = radiance_model

    def calculate_surface_temperature(self, t_app, emissivity: float | interp1d | np.array, t_bkg):
        if isinstance(emissivity, float) or isinstance(emissivity, int):
            l_app = self.radiance_model.temperature_to_radiance_fn(t_app)
            l_bkg = (1.0 - emissivity)*self.radiance_model.temperature_to_radiance_fn(t_bkg)
            l_em = l_app - l_bkg
            t_surface = self.radiance_model.radiance_to_temperature_fn(l_em / emissivity)
            return t_surface
        else:
            raise Exception("ArflNGFEON")

    def calculate_apparent_temperature(self, t_surface, emissivity: float | interp1d | np.array, t_bkg):
        if isinstance(emissivity, float) or isinstance(emissivity, int):
            l_em = emissivity*self.radiance_model.temperature_to_radiance_fn(t_surface)
            l_bkg = (1-emissivity)*self.radiance_model.temperature_to_radiance_fn(t_bkg)
            l_app =  l_em + l_bkg
            return self.radiance_model.radiance_to_temperature_fn(l_app)
        else:
            raise Exception("ArflNGFEON")