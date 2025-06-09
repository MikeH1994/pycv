import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List
from .esf import ESF
from .fittedesf import GaussianESF
from .binnedesf import BinnedESF
from ...metrics.targets.slantededge.core import get_edge_profile_from_image
from ...metrics.targets.slantededge.edge import Edge
from pycv.utils.settings import ESFSettings

def create_gaussian_esf(img: NDArray, **kwargs) -> Tuple[GaussianESF, Edge]:
    kwargs["edge_detection_mode"] = kwargs["edge_detection_mode"] if "edge_detection_mode" in kwargs else "fit_esf"
    data = get_edge_profile_from_image(img, **kwargs)
    x_data, f_data = data["edge_profiles"][0]
    edge_points = data["edges"][0]
    esf = GaussianESF(x_data, f_data, n_terms=4)
    return esf, edge_points


def create_binned_esf(img: NDArray, esf_settings: ESFSettings = ESFSettings()) -> Tuple[BinnedESF, Edge]:
    """

    :param img:
    :param esf_settings:
    :return:
    """
    data = get_edge_profile_from_image(img, esf_settings)
    x_data, f_data = data["edge_profiles"][0]
    edge_points = data["edges"][0]
    esf = BinnedESF(x_data, f_data, esf_settings)
    return esf, edge_points


def create_generic_esf(img: NDArray, esf_settings: ESFSettings = ESFSettings()) -> Tuple[ESF, Edge]:
    data = get_edge_profile_from_image(img, esf_settings.edge_detection_mode)
    x_data, f_data = data["edge_profiles"][0]
    edge_points = data["edges"][0]
    esf = ESF(x_data, f_data, esf_settings)
    return esf, edge_points
