import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List
from .esf import ESF
from .fittedesf import GaussianESF
from .binnedesf import BinnedESF


def creage_gaussian_esf(img: NDArray, bins_per_pixel=4, zero_centered=True, **kwargs) -> Tuple[GaussianESF, NDArray]:
    data = ESF.create_data_from_slanted_edge(img, **kwargs)
    x_data, f_data = data["edge_profiles"][0]
    edge_points = data["edge_points"][0]
    esf = GaussianESF(x_data, f_data, bins_per_pixel=bins_per_pixel, zero_centered=zero_centered)
    return esf, edge_points


def create_binned_esf(img: NDArray, bins_per_pixel=4, zero_centered=True, **kwargs) -> Tuple[BinnedESF, NDArray]:
    """

    :param img:
    :param bins_per_pixel:
    :param zero_centered:
    :param kwargs:
    :return:
    """
    data = ESF.create_data_from_slanted_edge(img, **kwargs)
    x_data, f_data = data["edge_profiles"][0]
    edge_points = data["edge_points"][0]
    esf = BinnedESF(x_data, f_data, bins_per_pixel=bins_per_pixel, zero_centered=zero_centered)
    return esf, edge_points

