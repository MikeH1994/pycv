from numpy.typing import NDArray
from typing import Tuple
from .esf import ESF
from .fittedesf import GaussianESF
from .binnedesf import BinnedESF
from pycv.slantededge import get_edges_from_image
from pycv.slantededge.edge import Edge
from pycv.utils.settings import ESFSettings


def create_gaussian_esf(img: NDArray, **kwargs) -> Tuple[GaussianESF, Edge]:
    kwargs["edge_detection_mode"] = kwargs["edge_detection_mode"] if "edge_detection_mode" in kwargs else "fit_esf"
    edge = get_edges_from_image(img, **kwargs)[0]
    esf = GaussianESF(edge.esf_x, edge.esf_f, n_terms=4)
    return esf, edge


def create_binned_esf(img: NDArray, esf_settings: ESFSettings = ESFSettings()) -> Tuple[BinnedESF, Edge]:
    """

    :param img:
    :param esf_settings:
    :return:
    """
    edge = get_edges_from_image(img, esf_settings)[0]
    esf = BinnedESF(edge.esf_x, edge.esf_f, esf_settings)
    return esf, edge


def create_generic_esf(img: NDArray, esf_settings: ESFSettings = ESFSettings()) -> Tuple[ESF, Edge]:
    edge = get_edges_from_image(img, esf_settings.edge_detection_mode)[0]
    esf = ESF(edge.esf_x, edge.esf_f, esf_settings)
    return esf, edge
