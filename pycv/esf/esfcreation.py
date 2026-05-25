from numpy.typing import NDArray
from typing import Tuple
from .esf import ESF
from .fittedesf import GaussianESF
from .binnedesf import BinnedESF
from pycv.edge import Edge
from pycv.utils.settings import ESFSettings


def create_gaussian_esf(img: NDArray) -> Tuple[GaussianESF, Edge]:
    edge = Edge(img)
    x, f = edge.parse_edge(img)
    esf = GaussianESF(x, f, n_terms=4)
    return esf, edge


def create_binned_esf(img: NDArray, esf_settings: ESFSettings = ESFSettings()) -> Tuple[BinnedESF, Edge]:
    """

    :param img:
    :param esf_settings:
    :return:
    """
    edge = Edge(img)
    x, f = edge.parse_edge(img)
    esf = BinnedESF(x, f, esf_settings)
    return esf, edge


def create_generic_esf(img: NDArray, esf_settings: ESFSettings = ESFSettings()) -> Tuple[ESF, Edge]:
    edge = get_edges_from_image(img, esf_settings.edge_detection_mode)[0]
    esf = ESF(edge.esf_x, edge.esf_f, esf_settings)
    return esf, edge
