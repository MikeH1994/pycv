from __future__ import annotations
from dataclasses import dataclass
from typing import Union, Tuple
from numpy.typing import NDArray


@dataclass
class FittingParams:
    n_terms: int = 3
    x_range: Union[Tuple[float, float], None] = None
    p0: Union[NDArray, None] = None
    bounds: Union[NDArray, None] = None
    clip_x_range: bool = True
    clip_x_range_k: float = 2.5


@dataclass
class ESFSettings:
    """
    Settings for creating an ESF.

    Args:
        edge_detection_mode (str): The method used to determine the edge location.
            edge_detection_mode = fit_esf

            default:
        wflag (int): flag to indicate what window, if any, to apply when calculating the edge position using centroid.
            wflag = 0: Tukey window
            wflag = 1: Hamming window
            wflag = 2: No window used

            Default: 0
        alpha (float): Parameter for defining how wide the Tukey window is.
            Only used when edge_detection_mode == "centroid". Default: 1.0
        npol (int): The order of the polynomial used to define the location
            for the window when using edge_detection_mode == "centroid".
            Default: 5

    """
    esf_type: str = "binned"
    edge_detection_mode: str = "fit_esf"
    wflag: int = 0
    alpha: float = 1
    npol: int = 5
    edge_location_points: Union[Tuple[float], None] = None
    n_bins_per_pixel: int = 4
    bins_zero_centred: bool = True

