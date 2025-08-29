from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from typing import List


def create_histogram(data,  bin_centers: np.ndarray | None = None, num_bins: int = 20, normalise=True):
    if bin_centers is not None:
        # Calculate bin edges from user-defined centers
        bin_width = bin_centers[1] - bin_centers[0]
        bin_edges = np.concatenate(([bin_centers[0] - bin_width / 2],
                                    bin_centers + bin_width / 2))
        counts, _ = np.histogram(data, bins=bin_edges)
        bin_centers = bin_centers
    else:
        # Automatically compute histogram and bin centers
        counts, bin_edges = np.histogram(data, bins=num_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
    counts = counts.astype(np.float32)
    if normalise:
        counts /= counts.sum()
    return counts, bin_centers, bin_width

