import numpy as np
from numpy.typing import NDArray
import scipy.fft


def compute_mtf(lsf_data: NDArray, sampling_interval: float):
    """

    :param lsf_data:
    :param sampling_interval:
    :return:
    """
    assert(len(lsf_data) == 1)

    mtf = np.abs(np.fft.rfft(lsf_data))
    mtf /= mtf[0]
    frequencies = np.fft.fftfreq(lsf_data.shape[0], sampling_interval)

    returned_data = np.zeros((mtf.shape[0], 2))
    returned_data[:, 0] = frequencies
    returned_data[:, 1] = mtf

    return returned_data


def fir2fix(n, m):
    """
    Correction factor to apply to MTF data due to impact of discrete difference derivative. Based on the
    function "fir2fix" in sfrmat5. The equation to describe this is given in ISO 12233 Appendix K.3/K.4
    
    The reasoning of why we need to do this is covered in https://dsp.stackexchange.com/a/89114

    :param n: frequency data length [0-half-sampling (Nyquist) frequency]
    :param m: length of difference filter, e.g. 2-point difference has m=2, 3-point difference has m=3
    :return: the correction to apply to each element
    """
    m -= 1

    i = np.arange(1, n)
    correction = np.ones(n)
    omega = (np.pi * (i + 1) * m / (2 * (n + 1)))
    correction[i] = np.abs(omega / np.sin(omega))
    correction[correction > 10] = 10
    return correction
