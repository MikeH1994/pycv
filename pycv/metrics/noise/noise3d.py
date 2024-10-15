# from https://github.com/bigrobinson/Imaging-System-3D-Noise-/blob/master/noise3d.py

import os
from PIL import Image
import numpy as np

# Calculate 3D noise components
def get_3dnoise(data, plot=False):
    Nt, Nv, Nh = data.shape[0], data.shape[1], data.shape[2]

    # Normal temporal noise
    noise_t = np.mean(np.mean(np.std(data, axis=0, keepdims=True), \
                              axis=1, keepdims=True), axis=2, keepdims=True)

    # Mean Signal Level
    S = np.mean(np.mean(np.mean(data, axis=0, keepdims=True), axis=1, \
                        keepdims=True), axis=2, keepdims=True)
    data = data - S

    # Fixed row noise
    mu_v = np.mean(np.mean(data, axis=0, keepdims=True), axis=2, keepdims=True)
    sigma_v = np.std(mu_v, axis=1)
    data = data - mu_v

    # Fixed column noise
    mu_h = np.mean(np.mean(data, axis=0, keepdims=True), axis=1, keepdims=True)
    sigma_h = np.std(mu_h, axis=2)
    data = data - mu_h

    # Temporal frame noise (flicker)
    mu_t = np.mean(np.mean(data, axis=1, keepdims=True), axis=2, keepdims=True)
    sigma_t = np.std(mu_t, axis=0)
    data = data - mu_t

    # Temporal column noise (rain)
    mu_th = np.mean(data, axis=1, keepdims=True)
    sigma_th = np.std(np.reshape(mu_th, (1, Nt * Nh)))
    data = data - mu_th

    # Temporal row noise (streaking)
    mu_tv = np.mean(data, axis=2, keepdims=True)
    sigma_tv = np.std(np.reshape(mu_tv, (1, Nt * Nv)))
    data = data - mu_tv

    # Fixed spatially uncorrelated noise
    mu_vh = np.mean(data, axis=0, keepdims=True)
    sigma_vh = np.std(np.reshape(mu_vh, (1, Nv * Nh)))
    data = data - mu_vh

    # Time varying, spatially uncorrelated noise
    sigma_tvh = np.std(np.reshape(data, (1, Nt * Nv * Nh)))

    # Output noise components dictionary
    noise_components = {"noise_t": noise_t,
                        "signal": S,
                        "sigma_v": sigma_v,
                        "sigma_h": sigma_h,
                        "sigma_t": sigma_t,
                        "sigma_th": sigma_th,
                        "sigma_tv": sigma_tv,
                        "sigma_vh": sigma_vh,
                        "sigma_tvh": sigma_tvh}

    return noise_components


# Method to compose synthetic data cube with given 3D noise components. If you
# want to apply artificial noise to a stack of image data, simply set the
# "signal" component of the noise equal to that stack of image data. Also note
# that there is no need to set "noise_t" in this implementation.
def set_3dnoise(noise_components, t_res, v_res, h_res):
    data_cube = np.zeros((t_res, v_res, h_res))

    # Add mean signal level
    data_cube += noise_components["signal"]

    # Add fixed row noise.
    data_cube += noise_components["sigma_v"] * np.random.randn(1, v_res, 1) * \
                 np.ones((1, v_res, 1))

    # Add fixed column noise.
    data_cube += noise_components["sigma_h"] * np.random.randn(1, 1, h_res) * \
                 np.ones((1, 1, h_res))

    # Add fixed temporal frame ("flicker") noise
    data_cube += noise_components["sigma_t"] * np.random.randn(t_res, 1, 1) * \
                 np.ones((t_res, 1, 1))

    # Add temporal column noise.
    data_cube += noise_components["sigma_th"] * np.random.randn(t_res, 1, h_res) * \
                 np.ones((t_res, 1, h_res))

    # Add temporal row noise.
    data_cube += noise_components["sigma_tv"] * np.random.randn(t_res, v_res, 1) * \
                 np.ones((t_res, v_res, 1))

    # Add fixed, spatially uncorrelated noise
    data_cube += noise_components["sigma_vh"] * np.random.randn(1, v_res, h_res) * \
                 np.ones((1, v_res, h_res))

    # Add temporally varying, spatially uncorrelated noise
    data_cube += noise_components["sigma_tvh"] * \
                 np.random.randn(t_res, v_res, h_res) * \
                 np.ones((t_res, v_res, h_res))

    return data_cube