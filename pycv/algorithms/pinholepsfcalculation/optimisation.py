import numpy as np
import pycv
from pycv import InterpolatedImage
from pycv.plt.plots import intensity_scatterplot
from pycv.algorithms.pinholepsfcalculation.psf import PSF
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.stats import qmc


def default_params(n_terms=1, fix_position=False):
    """x = np.linspace(-1.5, 1.5, int(np.sqrt(n_terms)))
    y = np.linspace(-1.5, 1.5, int(np.sqrt(n_terms)))
    xx, yy = np.meshgrid(x, y)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)"""
    params = [1, 0.5]
    x = [0.0] * n_terms
    y = [0.0] * n_terms
    amplitudes = [1.0, 0.2, 0.1, 0.05]
    widths = [0.5, 1.0, 2.0, 4.0]
    for i in range(n_terms):
        if fix_position:
            params += [amplitudes[i], widths[i]]
        else:
            params += [amplitudes[i], widths[i], x[i], y[i]]
    return np.array(params)

def default_bounds(n_terms=1, fix_position=False):
    bounds = [(-np.inf, np.inf), (-np.inf, np.inf)]
    for i in range(n_terms):
        if fix_position:
            bounds += [(1e-6, np.inf), (1e-6, np.inf)]
        else:
            bounds+= [(1e-6, np.inf), (1e-6, np.inf), (-np.inf, np.inf), (-np.inf, np.inf)]
    return bounds

def calculate_brightness(x, y, psf: PSF, aperture_radius, l_em, l_bkg):
    # The brightness seen on a point can be broken into 2 parts - the integral inside the aperture (D_ap)
    # and the integral outside the aperture, equal to the integral over infinity - integral over aperture
    D_ap = psf.integral_over_circle(x, y, aperture_radius)
    D_inf = psf.integral_over_infinity()

    return D_ap * l_em + (D_inf - D_ap) * l_bkg


def loss_fn_old(X, measured_brightness, xx, yy, background: InterpolatedImage, pbar=None):
    aperture_radius, aperture_brightness, psf_params = X[0], X[1], X[2:]
    psf = PSF(psf_params)
    calculated_brightness = calculate_brightness(xx, yy, psf, aperture_radius, aperture_brightness, background)

    err = np.mean(np.abs(measured_brightness - calculated_brightness))
    if pbar is not None:
        pbar.update(1)
        pbar.set_description(f"    loss: {err:.4E}")
    return err

def calculate_psf_old(object_brightness: InterpolatedImage, background:InterpolatedImage,
                      xlim=(-2,2), ylim=(-2,2), n_samples_rough = 100, n_samples=1000, show_progress_bar=True, n_terms=3):
    params = default_params(n_terms=n_terms)
    bounds = default_bounds(n_terms=n_terms)
    for n_s in [n_samples_rough, n_samples]:
        x = np.linspace(xlim[0], xlim[1], n_s)
        y = np.linspace(ylim[0], ylim[1], n_s)
        xx, yy = np.meshgrid(x, y)

        target = object_brightness(xx, yy)
        k = np.max(target)
        target /= k
        background.scale_image(1.0/k)
        pbar = tqdm(disable=not show_progress_bar)
        params = minimize(loss_fn_old, params, args=(target, xx, yy, background, pbar), bounds=bounds, method='L-BFGS-B').x
        pbar.close()
        background.scale_image(k)
    aperture_radius, aperture_brightness, psf_params = params[0], params[1], params[2:]
    aperture_brightness *= k
    psf = PSF(psf_params)

    return psf, aperture_radius, aperture_brightness

def loss_fn(X, meas, x_samples, y_samples, background: InterpolatedImage, fix_position=False, pbar=None):
    ap_radius, l_em, psf_params = X[0], X[1], X[2:]
    l_bkg = background(x_samples, y_samples)
    psf = PSF(psf_params, fix_position=fix_position)
    calc = np.mean(calculate_brightness(x_samples, y_samples, psf, ap_radius, l_em, l_bkg), axis=-1)
    mse = np.mean((meas - calc) ** 2)
    mae = np.mean(np.abs((meas - calc)))

    # f = D_ap * l_em + (D_inf - D_ap) * l_bkg
    # where D_ap = integral_over_circle() and D_inf = integral_over_infinity()
    # jacobian = 1/n*(f-meas)*df/d\theta_i, where \theta_i is the i-th parameter
    # (note that as x_samples is of the shape (n, n_samples) we need to average in the n_samples axis

    # gradients in D_ap have shape (n_params, n_points, n_subsamples)
    grad_D_ap = psf.integral_over_circle_derivative(x_samples, y_samples, ap_radius)
    # gradients in D_inf have shape (n_params, 1)
    grad_D_inf = psf.integral_over_infinity_derivative().reshape(-1, 1, 1)
    # the gradients of all parameters (except for L_em) can be calculated as
    dfdtheta = np.mean((grad_D_ap * l_em + (grad_D_inf - grad_D_ap)*l_bkg), axis=-1)
    dfdtheta[1] = np.mean(psf.integral_over_circle(x_samples, y_samples, ap_radius), axis=-1)
    jacobian = np.mean((calc-meas)*dfdtheta, axis=-1)

    if pbar is not None:
        if mae < pbar.best_loss:
            pbar.best_loss = mae
        pbar.update(1)
        pbar.set_description(f"    {pbar.title} best MAE = {pbar.best_loss:.4E}, n samples = {x_samples.shape[0]}, n subsamples = {x_samples.shape[1]} - ")
    return mse, jacobian

def calculate_psf(tgt_points, x, y, background:InterpolatedImage, show_progress_bar=True, n_terms=3, n_subsamples=100,
                  n_subsamples_coarse=50, rng_seed=12345, coarse_subsample_ratio=1.0, full_subsample_ratio=1.0,
                  fix_position=False):
    psf_params = default_params(n_terms=n_terms, fix_position=fix_position)
    bounds = default_bounds(n_terms=n_terms, fix_position=fix_position)
    x_full = x.reshape(-1, 1)
    y_full = y.reshape(-1, 1)
    tgt_points_full = tgt_points.reshape(-1)

    assert(full_subsample_ratio is not None or coarse_subsample_ratio is not None)
    # and 0.0 < full_subsample_ratio <= 1.0
    k = 1.0
    for run_index, subsample_ratio in enumerate([coarse_subsample_ratio, full_subsample_ratio]):
        if subsample_ratio is None:
            continue
        n_s = n_subsamples_coarse if run_index == 0 else n_subsamples
        rng = np.random.default_rng(rng_seed) if rng_seed is not None else None
        subsamples = qmc.Halton(d=2, scramble=True, rng=rng).random(n_s)

        subsample_indices = np.random.choice(np.arange(x_full.shape[0], dtype=np.int32), size=int(x_full.shape[0]*subsample_ratio))
        x = x_full[subsample_indices]
        y = y_full[subsample_indices]
        tgt_points = tgt_points_full[subsample_indices]

        intensity_scatterplot(x, y, tgt_points)

        x_samples = np.zeros((x.shape[0], n_s)) + x + subsamples[:, 0]
        y_samples = np.zeros((y.shape[0], n_s)) + y + subsamples[:, 1]

        options = {"maxiter": 250}

        k = np.max(tgt_points).item()
        tgt_points /= k
        background.scale_image(1.0/k)
        pbar = tqdm(disable=not show_progress_bar)
        pbar.title = ["Coarse fit:", "Full fit:  "][run_index]
        pbar.best_loss = np.inf
        result = minimize(loss_fn, psf_params, args=(tgt_points, x_samples, y_samples, background, fix_position, pbar),
                              options=options, method="BFGS", jac=True)
        psf_params = result.x
        pbar.close()
        background.scale_image(k)

    aperture_radius, aperture_brightness, psf_params = psf_params[0], psf_params[1], psf_params[2:]
    aperture_brightness *= k
    psf = PSF(psf_params, fix_position=fix_position)

    return psf, aperture_radius, aperture_brightness