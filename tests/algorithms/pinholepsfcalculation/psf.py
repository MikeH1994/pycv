import scipy.integrate
import numpy as np
from pycv.algorithms.pinholepsfcalculation import *
from tqdm.auto import tqdm

def create_psf_test_case(case_index, seed = 1234, n_terms = None, fix_position = None):
    rng = np.random.default_rng(seed=seed+case_index)
    n_terms: int = int (rng.integers(1, 5, 1).item()) if n_terms is None else n_terms
    fix_position: bool = bool(rng.integers(0, 2, 1)) if fix_position is None else fix_position
    k = rng.random((n_terms,)) + 0.1
    sigma = rng.random((n_terms,))*2 + 0.1
    dx= (-1 + 2*rng.random((n_terms,)))
    dy= (-1 + 2*rng.random((n_terms,)))

    params = []
    for i in range(n_terms):
        if fix_position:
            params += [k[i], sigma[i]]
        else:
            params += [k[i], sigma[i], dx[i], dy[i]]
    psf = PSF(np.array(params), fix_position)
    return psf

def integral_over_infinity_numeric(psf: PSF, n_samples=10000, width=100):
    subsamples = np.linspace(-width, width, n_samples)
    xx, yy = np.meshgrid(subsamples, subsamples)
    f = psf.f(xx, yy)
    fx = scipy.integrate.simpson(f, subsamples, axis=1)
    return scipy.integrate.simpson(fx, subsamples)

def integral_over_infinity_derivative_numeric(psf: PSF, eps=1e-6, n_samples=10000, width=100):
    calculated = np.zeros(4*psf.n_terms + 2)
    x0 = integral_over_infinity_numeric(psf, n_samples=n_samples, width=width)
    for j in range(2, calculated.shape[0]):
        p0 = np.copy(psf.params.flatten())
        p0[j - 2] += eps
        psf_2 = PSF(p0)
        x1 = integral_over_infinity_numeric(psf_2, n_samples=n_samples, width=width)
        calculated[j] = (x1 - x0) / eps
    return calculated

def integral_over_circle_numeric(psf: PSF, x, y, r, n_samples=5000):
    init_shape = x.shape
    x = x.reshape(-1)
    y.reshape(-1)
    dst = np.zeros_like(x)
    subsamples = np.linspace(-r, r, n_samples)
    xx, yy = np.meshgrid(subsamples, subsamples)
    mask = (xx * xx + yy * yy) <= r * r

    for i in range(dst.shape[0]):
        f = psf.f(xx + x[i], yy + y[i]) * mask
        fx = scipy.integrate.simpson(f, subsamples, axis=1)
        dst[i] = scipy.integrate.simpson(fx, subsamples)
    return dst.reshape(init_shape)

def integral_over_circle_derivative_numeric(psf: PSF, x, y, r, eps=1e-6, n_samples=3000):
    calculated = np.zeros((4*psf.n_terms + 2, x.shape[0]))
    x0 = integral_over_circle_numeric(psf, x, y, r, n_samples=n_samples)
    for j in range(calculated.shape[0]):
        if j == 0:
            x1 = integral_over_circle_numeric(psf, x, y, r+eps, n_samples=n_samples)
        elif j == 1:
            continue
        else:
            p0 = np.copy(psf.params.flatten())
            p0[j - 2] += eps
            psf_2 = PSF(p0)
            x1 = integral_over_circle_numeric(psf_2, x, y, r, n_samples=n_samples)
        calculated[j] = (x1 - x0) / eps
    return calculated


def test_integral_over_infinity(tol=1e-5):
    pbar = tqdm(range(10))
    pbar.set_description(f"TEST: Integral over infinity 0/{len(pbar)} - Calculating ...")
    for i in pbar:
        psf = create_psf_test_case(i)
        calculated = psf.integral_over_infinity()
        expected_numeric = integral_over_infinity_numeric(psf, n_samples=1000)
        expected_analytic = 0
        for i in range(psf.n_terms):
            k_i = psf.params[i, 0]
            sigma_i = psf.params[i, 1]
            expected_analytic += 2 * k_i * np.pi * sigma_i ** 2
        err  = np.abs(calculated-expected_numeric)
        max_err = np.max(err)
        pbar.set_description(f"TEST: Integral over infinity {i+1}/{len(pbar)} - Mean: {np.mean(calculated):.3E} Max Err: {max_err:.3E}")
        assert(np.all(err < tol))

def test_integral_over_infinity_derivative(tol=1e-5, eps=1e-6):
    pbar = tqdm(range(10))
    pbar.set_description(f"TEST: Integral over infinity derivative 0/{len(pbar)} - Calculating ...")
    for i in pbar:
        psf = create_psf_test_case(i)
        calculated = psf.integral_over_infinity_derivative()
        expected = integral_over_infinity_derivative_numeric(psf, eps, n_samples=5000)
        err  = np.abs(calculated-expected)
        max_err = np.max(err)
        pbar.set_description(f"TEST: Integral over infinity derivative {i+1}/{len(pbar)} - Mean: {np.mean(calculated):.3E} Max Err: {max_err:.3E}")
        assert(np.all(err < tol))

def test_integral_over_circle(tol=1e-5):
    x = np.linspace(-2, 2, 3)
    y = np.linspace(-2, 2, 3)
    x, y = np.meshgrid(x, y)
    x = x.flatten()
    y = y.flatten()

    pbar = tqdm(range(10))
    pbar.set_description(f"TEST: Integral over circle 0/{len(pbar)} - Calculating ...")
    for i in pbar:
        aperture_radius = np.linspace(0.5, 1.5, len(pbar))[i]
        psf = create_psf_test_case(i)
        calculated = psf.integral_over_circle(x, y, aperture_radius)
        expected_numeric = integral_over_circle_numeric(psf, x, y, aperture_radius, n_samples=7000)
        err  = np.abs(calculated-expected_numeric)
        max_err = np.max(err)
        pbar.set_description(f"TEST: Integral over circle {i+1}/{len(pbar)} - Mean: {np.mean(calculated):.3E} Max Err: {max_err:.3E}")
        assert(np.all(err < tol))


def test_integral_over_circle_derivative(tol=1e-5):
    x = np.array([0.0, 2.0])
    y = np.array([0.0, 2.0])

    pbar = tqdm(range(10))
    pbar.set_description(f"TEST: Integral over circle derivative 0/{len(pbar)} - Calculating ...")
    for i in pbar:
        aperture_radius = 0.8
        psf = create_psf_test_case(i, n_terms=1)
        calculated = psf.integral_over_circle_derivative(x, y, aperture_radius)
        expected_numeric = integral_over_circle_derivative_numeric(psf, x, y, aperture_radius, n_samples=8000)

        err  = np.abs(calculated-expected_numeric)
        max_err = np.max(err)
        pbar.set_description(f"TEST: Integral over circle derivative {i+1}/{len(pbar)} - Mean: {np.mean(calculated):.3E} Max Err: {max_err:.3E}")
        assert(np.all(err < tol))


if __name__ == '__main__':
    # test_integral_over_infinity()
    # test_integral_over_infinity_derivative()
    # test_integral_over_circle()
    test_integral_over_circle_derivative()