import numpy as np
import scipy
from scipy.stats import ncx2

class PSF:
    def __init__(self, params):
        self.params = params.reshape(-1, 4)
        self.n_terms = self.params.shape[0]

    def f(self, x, y):
        f = np.zeros_like(x)
        for i in range(self.n_terms):
            params = self.params[i]
            a, b, dx, dy = params
            f += self.eval(a, b, x + dx, y + dy)
        return f

    def integral_over_circle(self, x: np.ndarray, y: np.ndarray, r: float):
        I = np.zeros_like(x)
        for i in range(self.n_terms):
            a, sigma, dx, dy = self.params[i]
            I += self.eval_integral_over_circle(a, sigma, x + dx, y + dy, r)
        return I

    def integral_over_circle_numeric(self, x, y, r, n_samples=5000):
        init_shape = x.shape
        x = x.reshape(-1)
        y.reshape(-1)
        dst = np.zeros_like(x)
        subsamples = np.linspace(-r, r, n_samples)
        xx, yy = np.meshgrid(subsamples, subsamples)
        mask = (xx * xx + yy * yy) <= r * r

        for i in range(dst.shape[0]):
            f = self.f(xx + x[i], yy + y[i]) * mask
            fx = scipy.integrate.simpson(f, subsamples, axis=1)
            dst[i] = scipy.integrate.simpson(fx, subsamples)
        return dst.reshape(init_shape)

    def integral_over_infinity(self):
        I = 0
        for i in range(self.n_terms):
            a, sigma = self.params[i][:2]
            I += self.eval_integral_over_infinity(a, sigma)
        return I

    def integral_over_infinity_numeric(self, n_samples=10000, width=100):
        subsamples = np.linspace(-width, width, n_samples)
        xx, yy = np.meshgrid(subsamples, subsamples)
        f = self.f(xx, yy)
        fx = scipy.integrate.simpson(f, subsamples, axis=1)
        return scipy.integrate.simpson(fx, subsamples)

    def eval(self, a, sigma, dx, dy):
        return a * np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))

    def eval_integral_over_circle(self, a, sigma, dx, dy, r):
        """

        :param a: gaussian scale factor
        :param sigma: width of gaussian
        :param dx: distance from gaussian centre to the circle centre
        :param dy: distance from gaussian centre to the circle centre
        :param r: radius of circle
        :return:
        """
        d = np.sqrt(dx ** 2 + dy ** 2)
        lam = (d / sigma) ** 2
        x = (r / sigma) ** 2
        # CDF of noncentral chi-square with df=2
        cdf = ncx2.cdf(x, df=2, nc=lam)
        return 2 * np.pi * a * sigma ** 2 * cdf

    def eval_integral_over_infinity(self, a, sigma):
        return 2*a*np.pi*sigma**2