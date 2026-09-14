import numpy as np
import scipy
from scipy.stats import ncx2

class PSF:
    def __init__(self, params):
        assert(len(params.shape) == 1 and params.shape[0] %2 == 0)
        self.params = params.reshape(-1, 4)
        self.n_terms = self.params.shape[0]

    def f(self, x, y):
        total = np.zeros_like(x)
        for i in range(self.n_terms):
            params = self.params[i]
            k, sigma, dx, dy = params
            d2 = ((x + dx) ** 2 + (y + dy) ** 2)
            total += k * np.exp(-d2 / (2 * sigma ** 2))
        return total

    def normalise(self):
        pass

    def integral_over_circle(self, x: np.ndarray, y: np.ndarray, r: float):
        total = np.zeros_like(x)
        for i in range(self.n_terms):
            k, sigma, dx, dy = self.params[i]

            # CDF of noncentral chi-square with df=2
            d = np.sqrt((x+dx) ** 2 + (y+dy) ** 2)
            cdf = ncx2.cdf(x=(r / sigma) ** 2, nc=(d / sigma) ** 2, df=2)
            total += 2 * np.pi * k * sigma ** 2 * cdf
        return total

    def integral_over_infinity(self):
        total = 0
        for i in range(self.n_terms):
            k, sigma, _, _ = self.params[i]
            total += 2*k*np.pi*sigma**2
        return total

    def integral_over_circle_derivative(self, x: np.ndarray, y: np.ndarray, r: float):
        n_params = 4*self.n_terms + 2

        gradients = np.zeros((n_params, *x.shape))
        gradients[0] = 0.0 # aperture radius - will be sum of each term
        gradients[1] = 0.0 # brightness- not calculated here
        for i in range(self.n_terms):
            k, sigma, x0, y0 = self.params[i]

            d = np.sqrt((x + x0) ** 2 + (y + y0) ** 2)
            a = (r / sigma) ** 2
            lam = (d / sigma) ** 2
            F2 = ncx2.cdf(a, df=2, nc=lam)
            F4 = ncx2.cdf(a, df=4, nc=lam)
            F_lambda = 0.5 * (F4 - F2)
            f2 = ncx2.pdf(a, df=2, nc=lam)

            # partial derivative wrt radius, df / dr
            gradients[0] += 4 * np.pi * k * r * f2
            # partial derivative wrt k_i
            gradients[2 + 4*i] = 2 * np.pi * sigma ** 2 * F2
            # partial derivative wrt sigma_i
            gradients[2 + 4*i + 1] = 4 * np.pi * k * sigma * (F2 - a * f2 - lam * F_lambda)
            # x_i
            gradients[2 + 4*i + 2] = 4 * np.pi * k * (x + x0) * F_lambda
            # y_i
            gradients[2 + 4*i + 3] = 4 * np.pi * k * (y + y0) * F_lambda

        return gradients


    def integral_over_infinity_derivative(self):
        n_params = 4*self.n_terms + 2
        gradients = np.zeros(n_params, dtype=np.float32)
        # for a given term in the sum, its integral over infinity is
        # 2*k*pi*sigma^2
        # d/dk = 2*pi*sigma^2, d/dsigma = 4*k*pi*sigma; d/dxi = d/dyi = 0

        gradients[0] = 0.0 # aperture radius
        gradients[1] = 0.0 # brightness
        for i in range(self.n_terms):
            k, sigma, _, _ = self.params[i]
            gradients[2 + 4*i] = 2*np.pi*sigma**2         # df / d k
            gradients[2 + 4*i + 1] = 4*k*np.pi*sigma      # df / d sigma
            gradients[2 + 4*i + 2] = 0.0                  # df / d delta_x
            gradients[2 + 4*i + 3] = 0.0                  # df / d delta_y
        return gradients