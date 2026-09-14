import pickle
import time
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from tqdm import tqdm
from pycv import InterpolatedImage
from pycv.plt import intensity_scatterplot, set_colorbar
from pycv.radiometry import RadianceConverter
from .psf import PSF
from .utils import create_background, find_aperture, generate_pixel_data


class PSFSolver:
    def __init__(self, data_scan, data_bkg_closed, data_bkg_open, rad,
                 scan_mask=None, bkg_mask=None, roi_width = 15,
                 aperture_radius_init_guess=1.0, n_terms=3, title=""):
        """

        :param data_scan: the scan data - numpy array of shape (n, h, w)
        :param data_bkg_closed: series of stationary frames capturing the aperture, with source obstructed -
                                numpy array of shape (n, h, w)
        :param data_bkg_open: series of stationary frames capturing the aperture, with source unobstructed -
                              numpy array of shape (n, h, w) (note that aperture must be at the same position in the
                              image in both data_bkg_closed and data_bkg_open)
        :param rad: radiance model to convert radiance to temperature and vice versa
        :param init_search_location: where the search region for the aperture should be located, as a fraction of the
                                     image size - e.g. (0.5, 0.5) for an aperture located in the centre of the image
        :param roi_width: the size of the region of interest, centred on the aperture, used to calculated psf.
        :param aperture_radius_init_guess: the approximate aperture radius, in pixels. An accurate estimate improves
                                          the localisation stage. This is more accurately calculated during
                                          optimisation; if it is not known, it can be left as 1.0, and call run()
                                          with several iterations. The improved aperture radius estimation is then
                                          fed back into the localisation stage.
        :param n_terms: The number of terms to use in the series of Gaussians. More terms allows the model to fit
                        better but increases computation time.
        :param title:
        """
        # input data
        self.data_scan: np.ndarray = data_scan
        self.data_bkg_open: np.ndarray = data_bkg_open
        self.data_bkg_closed: np.ndarray = data_bkg_closed
        # masks for input data
        self.scan_mask: np.ndarray = np.ones(self.data_scan.shape[-2:], dtype=np.uint8) if scan_mask is None else scan_mask
        self.bkg_mask: np.ndarray = np.ones(self.data_bkg_closed.shape[-2:], dtype=np.uint8) if bkg_mask is None else bkg_mask

        self.rad: RadianceConverter = rad

        # parameters that are set
        self.roi_width = roi_width
        self.aperture_radius_init_guess: float = aperture_radius_init_guess
        self.n_terms = n_terms
        self.title = title

        # variables that will vary / be recomputed from run to run
        self.aperture_radius = aperture_radius_init_guess
        self.params = self._default_params(aperture_radius=self.aperture_radius)
        self.aperture_locations = None
        self.pixels_x = None
        self.pixels_y = None
        self.pixels_radiance = None
        self.aperture_locations = None
        self.background: InterpolatedImage | None = None
        self.l_em = None
        self.radiance_temperature = None
        self.pbar = None
        self.runs_performed = 0
        self.k = 1.0
        self.best_loss = np.nan
        self.psf = None

    def reinitialise(self, n_terms=None, aperture_radius_init_guess=None):
        """
        Resets the solver back to the state before any runs have been called.
        :param n_terms:
        :param aperture_radius_init_guess:
        :return:
        """
        if n_terms is not None:
            self.n_terms = n_terms
        if self.n_terms <= 0:
            raise Exception("n terms must be greater than 0")
        if aperture_radius_init_guess is not None:
            self.aperture_radius_init_guess = aperture_radius_init_guess

        self.params = self._default_params(aperture_radius=self.aperture_radius)
        self.aperture_radius = self.aperture_radius_init_guess
        self.aperture_locations = None
        self.pixels_x = None
        self.pixels_y = None
        self.pixels_radiance = None
        self.aperture_locations = None
        self.background = None
        self.l_em = None
        self.k = 1.0
        self.radiance_temperature = None
        self.pbar = None
        self.runs_performed = 0
        self.best_loss = np.nan

    def run(self, n_iterations=1, subsample_ratio=1, samples_per_pixel=100, update_background=True, update_scans=True):
        for i in range(n_iterations):
            self.runs_performed += 1
            print(f"Running iteration {self.runs_performed}")
            if self.background is None or update_background:
                self.create_background()
            if self.pixels_x is None or self.pixels_y is None or self.pixels_radiance is None or update_scans:
                self.find_aperture_locations()
                self.create_scan_data()

            if subsample_ratio > 0.0:
                self.params, self.l_em = self._solve(subsample_ratio=subsample_ratio, samples_per_pixel=samples_per_pixel)
                self.aperture_radius = self.params[0]
                self.radiance_temperature = self.rad.to_temperature(self.l_em, in_celsius=True)
                self.psf = PSF(self.params[2:])
        return{
            "psf": self.psf,
            "aperture_radius": self.aperture_radius,
            "radiance_temperature": self.radiance_temperature
        }

    def create_scan_data(self):
        dst = self._create_scan_data(self.data_scan, self.aperture_locations, self.roi_width, mask=self.scan_mask)
        self.pixels_x, self.pixels_y, self.pixels_radiance = dst
    def create_background(self):
        self.background = self._create_background(self.data_bkg_closed, self.data_bkg_open, self.aperture_radius,
                                mask=self.bkg_mask)

    def find_aperture_locations(self):
        self.aperture_locations = self._find_aperture_locations(self.data_scan, self.aperture_radius, self.scan_mask)

    def check_motion_blur(self, timestamps_in_ms, integration_time_in_ms, verbose=True):
        assert(self.aperture_locations is not None)
        timestamps_in_ms = np.array(timestamps_in_ms)
        aperture_locations = np.array(self.aperture_locations)
        dx = np.abs(np.diff(aperture_locations[:, 0]))
        dy = np.abs(np.diff(aperture_locations[:, 1]))
        step_size = np.sqrt(dx ** 2 + dy ** 2)
        timestep = np.diff(timestamps_in_ms) / 1000
        speed_px_per_sec = step_size / timestep
        movement_in_frame_px = speed_px_per_sec * (integration_time_in_ms/1000.0)
        ret = {
            "mean (px)": np.mean(movement_in_frame_px),
            "stdev (px)": np.std(movement_in_frame_px),
            "median (px)": np.median(movement_in_frame_px),
            "max (px)": np.max(movement_in_frame_px),
            "min (px)": np.min(movement_in_frame_px),
        }

        if verbose:
            print("Movement of aperture in frame:")
            for name, val in ret.items():
                print(f"    {name}: {val:.4f}")


    def display_residual_error(self, in_temperature=True, subsample_ratio=1.0, samples_per_pixel=100, title="", show=True, point_size=10):

        x_full = self.pixels_x.reshape(-1)
        y_full = self.pixels_y.reshape(-1)
        l_full = self.pixels_radiance.reshape(-1)

        np.random.seed(1234)
        subsample_indices = np.random.choice(np.arange(x_full.shape[0], dtype=np.int32),
                                             size=int(x_full.shape[0] * subsample_ratio))
        x = x_full[subsample_indices]
        y = y_full[subsample_indices]
        l = l_full[subsample_indices]

        l_calc = self.calc_brightness(x, y, samples_per_pixel)
        if in_temperature:
            err = np.abs(self.rad.to_temperature(l_calc, in_celsius=True) - self.rad.to_temperature(l, in_celsius=True))
        else:
            err = np.abs(l - l_calc)

        intensity_scatterplot(x, y, err, point_size=point_size, show=False)
        colorbar_label = "Residual error (au)" if not in_temperature else r"Residual error ($^\circ C$)"
        set_colorbar(label=colorbar_label)

        if show:
            plt.show()

    def set_3x3_parameters(self, w=0.3):
        self.reinitialise(n_terms=9)
        self.params = self._3x3_grid_parameter(w=w)

    def _3x3_grid_parameter(self, w=0.3, sigma_0 = 0.5, sigma_i=1.0, k_0=1.0, k_i=0.1):
        xi, yi = np.meshgrid([0.0, -w, w], [0.0, -w, w])

        xi = xi.flatten()
        yi = yi.flatten()
        params = [self.aperture_radius, 1]

        for i in range(xi.shape[0]):
            sigma = sigma_0 if i ==0 else sigma_i
            k = k_0 if i == 0 else k_i
            x = xi[i]
            y = yi[i]
            params += [k, sigma, x, y]

        return np.array(params)

    def _normalise_params(self):
        sigma = sigma_0 * k_sigma ** np.arange(self.n_terms)
        amplitude = w / (2 * np.pi * sigma ** 2)


    def display_line_plot(self, samples_per_pixel=100):
        raise Exception("TODO")

    def display_background(self, roi_width, show=True, in_temperature=True):
        x = np.arange(-roi_width, roi_width+1)
        y = np.arange(-roi_width, roi_width+1)
        xx, yy = np.meshgrid(x, y)
        img = self.background(xx, yy)
        if in_temperature:
            img = self.rad.to_temperature(img, in_celsius=True)
        plt.imshow(img)
        if show:
            plt.show()

    def calc_brightness(self, x, y, samples_per_pixel, aperture_radius=None, l_em=None, bkg=None, psf=None):
        init_shape = x.shape
        x = x.reshape(-1)
        y = y.reshape(-1)

        l_em = self.l_em if l_em is None else l_em
        aperture_radius = self.aperture_radius if aperture_radius is None else aperture_radius
        bkg = self.background if bkg is None else bkg
        psf = self.psf if psf is None else psf

        x_samples, y_samples = self._create_subsamples(x, y, samples_per_pixel)
        l_bkg = bkg(x_samples, y_samples)

        brightness_samples, _ = self._calc_brightness(x_samples, y_samples, psf, aperture_radius, l_em, l_bkg)
        l = np.mean(brightness_samples, axis=-1)
        return l.reshape(init_shape)

    def save(self, fpath):
        self.pbar = None
        with open(fpath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(fpath) -> PSFSolver:
        with open(fpath, "rb") as f:
            return pickle.load(f)

    def _find_aperture_locations(self, data: np.ndarray, aperture_radius_px, mask=None):
        assert(len(data.shape) == 3)
        aperture_locations = []
        pbar = tqdm(range(data.shape[0]))
        pbar.set_description("    Computing aperture locations... ")
        for i in pbar:
            centre = find_aperture(data[i], self.background, aperture_radius_px, self.rad, apply_blur=False, mask=mask)
            aperture_locations.append(centre)
        pbar.close()
        return aperture_locations



    def _create_scan_data(self, data: np.ndarray, aperture_locations, roi_width, mask=None):
        pixels_x, pixels_y, pixels_radiance = generate_pixel_data(data, aperture_locations, self.rad, roi_width, mask=mask)
        return pixels_x, pixels_y, pixels_radiance

    def _create_background(self, bkg_closed, bkg_open, aperture_radius: float,
                           mask=None, n_iterations=5, interpolate_over_aperture=True):

        background_open = np.mean(bkg_open.reshape(-1, *bkg_open.shape[-2:]), axis=0)
        background_closed = np.mean(bkg_closed.reshape(-1, *bkg_closed.shape[-2:]), axis=0)
        background = None
        pbar = tqdm(range(n_iterations))
        pbar.set_description("    Computing background... ")
        for _ in pbar:
            bkg_centre = find_aperture(background_open, background, aperture_radius, radiance_model=self.rad, mask=mask)
            background = create_background(background_closed, bkg_centre, self.rad, self.aperture_radius,
                                           interpolate_over_aperture=interpolate_over_aperture)
        pbar.close()
        time.sleep(0.01)
        return background



    def _default_params(self, aperture_radius=1.0, k_sigma=1.5, k_w=0.75, sigma_0 = 0.7):
        if aperture_radius <= 0:
            aperture_radius = 0.1

        params = [aperture_radius, 1]

        x = [0.0] * self.n_terms
        y = [0.0] * self.n_terms

        # create a set of gaussian widths and amplitudes that get progressively smaller and wider,
        # and integrate to 1 for energy conservation (reduces total number of iterations required)
        w = k_w ** np.arange(self.n_terms) / np.sum(k_w ** np.arange(self.n_terms))
        sigma = sigma_0 * k_sigma ** np.arange(self.n_terms)
        amplitude = w / (2 * np.pi * sigma ** 2)

        for i in range(self.n_terms):
            params += [amplitude[i], sigma[i], x[i], y[i]]
        return np.array(params)

    def _create_parameter_mask(self, fix_sigma=False, fix_amplitude=False, fix_position=False):
        param_mask = [True for _ in self.params]
        for i in range(2, len(self.params), 4):
            if fix_amplitude:
                param_mask[i] = False
            if fix_sigma:
                param_mask[i+1] = False
            if fix_position:
                param_mask[i+2] = False
                param_mask[i+3] = False
        return param_mask

    def _calc_brightness(self, x, y, psf: PSF, aperture_radius, l_em, l_bkg):
        # The brightness seen on a point can be broken into 2 parts - the integral inside the aperture (D_ap)
        # and the integral outside the aperture, equal to the integral over infinity - integral over aperture
        # f = D_ap * l_em + (D_inf - D_ap) * l_bkg
        # where D_ap = integral_over_circle() and D_inf = integral_over_infinity()
        D_ap = psf.integral_over_circle(x, y, aperture_radius)
        D_inf = psf.integral_over_infinity()
        brightness =  D_ap * l_em + (D_inf - D_ap) * l_bkg

        n_params = 4*psf.n_terms + 2
        grad_D_ap = psf.integral_over_circle_derivative(x, y, aperture_radius)
        # reshape grad_D_inf after computing so it is compatible with the shape of grad_D_ap
        grad_D_inf = psf.integral_over_infinity_derivative().reshape((n_params,) + (1,) * x.ndim)

        # the gradients of all parameters (except for L_em) can be calculated as
        partial_derivatives = grad_D_ap * l_em + (grad_D_inf - grad_D_ap) * l_bkg
        # df / d l_em  is just equal to D_ap
        partial_derivatives[1] = D_ap
        return brightness, partial_derivatives


    def _loss_fn(self, X, meas, x_samples, y_samples, free_params: List[bool] | None = None):
        # if we want to freeze certain parameters, we use a mask to define
        # what is fixed.
        if free_params is None:
            free_params = [True for _ in self.params]
        params = np.copy(self.params)
        params[free_params] = X

        ap_radius, l_em, psf_params = params[0], params[1], params[2:]
        l_bkg = self.background(x_samples, y_samples)
        psf = PSF(psf_params)
        brightness_samples, dfdtheta = self._calc_brightness(x_samples, y_samples, psf, ap_radius, l_em, l_bkg)
        calc = np.mean(brightness_samples, axis=-1)
        mse = np.mean((meas - calc) ** 2)

        t_meas = self.rad.to_temperature(self.k*meas, in_celsius=True)
        t_calc = self.rad.to_temperature(self.k*calc, in_celsius=True)
        mae_temp = np.mean(np.abs(t_meas - t_calc))
        max_err_temp = np.max(np.abs(t_meas - t_calc))

        # average dfdtheta to n_parans, n
        dfdtheta = np.mean(dfdtheta, axis=-1)
        jacobian = np.mean((calc - meas) * dfdtheta, axis=-1)
        # only return the jacobian for free parameters
        jacobian = jacobian[free_params]

        if self.pbar is not None:
            if mae_temp < self.best_loss:
                self.best_loss = mae_temp
            self.pbar.update(1)
            self.pbar.set_description(f"    Solving ({self.title}{x_samples.shape[0]} points / {x_samples.shape[1]} subsamples) "
                                      f"MAE = {mae_temp:.3f}°C MAX = {max_err_temp:.3f}°C")
        return mse, jacobian

    def _get_subsample_pattern(self, n_subsamples, rng_seed = 1234):
        if n_subsamples == 0:
            return np.zeros((2, ), dtype=np.float32)

        rng = np.random.default_rng(rng_seed) if rng_seed is not None else None
        return qmc.Halton(d=2, scramble=True, rng=rng).random(n_subsamples) - 0.5

    def _create_subsamples(self, x, y, n_subsamples):
        subsamples = self._get_subsample_pattern(n_subsamples)
        x_subsamples = subsamples[:, 0].reshape((1,) * x.ndim + (n_subsamples,))
        y_subsamples = subsamples[:, 1].reshape((1,) * x.ndim + (n_subsamples,))
        x_samples = np.zeros((*x.shape, n_subsamples)) + x.reshape(*x.shape, 1) + x_subsamples
        y_samples = np.zeros((*y.shape, n_subsamples)) + y.reshape(*y.shape, 1) + y_subsamples
        return x_samples, y_samples

    def _constraint_fn(self, params):
        psf = PSF(params[2:])
        return psf.integral_over_infinity() - 1.0

    def _constraint_jac(self, params):
        psf = PSF(params[2:])
        return psf.integral_over_infinity_derivative()

    def _solve(self, show_progress_bar: bool=True, samples_per_pixel: int=100, subsample_ratio=1.0,
               fix_sigma=False, fix_amplitude=False, fix_position=False):
        params = self.params
        param_mask = self._create_parameter_mask(fix_sigma=fix_sigma, fix_amplitude=fix_amplitude, fix_position=fix_position)
        x_full = self.pixels_x.reshape(-1)
        y_full = self.pixels_y.reshape(-1)
        tgt_points_full = self.pixels_radiance.reshape(-1)

        np.random.seed(1234)
        subsample_indices = np.random.choice(np.arange(x_full.shape[0], dtype=np.int32), size=int(x_full.shape[0]*subsample_ratio))
        x = x_full[subsample_indices]
        y = y_full[subsample_indices]
        tgt_points = tgt_points_full[subsample_indices]

        x_samples, y_samples = self._create_subsamples(x, y, samples_per_pixel)
        options = {"maxiter": 250}

        self.k = np.max(tgt_points).item()
        tgt_points /= self.k
        self.background.scale_image(1.0/self.k)
        self.pbar = tqdm(disable=not show_progress_bar)
        self.pbar.set_description(f"    Solving ({self.title} {x.shape[0]} points / {samples_per_pixel} subsamples)  MAE = N/A MAX = N/A")
        self.best_loss = np.inf
        result = minimize(self._loss_fn, params, args=(tgt_points, x_samples, y_samples),
                          options=options, jac=True, method='trust-constr',
                          constraints={'type': 'eq', 'fun': self._constraint_fn, 'jac': self._constraint_jac})
        params = result.x
        self.pbar.close()
        self.background.scale_image(self.k)

        aperture_brightness = params[1]
        aperture_brightness *= self.k
        return params, aperture_brightness
