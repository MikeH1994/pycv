import random
import numpy as np
from tqdm.auto import tqdm
from pycv import InterpolatedImage
from pycv.algorithms.pinholepsfcalculation import PSFSolver, PSF
from pycv.algorithms.pinholepsfcalculation.localisation import ApertureLocator
from pycv.radiometry import RadianceConverter
import matplotlib.pyplot as plt

class SimulatedImageGenerator:
    def __init__(self, image_size=(30, 30), n_samples=500,
                       netd_lower=0.0, netd_upper=0.4,
                       tbb_lower=50.0, tbb_upper=100.0,
                       tbkg_lower=10.0, tbkg_upper=40.0,
                       radius_lower=0.2, radius_upper=3.0,
                       bkg_gradient_lower = 0.1, bkg_gradient_upper = 0.5,
                       bkg_gradient_delta=0.05):
        self.image_size = image_size
        self.n_samples = n_samples
        self.rad_choices = [RadianceConverter(np.linspace(1000, 2000, 1000)),
                            RadianceConverter(np.linspace(2000, 3000, 1000)),
                            RadianceConverter(np.linspace(2500, 5700, 1000)),
                            RadianceConverter(np.linspace(7000, 14000, 1000))]
        self.n_terms_choices = [1, 3, 5]
        self.bkg_mode_choices = [0, 1, 2]
        self.netd_lower = netd_lower
        self.netd_upper = netd_upper
        self.tbb_lower = tbb_lower
        self.tbb_upper = tbb_upper
        self.tbkg_lower = tbkg_lower
        self.tbkg_upper = tbkg_upper
        self.radius_lower = radius_lower
        self.radius_upper = radius_upper
        self.bkg_gradient_lower = bkg_gradient_lower
        self.bkg_gradient_upper = bkg_gradient_upper
        self.bkg_gradient_delta = bkg_gradient_delta

    def get_aperture_loc(self, delta):
        w, h = self.image_size
        cx, cy = (w-1)/2.0, (h-1)/2.0
        x0 = cx + np.random.uniform(-delta, delta)
        y0 = cy + np.random.uniform(-delta, delta)
        return x0, y0

    def create_psf(self):
        n_terms = random.choice(self.n_terms_choices)
        k = np.empty((n_terms,), dtype=np.float32)
        sigma = np.empty((n_terms,), dtype=np.float32)
        k[0] = 1.0
        dim_1 = np.random.uniform(0.1, 1.0)
        dim_2 = np.random.uniform(0.1, 1.0)
        angle = np.random.uniform(0.0, 2 * np.pi)
        sigma[0] = np.random.uniform(0.3, 1.5)
        for i in range(1, n_terms, 2):
            k_i = np.random.uniform(0.1, 0.3)
            sigma_i = np.random.uniform(0.5, 2.0)
            k[i] = k[i + 1] = k_i
            sigma[i] = sigma[i + 1] = sigma_i
        return create_psf(k, sigma, dim_1, dim_2, angle)

    def create_background(self, rad):
        w, h = self.image_size
        t_bkg = np.random.uniform(self.tbkg_lower, self.tbkg_upper)
        gradient_centre = (np.random.uniform(-0.2 * w, 0.2 * w), np.random.uniform(-0.2 * h, 0.2 * h))
        gradient = np.random.uniform(0.1, 0.5)
        angle = np.random.uniform(0.0, 2 * np.pi)
        netd = np.random.uniform(0, 1.0)
        bkg_mode = random.choice(self.bkg_mode_choices)
        bkg = create_background(bkg_mode, rad, t_bkg, netd, gradient, gradient_centre, angle, self.image_size)
        return bkg

    def create_simulated_image(self, psf: PSF, bkg: InterpolatedImage, rad: RadianceConverter,
                               t_bb: float, ap_radius: float, aperture_loc, in_radiance=True):
        solver = PSFSolver(np.empty(self.image_size), np.empty(self.image_size), np.empty(self.image_size), None)
        solver.psf = psf
        solver.background = bkg
        solver.rad = rad

        w, h = self.image_size
        cx, cy = aperture_loc
        xx, yy = np.meshgrid(np.arange(w).astype(np.float32), np.arange(h).astype(np.float32))
        xx -= cx
        yy -= cy
        l_em = rad.to_radiance(t_bb, in_celsius=True)
        img = solver.calc_brightness(xx, yy, samples_per_pixel=self.n_samples, aperture_radius=ap_radius, l_em=l_em)
        img = rad.to_temperature(img, in_celsius=True)
        netd = np.random.uniform(self.netd_lower, self.netd_upper)
        noise = np.random.normal(scale=netd, size=(h, w))
        img += noise
        if in_radiance:
            img = rad.to_radiance(img, in_celsius=True)
        return img

    def create_simulated_images(self, n_images):
        images = []
        aperture_locations = []
        pbar = tqdm(range(n_images))
        for _ in pbar:
            # basic settings
            rad = random.choice(self.rad_choices)
            aperture_radius = np.random.uniform(self.radius_lower, self.radius_upper)
            aperture_location = self.get_aperture_loc(1.0)
            bkg_aperture_loc = self.get_aperture_loc(1.0)
            psf = self.create_psf()
            bkg = self.create_background(rad)
            t_bb = np.random.uniform(self.tbb_lower, self.tbb_upper)
            t_bkg_block = rad.to_temperature(bkg.max(), in_celsius=True) + np.random.uniform(-4.0, 4.0)
            img_scan = self.create_simulated_image(psf, bkg, rad, t_bb, aperture_radius, aperture_location)
            img_bkg = self.create_simulated_image(psf, bkg, rad, t_bkg_block, aperture_radius, bkg_aperture_loc)
            img = np.hstack([img_scan, img_bkg])
            images.append(img)
            aperture_locations.append(aperture_location)

        return images, aperture_locations


def create_psf(k, sigma, dim_1, dim_2, angle):
    assert(k.shape == sigma.shape)
    n_terms = k.shape[0]
    psf_params = np.zeros((n_terms, 4), dtype=np.float32)
    psf_params[:, 0] = k
    psf_params[:, 1] = sigma
    sin1 = np.sin(angle)
    sin2 = np.sin(angle+np.pi/2)
    cos1 = np.cos(angle)
    cos2 = np.cos(angle+np.pi/2)

    x_terms = np.array([0.0, dim_1*cos1, -dim_1*cos1, dim_2*cos2, -dim_2*cos2])
    y_terms = np.array([0.0, dim_1*sin1, -dim_1*sin1, dim_2*sin2, -dim_2*sin2])
    psf_params[:, 2] = x_terms[:n_terms]
    psf_params[:, 3] = y_terms[:n_terms]
    psf = PSF(psf_params)
    psf.params[:, 0] /= psf.integral_over_infinity()
    assert(np.allclose(psf.integral_over_infinity(), 1))
    return psf

def create_background(mode, rad: RadianceConverter, t_bkg, netd, gradient, gradient_origin, angle=0.0, shape=(61, 61)):
    w, h = shape
    cx, cy = (w-1)/2, (h-1)/2
    x0, y0 = gradient_origin
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32)-cx, np.arange(h, dtype=np.float32)-cy)

    if mode == 0: # flat background
        img = np.full(shape, fill_value=t_bkg, dtype=np.float32) + netd
    elif mode == 1: # radial gradient
        r = np.sqrt((xx-x0)**2 + (yy-y0)**2)
        img = t_bkg - r * gradient + netd
    elif mode == 2: # diagonal gradient
        g = (xx-x0) * np.cos(angle) + (yy-y0) * np.sin(angle)
        g = (g - g.min()) / g.max()
        img = t_bkg + g * gradient
    else:
        raise Exception(f"Invalid option")
    img = rad.to_radiance(img, in_celsius=True)
    return InterpolatedImage(img, xx, yy)