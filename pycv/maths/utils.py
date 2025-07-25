import numpy as np
from scipy.interpolate import interp1d

def create_fitted_functions(x, y, order):
    """
    Fit an n-th order polynomial to x and y data, and return interp1d functions
    for forward (x -> y) and inverse (y -> x) interpolation.

    Parameters:
        x (array-like): Input x data.
        y (array-like): Input y data.
        order (int): Order of the polynomial to fit.

    Returns:
        f_xy (callable): Interpolator to get y from x.
        f_yx (callable): Interpolator to get x from y.
    """
    # Ensure numpy arrays and sort by x for interpolation stability
    x = np.asarray(x)
    y = np.asarray(y)

    # Fit polynomial
    coeffs = np.polyfit(x, y, order)
    poly = np.poly1d(coeffs)

    x_fitted = np.linspace(np.min(x), np.max(x), 1000)
    y_fitted = poly(x_fitted)


    # Ensure monotonicity for invertibility
    if not np.all(np.diff(y_fitted) > 0) and not np.all(np.diff(y_fitted) < 0):
        raise ValueError("Fitted polynomial is not monotonic. Inverse interpolation may not be reliable.")

    # Interpolators
    x_to_y_fn = interp1d(x_fitted, y_fitted, bounds_error=False, fill_value="extrapolate")
    y_to_x_fn = interp1d(y_fitted, x_fitted, bounds_error=False, fill_value="extrapolate")

    return x_to_y_fn, y_to_x_fn, coeffs