import numpy as np
from scipy.interpolate import griddata


def create_2d_binned_data(vals, x, y, x0, x1, y0, y1, grid_size, interpolate_between_missing_vals=False):
    bins_x = np.arange(x0, x1, grid_size, dtype=np.float32)
    bins_y = np.arange(y0, y1, grid_size, dtype=np.float32)

    binned_sum = np.zeros((bins_y.shape[0], bins_x.shape[0]), dtype=np.float32)
    binned_sum_sq = np.zeros_like(binned_sum)
    n_elems = np.zeros(binned_sum.shape, dtype=np.float32)

    x_idx = np.digitize(x, bins_x) - 1
    y_idx = np.digitize(y, bins_y) - 1
    valid_indices = (x_idx >= 0) & (y_idx >= 0) & (x_idx < bins_x.shape[0]) & (y_idx < bins_y.shape[0])

    x_idx = x_idx[valid_indices]
    y_idx = y_idx[valid_indices]
    np.add.at(binned_sum, (y_idx, x_idx), vals[valid_indices])
    np.add.at(binned_sum_sq, (y_idx, x_idx), vals[valid_indices]*vals[valid_indices])
    np.add.at(n_elems, (y_idx, x_idx), 1)

    bin_mean = np.zeros_like(binned_sum)
    nonempty = n_elems > 0
    bin_mean[nonempty] = binned_sum[nonempty] / n_elems[nonempty]


    valid_n = n_elems > 1
    var_sample = np.zeros_like(binned_sum)
    var_sample[valid_n] = (binned_sum_sq[valid_n] - n_elems[valid_n] * (bin_mean[valid_n] ** 2)) / (n_elems[valid_n] - 1)

    bin_std = np.zeros_like(binned_sum)
    bin_std[valid_n] = np.sqrt(var_sample[valid_n])


    bins_x += grid_size / 2
    bins_y += grid_size / 2
    bins_xx, bins_yy = np.meshgrid(bins_x, bins_y)

    if interpolate_between_missing_vals:
        binned_sum = interpolate_missing(bins_xx, bins_yy, binned_sum, n_elems)

    return {
        "bin_mean": bin_mean,
        "bin_stdev": bin_std,
        "bin_n": n_elems,
        "bin_xx": bins_xx,
        "bin_yy": bins_yy,
        "bin_x": bins_xx[0],
        "bin_y": bins_yy[:, 0],
    }


def interpolate_missing(xx, yy, vals, n_vals, method="linear"):
    """
    Interpolate missing values on a 2D grid.

    Parameters
    ----------
    xx : (M, N) ndarray
        2D grid of x coordinates.
    yy : (M, N) ndarray
        2D grid of y coordinates.
    vals : (M, N) ndarray
        Grid of measured/averaged values. Cells with n_vals == 0 should be 0 here.
    n_vals : (M, N) ndarray
        Number of samples used per cell. Cells with n_vals == 0 are treated as missing.
    method : {"linear", "cubic", "nearest"}, optional
        Interpolation method for interior gaps (default "linear").
        Note: "cubic" works only if there are enough points and can be slower.
    Returns
    -------
    filled : (M, N) ndarray
        Interpolated grid with missing cells filled. Outside the convex hull,
        nearest-neighbor is used to avoid NaNs.

    Notes
    -----
    - Missing cells are defined by n_vals == 0 (vals in those cells are ignored).
    - Inside the convex hull of valid points: `method` (default linear) is used.
    - Outside the convex hull: nearest-neighbor fallback fills remaining NaNs.
    """
    # Validate shapes
    if xx.shape != yy.shape or xx.shape != vals.shape or vals.shape != n_vals.shape:
        raise ValueError("xx, yy, vals, and n_vals must have the same shape.")

    # Mask of valid data
    valid_mask = n_vals > 0
    if not np.any(valid_mask):
        raise ValueError("No valid data points found (all n_vals == 0).")

    # Assemble known points and values
    points = np.column_stack([xx[valid_mask], yy[valid_mask]])
    values = vals[valid_mask]

    # Target points (grid)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    filled = np.empty_like(vals, dtype=float)

    # First pass: interior gaps via chosen method
    interp_vals = griddata(points, values, grid_points, method=method)
    filled_linear = interp_vals.reshape(vals.shape)

    # Identify remaining NaNs (likely outside convex hull)
    nan_mask = np.isnan(filled_linear)
    if np.any(nan_mask):
        # Second pass: nearest neighbor to fill those NaNs
        interp_nearest = griddata(points, values, grid_points, method="nearest")
        filled_nearest = interp_nearest.reshape(vals.shape)
        # Merge: linear/cubic where available, nearest elsewhere
        filled = np.where(nan_mask, filled_nearest, filled_linear)
    else:
        filled = filled_linear
    return filled