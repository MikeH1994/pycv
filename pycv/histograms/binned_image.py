import numpy as np


def create_2d_binned_data(vals, x, y, x0, x1, y0, y1, grid_size):
    bins_x = np.arange(x0, x1, grid_size, dtype=np.float32)
    bins_y = np.arange(y0, y1, grid_size, dtype=np.float32)

    binned_data = np.zeros((bins_y.shape[0], bins_x.shape[0]), dtype=np.float32)
    n_elems = np.zeros(binned_data.shape, dtype=np.float32)

    x_idx = np.digitize(x, bins_x) - 1
    y_idx = np.digitize(y, bins_y) - 1
    valid_indices = (x_idx >= 0) & (y_idx >= 0) & (x_idx < bins_x.shape[0]) & (y_idx < bins_y.shape[0])

    x_idx = x_idx[valid_indices]
    y_idx = y_idx[valid_indices]
    np.add.at(binned_data, (y_idx, x_idx), vals[valid_indices])
    np.add.at(n_elems, (y_idx, x_idx), 1)
    binned_data[n_elems > 0] /= n_elems[n_elems > 0]

    bins_x += grid_size / 2
    bins_y += grid_size / 2

    bins_xx, bins_yy = np.meshgrid(bins_x, bins_y)

    return binned_data, bins_xx, bins_yy, n_elems