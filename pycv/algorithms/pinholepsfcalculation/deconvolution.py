import numpy as np

from scipy.sparse import coo_matrix, eye, vstack
from scipy.sparse.linalg import lsqr


def reconstruct_integrated_samples(x_meas, y_meas, L, dx=0.25, dy=0.25, pixel_size=1.0, regularization=1e-3, use_average=True):
    """
    Reconstruct a high-resolution image from measurements that are
    averages/integrals over square pixel footprints.

    Parameters
    ----------
    x_meas, y_meas : array_like
        Measurement positions.

    L : array_like
        Measured pixel intensities.

    dx, dy : float
        High-resolution reconstruction pixel size.

    pixel_size : float
        Width and height of detector footprint.

    regularization : float
        Tikhonov regularization strength.

    use_average : bool
        True:
            Measurement represents pixel average.
        False:
            Measurement represents integrated energy.

    Returns
    -------
    image : ndarray (ny, nx)

    x_grid : ndarray
        Grid centre x coordinates.

    y_grid : ndarray
        Grid centre y coordinates.

    result : dict
        Additional solver information.
    """

    x_meas = np.asarray(x_meas, dtype=float)
    y_meas = np.asarray(y_meas, dtype=float)
    L = np.asarray(L, dtype=float)

    xmin = np.floor(x_meas.min() - 1)
    xmax = np.ceil(x_meas.max() + 1)
    ymin = np.floor(y_meas.min() - 1)
    ymax = np.ceil(y_meas.max() + 1)

    nx = int(np.ceil((xmax - xmin) / dx))
    ny = int(np.ceil((ymax - ymin) / dy))

    rows = []
    cols = []
    vals = []

    half = pixel_size / 2.0

    for i, (xc, yc) in enumerate(zip(x_meas, y_meas)):

        x0 = xc - half
        x1 = xc + half

        y0 = yc - half
        y1 = yc + half

        ix0 = max(0, int(np.floor((x0 - xmin) / dx)))
        ix1 = min(nx - 1, int(np.floor((x1 - xmin) / dx)))

        iy0 = max(0, int(np.floor((y0 - ymin) / dy)))
        iy1 = min(ny - 1, int(np.floor((y1 - ymin) / dy)))

        for iy in range(iy0, iy1 + 1):

            py0 = ymin + iy * dy
            py1 = py0 + dy

            overlap_y = max(
                0.0,
                min(y1, py1) - max(y0, py0)
            )

            if overlap_y == 0:
                continue

            for ix in range(ix0, ix1 + 1):

                px0 = xmin + ix * dx
                px1 = px0 + dx

                overlap_x = max(
                    0.0,
                    min(x1, px1) - max(x0, px0)
                )

                if overlap_x == 0:
                    continue

                area = overlap_x * overlap_y

                rows.append(i)
                cols.append(iy * nx + ix)
                vals.append(area)

    A = coo_matrix(
        (vals, (rows, cols)),
        shape=(len(L), nx * ny)
    ).tocsr()

    if use_average:
        row_sums = np.asarray(A.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        A = A.multiply(1.0 / row_sums[:, None])

    if regularization > 0:
        A_aug = vstack([A, np.sqrt(regularization) * eye(nx * ny)])
        b_aug = np.concatenate([L,np.zeros(nx * ny)])
    else:
        A_aug = A
        b_aug = L

    solution = lsqr(A_aug, b_aug)

    u = solution[0]

    image = u.reshape(ny, nx)

    x_grid = xmin + (np.arange(nx) + 0.5) * dx
    y_grid = ymin + (np.arange(ny) + 0.5) * dy

    return image, x_grid, y_grid, {
        "iterations": solution[2],
        "residual_norm": solution[3],
        "condition_estimate": solution[6],
    }