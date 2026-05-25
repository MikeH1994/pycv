import math
import numpy as np


def fraction_of_square_in_circle(x, y, r, nx=2000, block_size=None):
    """
    Vectorized fraction of a unit square (side=1) centered at (x, y) inside
    a circle of radius r centered at (0,0).

    Parameters
    ----------
    x, y, r : array_like
        Broadcastable arrays of the same or compatible shape.
    nx : int
        Number of samples along x for midpoint integration (higher = more accurate).
    block_size : int or None
        If not None, process input in blocks to limit peak memory usage for huge arrays.

    Returns
    -------
    area : ndarray
        Array of the same broadcasted shape as (x, y, r), with values in [0, 1].
    """
    # Broadcast inputs to a common shape
    x = np.asarray(x)
    y = np.asarray(y)
    r = np.asarray(r)
    x, y, r = np.broadcast_arrays(x, y, r)
    out = np.empty_like(r, dtype=np.float64)

    # Precompute the offset grid for the unit square width
    # Midpoint samples across [-0.5, +0.5]
    offsets = (-0.5 + (np.arange(nx) + 0.5) / nx).astype(np.float64)  # shape (nx,)
    dx = 1.0 / nx

    # Helper: compute trivial masks
    absx = np.abs(x)
    absy = np.abs(y)
    dmin = np.hypot(np.maximum(absx - 0.5, 0.0), np.maximum(absy - 0.5, 0.0))
    dmax = np.hypot(absx + 0.5, absy + 0.5)
    none_mask = dmin >= r
    full_mask = r >= dmax

    # Optionally process in blocks to keep memory bounded
    it = np.ndindex(x.shape)
    if block_size is None:
        # Single block
        _compute_block(x, y, r, offsets, dx, out)
    else:
        # Chunk over the flattened view
        x_flat = x.ravel()
        y_flat = y.ravel()
        r_flat = r.ravel()
        out_flat = out.ravel()
        N = x_flat.size
        for start in range(0, N, block_size):
            end = min(start + block_size, N)
            _compute_block(x_flat[start:end], y_flat[start:end],
                           r_flat[start:end], offsets, dx, out_flat[start:end])
        out = out_flat.reshape(x.shape)

    # Apply trivial masks
    out = np.where(none_mask, 0.0, out)
    out = np.where(full_mask, 1.0, out)
    return out

def _compute_block(xb, yb, rb, offsets, dx, outb):
    """
    Compute a block of results with broadcasting across the sample axis.
    xb, yb, rb: shape (M,)
    offsets: shape (nx,)
    outb: shape (M,)
    """
    # X samples across each square: shape (M, nx)
    X = xb[:, None] + offsets[None, :]

    # Mask out columns where |X| >= r (no circle vertical extent there)
    valid = np.abs(X) < rb[:, None]

    # Yc = sqrt(r^2 - X^2) on valid positions; else 0
    Yc = np.zeros_like(X, dtype=np.float64)
    under = rb[:, None]**2 - X**2
    np.maximum(under, 0.0, out=under)  # clamp negatives to 0 for numerical safety
    Yc[valid] = np.sqrt(under[valid])

    # Square vertical bounds
    yT = yb[:, None] + 0.5
    yB = yb[:, None] - 0.5

    # Vertical overlap per sample
    top = np.minimum(yT, Yc)
    bot = np.maximum(yB, -Yc)
    L = np.clip(top - bot, 0.0, None)

    # Integrate along x
    outb[:] = dx * np.sum(L, axis=1)
