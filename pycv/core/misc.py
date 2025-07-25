from __future__ import annotations
import numpy as np
from typing import Tuple
from numpy.typing import NDArray
import os

def stack_coords(arrays: Tuple[NDArray]) -> NDArray:
    assert(len(arrays) == 2 or len(arrays) == 3)
    assert(isinstance(arrays, tuple))
    if isinstance(arrays[0], int) or isinstance(arrays[0], float):
        return np.array(arrays)
    assert(all([arr.shape == arrays[0].shape for arr in arrays]))
    assert(all([arr.dtype == arrays[0].dtype for arr in arrays]))

    n_arr = len(arrays)
    n_elems = np.prod(arrays[0].shape)
    init_shape = arrays[0].shape
    dtype = arrays[0].dtype
    dst_shape = (*init_shape, n_arr)
    dst_arr = np.zeros((n_elems, n_arr), dtype=dtype)

    for i in range(n_arr):
        dst_arr[:, i] = arrays[i].reshape(-1)
    dst_arr.reshape(dst_shape)
    return dst_arr


def unstack_coords(array: NDArray) -> Tuple:
    assert(len(array.shape) > 1)
    n_stacks = array.shape[-1]
    stack_shape = array.shape[:-1]
    array = array.reshape(-1, n_stacks)
    arrays = [array[:, i].reshape(stack_shape) for i in range(n_stacks)]
    return tuple(arrays)


def get_subfolders(root, full_path=True):
    subfolders = [os.path.join(root, f) for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]
    if not full_path:
        subfolders = [os.path.basename(f) for f in subfolders]
    return subfolders