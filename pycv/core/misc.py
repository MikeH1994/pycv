from __future__ import annotations
import numpy as np
from typing import Tuple
from numpy.typing import NDArray
import os
import glob

def stack_coords(arrays: Tuple[np.ndarray, ...]) -> NDArray:
    assert(isinstance(arrays, tuple))
    if isinstance(arrays[0], int) or isinstance(arrays[0], float):
        return np.array(arrays)
    assert(all([arr.shape == arrays[0].shape for arr in arrays]))
    assert(all([arr.dtype == arrays[0].dtype for arr in arrays]))
    return np.stack(arrays, axis=-1)


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

def get_all_files_in_folder(folderpath, extension, recursive=False):
    if recursive:
        return sorted(glob.glob(os.path.join(folderpath, "**/*{}".format(extension)), recursive=True))
    else:
        return sorted(glob.glob(os.path.join(folderpath, "*{}".format(extension)), recursive=False))


def get_all_folders_containing_filetype(root_dir, extension, recursive=True):
    folders_with_files = []
    files_in_folders = []

    root_depth = root_dir.rstrip(os.sep).count(os.sep)
    max_depth = 1 if not recursive else None
    for current_dir, dirs, files in os.walk(root_dir):
        current_depth = current_dir.count(os.sep) - root_depth
        if max_depth is not None and current_depth > max_depth:
            # Prevent descending further by clearing dirs
            dirs[:] = []
            continue

        pattern = os.path.join(current_dir, f"*{extension}")
        matched_files = glob.glob(pattern)

        if matched_files:
            folders_with_files.append(current_dir)
            files_in_folders.append(matched_files)

    return folders_with_files, files_in_folders
