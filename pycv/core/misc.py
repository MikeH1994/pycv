from __future__ import annotations
import numpy as np
from typing import Tuple
from numpy.typing import NDArray
import os
import glob
import math
from pathlib import Path
from typing import Union
from datetime import datetime


def stack(*arrays: Union[np.ndarray, int, float, tuple, list]) -> np.ndarray:
    # If the first argument is a tuple, unpack it
    if len(arrays) == 1 and isinstance(arrays[0], tuple) or isinstance(arrays[0], list):
        arrays = arrays[0]

    if isinstance(arrays[0], int) or isinstance(arrays[0], float):
        return np.array(arrays)
    assert(all([arr.shape == arrays[0].shape for arr in arrays]))
    assert(all([arr.dtype == arrays[0].dtype for arr in arrays]))
    return np.stack(arrays, axis=-1)

def unstack(array: np.ndarray) -> Tuple:
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

    return folders_with_files

def round_up_to(x, base):
    return math.ceil(x / base) * base

def round_to(x, base):
    return round(x / base) * base

def round_down_to(x, base):
    return math.floor(x / base) * base

def clamp(x, lower, upper):
    return np.clip(x, lower, upper)


def get_nth_parent_name(filepath, n):
    path = Path(filepath)
    if n-2 >= len(path.parents):
        raise Exception("n is out of range")
    return path.parents[n-2].name


def sort_lists_together(tuple_of_lists):
    sort_indices = sorted(range(len(tuple_of_lists[0])), key=lambda i: tuple_of_lists[0][i])
    sorted_lists = tuple([[lst[i] for i in sort_indices] for lst in tuple_of_lists])
    return sorted_lists

def current_time_str():
    # Format local time as HH:MM:SS
    return datetime.now().strftime("%H:%M:%S")

def format_bounds(x0, x1, interval, mode = "outside"):
    assert(mode in ["outside", "inside", "nearest"])
    if mode == "outside":
        x0 = round_down_to(x0, interval)
        x1 = round_up_to(x1, interval)
    elif mode == "inside":
        x0 = round_up_to(x0, interval)
        x1 = round_down_to(x1, interval)
    elif mode == "nearest":
        x0 = round_to(x0, interval)
        x1 = round_to(x1, interval)
    return x0, x1


def rms(x):
    return np.sqrt(np.square(x).mean())