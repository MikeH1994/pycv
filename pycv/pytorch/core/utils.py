from __future__ import annotations
import numpy as np
import torch
from typing import List, Union, Tuple
from numpy.typing import NDArray
from tqdm import tqdm
import pickle
import random
import cv2
from typing import Dict
from .configuration import Configuration


def setup_namespace():
    return


def setup_system(config: Configuration):
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn_benchmark_enabled = config.cudnn_benchmark_enabled
        torch.backends.cudnn.deterministic = config.cudnn_deterministic


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def get_min_and_max_from_list(src: List[NDArray]):
    max_val = -np.inf
    min_val = np.inf
    for i in range(len(src)):
        if src[i][src[i] > 0].size == 0:
            continue
        max_val = max(max_val, np.max(src[i][src[i] > 0]))
        min_val = min(min_val, np.min(src[i][src[i] > 0]))
    return min_val, max_val


def convert_numpy_to_8_bit(src: NDArray, min_val=None, max_val=None, return_as_rgb=False):
    # if a single image is supplied
    if src.dtype == np.uint8:
        return src

    min_val = np.min(src) if min_val is None else min_val
    max_val = np.max(src) if max_val is None else max_val

    if min_val > max_val:
        raise Exception("Invalid min and max bounds found!")

    if min_val == max_val:
        scale_factor = 1.0
    else:
        scale_factor = 255.0 / (max_val - min_val)
    img = src.astype(np.float32)
    img[img < min_val] = min_val
    img[img > max_val] = max_val
    img -= min_val
    img *= scale_factor
    img = img.astype(np.uint8)
    if return_as_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def save_model(model, device, model_fpath):
    assert(model_fpath.endswith('.pt'))
    if device == 'cuda':
        model.to('cpu')
    torch.save(model.state_dict(), model_fpath)
    if device == 'cuda':
        model.to('cuda')

def load_model(model, model_fpath):
    model.load_state_dict(torch.load(model_fpath), strict=False)
    return model


def save_config(config, config_fpath):
    with open(config_fpath, 'wb') as f:
        pickle.dump(config, f)


def load_config(config_fpath):
    with open(config_fpath, 'rb') as f:
        config = pickle.load(f)
    return config


def tensor_img_to_numpy(img, offset=(0.0, 0.0, 0.0), scale_factor=(1.0, 1.0, 1.0), convert_to_8_bit=False):
    img = torch.clone(img).float()
    if isinstance(scale_factor, float) or isinstance(scale_factor, int):
        img *= scale_factor
    else:
        img[0] *= scale_factor[0]
        img[1] *= scale_factor[1]
        img[2] *= scale_factor[2]
    if isinstance(offset, float) or isinstance(scale_factor, int):
        img += offset
    else:
        img[0] += offset[0]
        img[1] += offset[1]
        img[2] += offset[2]

    if len(img.shape) == 2:
        img = img.numpy()
    else:
        img = img.numpy().transpose(1, 2, 0)
    if convert_to_8_bit:
        img = img.astype(np.uint8)
    return img


def numpy_img_to_tensor(image: NDArray, mean: NDArray = np.array([0.0, 0.0, 0.0]),
                        std: NDArray = np.array([1.0, 1.0, 1.0])) -> torch.Tensor:
    """
    Converts a numpy image to a pytorch
    tensor, so that it can be passed to the model. The images are scaled by the
        mean and standard deviation of the training dataset to give a mean of zero and a standard deviation of 1.0,
        i.e. img = (img - mean) / std
    :param image: the image to be converted to a tor
    :type image: numpy array
    :param mean: the mean
    :param std:
    :return:
    """
    assert(isinstance(image, np.ndarray))
    assert(len(image.shape) == 3 and image.shape[2] == 3)

    image = image.astype(np.float32)
    image -= mean
    image /= std

    image = image.transpose((2, 0, 1))
    image = torch.tensor(image)

    return image

def calc_mean_std(loader, show_loading_bar=False, description="Calculating mean") -> Tuple[torch.FloatTensor,
                                                                                           torch.FloatTensor]:
    batch_mean = torch.zeros(3)
    batch_mean_sqrd = torch.zeros(3)
    progress_bar = tqdm(total=len(loader), disable=not show_loading_bar, dynamic_ncols=True)
    progress_bar.set_description(description)

    for samples in loader:
        progress_bar.update()
        batch_data = samples["inputs"]
        batch_mean += batch_data.mean(dim=(0, 2, 3))
        batch_mean_sqrd += (batch_data ** 2).mean(dim=(0, 2, 3))

    mean = batch_mean / len(loader)
    var = (batch_mean_sqrd / len(loader)) - (mean ** 2)
    std = var ** 0.5
    return mean, std


def verify_classes(classes: Dict[int, str]):
    names = list(classes.values())
    indices = list(classes.keys())

    # check keys and indices are valid
    for i, class_idx in enumerate(classes):
        class_name = classes[class_idx]
        assert(isinstance(class_idx, int)), "Position {}: Class indices must be integers. " \
                                            "{} is of type {}".format(i, class_idx, type(class_idx))
        assert(isinstance(class_name, str)), "Position {}: Class names must be strings. " \
                                             "{} is of type {}".format(i, class_name, type(class_name))
        assert(names.count(class_name) == 1), "Position {}: Class name is repeated.  " \
                                              "{} occurs {} times. Classes: {}".format(i, class_name,
                                                                                       names.count(class_name), classes)

    # check indices are valid
    min_index = min(indices)
    max_index = max(indices)
    assert(min_index == 0), "The minimum index should be 0 (for background)"
    for i in range(max_index+1):
        assert(i in classes), "Class index {} not found in supplied classes. Class indices: {}".format(i, indices)


def load_classes_from_txt_file(fpath, add_background=False) -> Dict[int, str]:
    classes = {}
    offset = 1 if add_background else 0

    if add_background:
        classes[0] = "background"

    with open(fpath, 'r') as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            class_idx = int(index + offset)
            classes[class_idx] = line
    verify_classes(classes)
    return classes


def flip_classes(classes: Union[Dict[int, str], Dict[str, int]]) -> Dict:
    return dict((v, k) for k, v in classes.items())
