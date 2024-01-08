import errno
import numpy as np
import os
import pickle
import torch
from torchvision.utils import save_image
from .utils import recur


def check_exists(path):
    return os.path.exists(path)


def makedir_exist_ok(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    return


def save(input, path, mode='pickle'):
    dirname = os.path.dirname(path)
    makedir_exist_ok(dirname)
    if mode == 'torch':
        torch.save(input, path)
    elif mode == 'np':
        np.save(path, input, allow_pickle=True)
    elif mode == 'pickle':
        pickle.dump(input, open(path, 'wb'))
    else:
        raise ValueError('Not valid save mode')
    return


def load(path, mode='pickle'):
    if mode == 'torch':
        return torch.load(path, map_location=lambda storage, loc: storage)
    elif mode == 'np':
        return np.load(path, allow_pickle=True)
    elif mode == 'pickle':
        return pickle.load(open(path, 'rb'))
    else:
        raise ValueError('Not valid save mode')
    return


def save_img(img, path, nrow=10, padding=1, pad_value=0, value_range=None):
    makedir_exist_ok(os.path.dirname(path))
    normalize = False if range is None else True
    save_image(img, path, nrow=nrow, padding=padding, pad_value=pad_value, normalize=normalize, value_range=value_range)
    return


def to_device(input, device):
    output = recur(lambda x, y: x.to(y), input, device)
    return output


def check(result, path):
    for filename in result:
        save(result[filename], os.path.join(path, filename))
    return


def resume(path, resume_mode=1, key=None, verbose=True):
    if os.path.exists(path) and resume_mode == 1:
        result = {}
        filenames = os.listdir(path)
        for filename in filenames:
            if key is not None and filename not in key:
                continue
            result[filename] = load(os.path.join(path, filename))
        if len(result) > 0 and verbose:
            print('Resume complete')
    else:
        if resume_mode == 1 and verbose:
            print('Not exists: {}'.format(path))
        result = None
    return result
