import numpy as np
import os
import pickle
import torch


def check_exists(path):
    return os.path.exists(path)


def makedir_exist_ok(path):
    os.makedirs(path, exist_ok=True)
    return


def save(input, path, mode='torch'):
    dirname = os.path.dirname(path)
    makedir_exist_ok(dirname)
    if mode == 'torch':
        torch.save(input, path)
    elif mode == 'np':
        np.save(path, input, allow_pickle=True)
    elif mode == 'pickle':
        with open(path, 'wb') as file:
            pickle.dump(input, file)
    else:
        raise ValueError('Not valid save mode')
    return


def load(path, mode='torch'):
    if mode == 'torch':
        output = torch.load(path, weights_only=False)
    elif mode == 'np':
        output = np.load(path, allow_pickle=True)
    elif mode == 'pickle':
        with open(path, 'rb') as file:
            output = pickle.load(file)
    else:
        raise ValueError('Not valid save mode')
    return output


def io_mode(filename):
    if filename in ['cfg', 'model', 'optimizer']:
        mode = 'torch'
    else:
        mode = 'pickle'
    return mode


def check(result, path, mode=None):
    for filename in result:
        mode_i = io_mode(filename) if mode is None else mode
        save(result[filename], os.path.join(path, filename), mode_i)
    return


def resume(path, resume_mode=True, key=None, verbose=True, mode=None):
    if os.path.exists(path):
        if isinstance(resume_mode, bool) and resume_mode:
            result = {}
            filenames = os.listdir(path)
            for filename in filenames:
                if not os.path.isfile(os.path.join(path, filename)) or (key is not None and filename not in key):
                    continue
                mode_i = io_mode(filename) if mode is None else mode
                result[filename] = load(os.path.join(path, filename), mode_i)
        elif isinstance(resume_mode, dict):
            result = {}
            for filename in resume_mode:
                if not resume_mode[filename] or not os.path.isfile(os.path.join(path, filename)) or \
                        (key is not None and filename not in key):
                    continue
                mode_i = io_mode(filename) if mode is None else mode
                result[filename] = load(os.path.join(path, filename), mode_i)
        else:
            result = None
            if resume_mode and verbose:
                print('Not exists: {}'.format(path))
        if result is not None and len(result) > 0 and verbose:
            print('Resume complete')
    else:
        if resume_mode and verbose:
            print('Not exists: {}'.format(path))
        result = None
    return result
