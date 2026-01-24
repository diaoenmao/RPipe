import inspect
import torch
from collections.abc import Iterable, Mapping
from itertools import repeat


def filter_args(func, arg_dict):
    sig = inspect.signature(func)
    valid_args = {k: v for k, v in arg_dict.items() if k in sig.parameters}
    return valid_args


def ntuple(n):
    def parse(x):
        if isinstance(x, Iterable) and not isinstance(input, (str, bytes)):
            return x
        return tuple(repeat(x, n))

    return parse


def apply_recursively(fn, input, *args, apply_condition, identity_condition=None, key=None):
    if apply_condition(input):
        sig = inspect.signature(fn)
        if 'key' in sig.parameters:
            result = fn(input, *args, key=key)
        else:
            result = fn(input, *args)
    elif identity_condition is not None and identity_condition(input):
        result = input
    elif isinstance(input, Mapping):
        result = {}
        for key_, value_ in input.items():
            updated_key = key_ if key is None else '{}.{}'.format(key, key_)
            result[key_] = apply_recursively(fn, value_, *args, apply_condition=apply_condition,
                                             identity_condition=identity_condition, key=updated_key)
    elif isinstance(input, Iterable) and not isinstance(input, (str, bytes)):
        result = []
        for i, item in enumerate(input):
            updated_key = i if key is None else '{}.{}'.format(key, i)
            result_i = apply_recursively(fn, item, *args, apply_condition=apply_condition,
                                         identity_condition=identity_condition, key=updated_key)
            result.append(result_i)
    else:
        raise ValueError('Not valid input type: {} with value {}'.format(type(input), input))
    return result


def to_device(input, device):
    apply_condition = lambda x: isinstance(x, torch.Tensor)
    identity_condition = lambda x: isinstance(x, (str, type(None)))
    fn = lambda x, y: x.to(y)
    output = apply_recursively(fn, input, device,
                               apply_condition=apply_condition, identity_condition=identity_condition)
    return output
