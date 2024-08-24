import torch
from collections.abc import Iterable, Mapping
from itertools import repeat


def ntuple(n):
    def parse(x):
        if isinstance(x, Iterable) and not isinstance(input, (str, bytes)):
            return x
        return tuple(repeat(x, n))

    return parse


def apply_recursively(fn, input, *args, apply_condition, identity_condition=None):
    if apply_condition(input):
        return fn(input, *args)
    elif identity_condition is not None and identity_condition(input):
        return input
    elif isinstance(input, Mapping):
        return {key: apply_recursively(fn, value, *args, apply_condition=apply_condition,
                                       identity_condition=identity_condition) for key, value in input.items()}
    elif isinstance(input, Iterable) and not isinstance(input, (str, bytes)):
        return [apply_recursively(fn, item, *args, apply_condition=apply_condition,
                                  identity_condition=identity_condition) for item in input]
    else:
        raise ValueError('Not valid input type: {} with value {}'.format(type(input), input))


def to_device(input, device):
    apply_condition = lambda x: isinstance(x, torch.Tensor)
    identity_condition = lambda x: isinstance(x, (str, type(None)))
    fn = lambda x, y: x.to(y)
    output = apply_recursively(fn, input, device,
                               apply_condition=apply_condition, identity_condition=identity_condition)
    return output
