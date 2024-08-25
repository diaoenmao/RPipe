import os
import torch
from .io import makedir_exist_ok, load


def make_stats(name):
    stats = None
    stats_path = os.path.join('output', 'stats')
    makedir_exist_ok(stats_path)
    filenames = os.listdir(stats_path)
    for filename in filenames:
        if name == filename:
            stats = load(os.path.join(stats_path, filename), 'torch')
    return stats


class Stats(object):
    def __init__(self, dim):
        self.dim = dim
        self.n_samples = 0
        self.n_features = None
        self.mean = None
        self.std = None
        self.min = None
        self.max = None

    def update(self, data):
        data = data.transpose(self.dim, -1).reshape(-1, data.size(self.dim))
        if self.n_samples == 0:
            self.n_samples = data.size(0)
            self.n_features = data.size(1)
            self.mean = data.mean(dim=0)
            self.std = data.std(dim=0)
        else:
            m = float(self.n_samples)
            n = data.size(0)
            new_mean = data.mean(dim=0)
            new_std = 0 if n == 1 else data.std(dim=0)
            old_mean = self.mean
            old_std = self.std
            self.mean = m / (m + n) * old_mean + n / (m + n) * new_mean
            self.std = torch.sqrt(m / (m + n) * old_std ** 2 + n / (m + n) * new_std ** 2 + m * n / (m + n) ** 2 * (
                    old_mean - new_mean) ** 2)
            self.n_samples += n

        new_min = data.min(dim=0)[0]
        if self.min is None:
            self.min = new_min
        min_mask = new_min < self.min
        self.min[min_mask] = new_min[min_mask]

        new_max = data.max(dim=0)[0]
        if self.max is None:
            self.max = new_max
        max_mask = new_max > self.max
        self.max[max_mask] = new_max[max_mask]
        return

    def __repr__(self):
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        attrs_str = ', '.join(f'{k}={v}' for k, v in attrs.items())
        return 'Stats({})'.format(attrs_str)

