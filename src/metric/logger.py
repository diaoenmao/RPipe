import torch
from collections import defaultdict
from collections.abc import Iterable
from torch.utils.tensorboard import SummaryWriter
from numbers import Number
from module import ntuple
from .metric import make_metric


class Logger:
    def __init__(self, path, **kwargs):
        self.path = path
        if path is not None:
            self.writer = SummaryWriter(self.path)
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.path),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True
            )
        self.tracker = defaultdict(int)
        self.counter = defaultdict(int)
        self.mean = defaultdict(int)
        self.history = defaultdict(list)
        self.iterator = defaultdict(int)
        self.metric = make_metric(['train', 'test'], **kwargs)

    def save(self, flush):
        for name in self.mean:
            self.history[name].append(self.mean[name])
        if flush and self.writer is not None:
            self.flush()
        return

    def reset(self):
        self.tracker = defaultdict(int)
        self.counter = defaultdict(int)
        self.mean = defaultdict(int)
        return

    def append(self, result, split, n=1):
        for k in result:
            name = '{}/{}'.format(split, k)
            self.tracker[name] = result[k]
            if isinstance(result[k], Number):
                self.counter[name] += n
                self.mean[name] = ((self.counter[name] - n) * self.mean[name] + n * result[k]) / self.counter[name]
            elif isinstance(result[k], list) and len(result[k]) > 0 and isinstance(result[k][0], Number):
                if name not in self.mean:
                    self.counter[name] = [0 for _ in range(len(result[k]))]
                    self.mean[name] = [0 for _ in range(len(result[k]))]
                _ntuple = ntuple(len(result[k]))
                n = _ntuple(n)
                for i in range(len(result[k])):
                    self.counter[name][i] += n[i]
                    self.mean[name][i] = ((self.counter[name][i] - n[i]) * self.mean[name][i] + n[i] *
                                          result[k][i]) / self.counter[name][i]
        return

    def write(self, split, metric_name=None, writer=None):
        metric_name = self.metric.metric_name[split] if metric_name is None else metric_name
        writer = self.writer if writer is None else writer
        names = ['{}/{}'.format(split, k) for k in metric_name]
        evaluation_info = []
        for name in names:
            split, k = name.split('/')
            if isinstance(self.mean[name], Number):
                s = self.mean[name]
                evaluation_info.append('{}: {:.4f}'.format(k, s))
                if writer is not None:
                    self.iterator[name] += 1
                    writer.add_scalar(name, s, self.iterator[name])
            elif isinstance(self.mean[name], Iterable):
                s = tuple(self.mean[name])
                evaluation_info.append('{}: {}'.format(k, s))
                if writer is not None:
                    self.iterator[name] += 1
                    writer.add_scalar(name, s[0], self.iterator[name])
            else:
                raise ValueError('Not valid data type')
        info_name = '{}/info'.format(split)
        info = self.tracker[info_name]
        info[2:2] = evaluation_info
        info = '  '.join(info)
        if writer is not None:
            self.iterator[info_name] += 1
            writer.add_text(info_name, info, self.iterator[info_name])
        return info

    def add(self, split, input, output):
        evaluation = self.metric.add(split, input, output)
        return evaluation

    def evaluate(self, split, mode, input=None, output=None, metric_name=None):
        metric_name = self.metric.metric_name if metric_name is None else metric_name
        evaluation = self.metric.evaluate(split, mode, input, output, metric_name)
        return evaluation

    def compare(self, split, if_update=True):
        compare = self.metric.compare(self.mean['{}/{}'.format(split, self.metric.best_metric_name)], if_update)
        return compare

    def flush(self):
        self.writer.flush()
        return

    def state_dict(self):
        return {'tracker': self.tracker, 'counter': self.counter, 'mean': self.mean, 'history': self.history,
                'iterator': self.iterator, 'metric': self.metric.state_dict()}

    def load_state_dict(self, state_dict):
        self.tracker = state_dict['tracker']
        self.counter = state_dict['counter']
        self.mean = state_dict['mean']
        self.history = state_dict['history']
        self.iterator = state_dict['iterator']
        self.metric.load_state_dict(state_dict['metric'])
        return

    def update_state_dict(self, state_dict):
        for name in state_dict['tracker']:
            self.tracker[name] = state_dict['tracker'][name]
            if isinstance(state_dict['mean'][name], Number):
                n = state_dict['counter'][name]
                self.counter[name] += n
                self.mean[name] = ((self.counter[name] - n) * self.mean[name] + n * state_dict['mean'][name]) / \
                                  self.counter[name]
            elif isinstance(state_dict['mean'][name], list) and len(state_dict['mean'][name]) > 0:
                if name not in self.mean:
                    self.counter[name] = [0 for _ in range(len(state_dict['mean'][name]))]
                    self.mean[name] = [0 for _ in range(len(state_dict['mean'][name]))]
                _ntuple = ntuple(len(state_dict['mean'][name]))
                n = state_dict['counter'][name]
                n = _ntuple(n)
                for i in range(len(state_dict['mean'][name])):
                    if isinstance(state_dict['mean'][name][i], Number):
                        self.counter[name][i] += n[i]
                        self.mean[name][i] = ((self.counter[name][i] - n[i]) * self.mean[name][i] + n[i] *
                                              state_dict['mean'][name][i]) / self.counter[name][i]
            self.history[name].extend(state_dict['history'][name])
            self.iterator[name] += state_dict['iterator'][name]
        return


def make_logger(path=None, **kwargs):
    logger = Logger(path, **kwargs)
    return logger
