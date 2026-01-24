import copy
import torch
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from numbers import Number
from module import ntuple
from .metric import make_metric


class Logger:
    def __init__(self, path, tensorboard=True, profile=False, schedule=None, **kwargs):
        self.path = path
        if path is not None:
            if tensorboard:
                self.writer = SummaryWriter(self.path)
            else:
                self.writer = None
            if profile:
                tag = kwargs.get('tag', None)
                with_default_name = kwargs.get('with_default_name', True)
                with_ts = kwargs.get('with_ts', True)
                self.profiler = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    schedule=torch.profiler.schedule(**schedule),
                    on_trace_ready=tensorboard_trace_handler(self.path, worker_name=tag,
                                                             with_default_name=with_default_name, with_ts=with_ts),
                    profile_memory=True,
                )
            else:
                self.profiler = None
        else:
            self.writer = None
            self.profiler = None
        self.tracker = defaultdict(int)
        self.counter = defaultdict(int)
        self.mean = defaultdict(int)
        self.history = defaultdict(list)
        self.iterator = defaultdict(int)
        self.metric = make_metric(copy.deepcopy(kwargs['metric']))

    def save(self):
        for name in self.mean:
            self.history[name].append(self.mean[name])
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
        writer = self.writer if writer is None else writer
        if metric_name is None and split in self.metric.metric_name:
            metric_name = self.metric.metric_name[split]

        info_name = '{}/info'.format(split)
        info = self.tracker[info_name]

        evaluation_info = []
        if metric_name is not None:
            names = ['{}/{}'.format(split, k) for k in metric_name]
            for name in names:
                split, k = name.split('/')
                s = self.mean[name]
                evaluation_info.append('{}: {:.4f}'.format(k, s))
                if writer is not None:
                    self.iterator[name] += 1
                    writer.add_scalar(name, s, self.iterator[name])
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

    def compare(self, if_update=True):
        compare = self.metric.compare(self.mean['{}/{}'.format(self.metric.best_split,
                                                               self.metric.best_metric_name)], if_update)
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


def tensorboard_trace_handler(dir_name, worker_name=None, with_default_name=True, with_ts=True, use_gzip=False):
    """
    Outputs tracing files to directory of ``dir_name``, then that directory can be
    directly delivered to tensorboard as logdir.
    ``worker_name`` should be unique for each worker in distributed scenario,
    it will be set to '[hostname]_[pid]' by default.
    """
    import os
    import socket
    import time

    def handler_fn(prof) -> None:
        nonlocal worker_name
        if not os.path.isdir(dir_name):
            try:
                os.makedirs(dir_name, exist_ok=True)
            except Exception as e:
                raise RuntimeError("Can't create directory: " + dir_name) from e
        if not worker_name:
            worker_name = f"{socket.gethostname()}_{os.getpid()}"
        else:
            if with_default_name:
                worker_name = f"{worker_name}_{socket.gethostname()}_{os.getpid()}"

        # Use nanosecond here to avoid naming clash when exporting the trace
        if with_ts:
            file_name = f"{worker_name}.{time.time_ns()}.pt.trace.json"
        else:
            file_name = f"{worker_name}.pt.trace.json"
        if use_gzip:
            file_name = file_name + ".gz"
        prof.export_chrome_trace(os.path.join(dir_name, file_name))

    return handler_fn


def make_logger(path=None, **kwargs):
    logger = Logger(path, **kwargs)
    return logger
