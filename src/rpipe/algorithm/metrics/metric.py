from collections import defaultdict
from .utils import Accuracy, MSE, RMSE, GLUE


class Metric:

    def __init__(self, metric_name, best, best_split, best_direction, best_metric_name, **kwargs):
        self.metric_name = metric_name
        self.best = best
        self.best_split = best_split
        self.best_direction = best_direction
        self.best_metric_name = best_metric_name
        self.metric = self.make_metric(metric_name, **kwargs)

    def make_metric(self, metric_name, **kwargs):
        metric = defaultdict(dict)
        for split in metric_name:
            for m in metric_name[split]:
                if m == 'Loss':
                    metric[split][m] = {'mode': 'batch', 'metric': (lambda input, output: output['loss'].item())}
                elif m == 'Accuracy':
                    metric[split][m] = {'mode': 'batch', 'metric': (lambda input, output: Accuracy(input, output))}
                elif m == 'MSE':
                    metric[split][m] = {'mode': 'batch', 'metric': (lambda input, output: MSE(input, output))}
                elif m == 'RMSE':
                    metric[split][m] = {'mode': 'full', 'metric': RMSE()}
                elif m == 'GLUE':
                    metric[split][m] = {'mode': 'batch', 'metric': GLUE(kwargs['subset_name'])}
                else:
                    msg = 'Not valid metric name'
                    raise ValueError(msg)
        return metric

    def add(self, split, input, output):
        for metric_name in self.metric_name[split]:
            if self.metric[split][metric_name]['mode'] == 'full':
                self.metric[split][metric_name]['metric'].add(input, output)
        return

    def evaluate(self, split, mode, input, output, metric_name):
        evaluation = {}
        for metric_name_i in metric_name[split]:
            if self.metric[split][metric_name_i]['mode'] == mode:
                evaluation[metric_name_i] = self.metric[split][metric_name_i]['metric'](input, output)
        return evaluation

    def compare(self, val, if_update):
        if self.best_direction == 'down':
            compared = self.best > val
        elif self.best_direction == 'up':
            compared = self.best < val
        else:
            compared = None
            msg = 'Not valid best direction'
            raise ValueError(msg)
        if compared and if_update:
            self.best = val
        return compared

    def load_state_dict(self, state_dict):
        self.best = state_dict['best']
        self.best_split = state_dict['best_split']
        self.best_direction = state_dict['best_direction']
        self.best_metric_name = state_dict['best_metric_name']
        return

    def state_dict(self):
        return {'best': self.best, 'best_split': self.best_split,
                'best_direction': self.best_direction, 'best_metric_name': self.best_metric_name}


def make_metric(kwargs):
    metric_name = kwargs.pop('metric_name')
    best_split = kwargs.pop('best_split')
    best_metric_name = kwargs.pop('best_metric_name')
    if best_metric_name in ['Loss']:
        best = float('inf')
        best_direction = 'down'
    elif best_metric_name in ['Accuracy']:
        best = -float('inf')
        best_direction = 'up'
    else:
        best, best_direction = None, None
        msg = 'Not valid best metric name'
        raise ValueError(msg)
    metric = Metric(metric_name, best, best_split, best_direction, best_metric_name, **kwargs)
    return metric
