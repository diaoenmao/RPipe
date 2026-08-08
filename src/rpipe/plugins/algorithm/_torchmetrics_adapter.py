"""Adapt torchmetrics objects into the native Metric evaluate/add API."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch


def build_torchmetrics_bank(metric_name: dict[str, list[str]]) -> dict:
    import torchmetrics

    bank = defaultdict(dict)
    for split, names in metric_name.items():
        for m in names:
            if m == 'Loss':
                bank[split][m] = {
                    'mode': 'batch',
                    'metric': (lambda input, output: float(output['loss'].detach().item())),
                }
            elif m == 'Accuracy':
                tm = torchmetrics.classification.MulticlassAccuracy(num_classes=10, average='micro')
                bank[split][m] = {'mode': 'batch', 'metric': _AccuracyAdapter(tm)}
            elif m in ('MSE', 'MeanSquaredError'):
                tm = torchmetrics.regression.MeanSquaredError()
                bank[split][m] = {'mode': 'batch', 'metric': _RegressionAdapter(tm)}
            else:
                raise ValueError(f'torchmetrics provider does not map metric {m!r} yet')
    return bank


class _AccuracyAdapter:
    def __init__(self, metric):
        self.metric = metric

    def __call__(self, input: dict[str, Any], output: dict[str, Any]) -> float:
        target = input.get('target', input.get('labels'))
        pred = output.get('pred', output.get('logits'))
        if pred.ndim > 1:
            pred_cls = pred.argmax(dim=-1)
        else:
            pred_cls = pred
        # reset per-batch for batch-mode compatibility with current Logger
        self.metric.reset()
        num_classes = int(pred.shape[-1]) if pred.ndim > 1 else int(max(pred_cls.max().item(), target.max().item()) + 1)
        if getattr(self.metric, 'num_classes', None) != num_classes:
            import torchmetrics
            self.metric = torchmetrics.classification.MulticlassAccuracy(
                num_classes=max(num_classes, 2), average='micro')
        val = self.metric(pred_cls.cpu(), target.cpu())
        return float(val.item() * 100.0)


class _RegressionAdapter:
    def __init__(self, metric):
        self.metric = metric

    def __call__(self, input: dict[str, Any], output: dict[str, Any]) -> float:
        target = input['target'].float()
        pred = output['pred'].float()
        self.metric.reset()
        return float(self.metric(pred.cpu(), target.cpu()).item())
