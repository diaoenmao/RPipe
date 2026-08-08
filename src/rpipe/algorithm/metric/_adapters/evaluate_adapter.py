from __future__ import annotations

from collections import defaultdict
from typing import Any


def build_hf_evaluate_bank(metric_name: dict[str, list[str]]) -> dict:
    import evaluate

    bank = defaultdict(dict)
    for split, names in metric_name.items():
        for m in names:
            if m == 'Loss':
                bank[split][m] = {
                    'mode': 'batch',
                    'metric': (lambda input, output: float(output['loss'].detach().item())),
                }
            elif m == 'Accuracy':
                metric = evaluate.load('accuracy')
                bank[split][m] = {'mode': 'batch', 'metric': _AccuracyHF(metric)}
            else:
                raise ValueError(f'hf_evaluate provider does not map {m!r} yet')
    return bank


class _AccuracyHF:
    def __init__(self, metric):
        self.metric = metric

    def __call__(self, input: dict[str, Any], output: dict[str, Any]) -> float:
        target = input.get('target', input.get('labels'))
        pred = output.get('pred', output.get('logits'))
        if hasattr(pred, 'ndim') and pred.ndim > 1:
            pred = pred.argmax(dim=-1)
        refs = target.detach().cpu().tolist()
        preds = pred.detach().cpu().tolist()
        out = self.metric.compute(predictions=preds, references=refs)
        return float(out['accuracy'] * 100.0)
