"""collect: metrics summary from AlgorithmTracker (not full curves)."""

from __future__ import annotations

from typing import Any

from rpipe.flow.context import FlowContext


def run(ctx: FlowContext) -> None:
    observations = list(ctx.state.get('observations') or [])
    metrics: dict[str, Any] = {}
    tracker = ctx.state.get('tracker')
    if tracker is not None:
        train = tracker.segment_mean('train')
        test = tracker.segment_mean('test')
        for name, value in train.items():
            metrics[f'train_{name.lower()}'] = float(value)
        for name, value in test.items():
            metrics[f'test_{name.lower()}'] = float(value)
        if 'Loss' in train:
            metrics['train_loss'] = float(train['Loss'])
        if 'Accuracy' in test:
            metrics['accuracy'] = float(test['Accuracy'])
        elif 'Accuracy' in train:
            metrics['accuracy'] = float(train['Accuracy'])
    execute = ctx.state.get('execute') or {}
    if execute.get('best_accuracy') is not None:
        metrics['best_accuracy'] = float(execute['best_accuracy'])
    elif execute.get('best_value') is not None:
        metrics['best_value'] = float(execute['best_value'])
        if execute.get('best_metric'):
            metrics['best_metric'] = execute['best_metric']
    if execute.get('train_loss') is not None and 'train_loss' not in metrics:
        metrics['train_loss'] = float(execute['train_loss'])
    if execute.get('accuracy') is not None and 'accuracy' not in metrics:
        metrics['accuracy'] = float(execute['accuracy'])
    if execute.get('elapsed_seconds') is not None:
        metrics['elapsed_seconds'] = float(execute['elapsed_seconds'])
    if execute.get('elapsed') is not None:
        metrics['elapsed'] = execute['elapsed']
    if execute.get('mode') == 'eval' and 'accuracy' in metrics:
        metrics['eval_accuracy'] = float(metrics['accuracy'])
    ctx.state['collected'] = {
        'metrics': metrics,
        'observations': observations,
    }
