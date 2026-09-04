"""collect: metrics summary from AlgorithmTracker (not full curves)."""

from __future__ import annotations

from rpipe.flow.context import FlowContext


def run(ctx: FlowContext) -> None:
    observations = list(ctx.state.get('observations') or [])
    metrics: dict[str, float] = {}
    tracker = ctx.state.get('tracker')
    if tracker is not None:
        train = tracker.segment_mean('train')
        test = tracker.segment_mean('test')
        if 'Loss' in train:
            metrics['train_loss'] = train['Loss']
        if 'Accuracy' in test:
            metrics['accuracy'] = test['Accuracy']
        elif 'Accuracy' in train:
            metrics['accuracy'] = train['Accuracy']
    execute = ctx.state.get('execute') or {}
    if execute.get('best_accuracy') is not None:
        metrics['best_accuracy'] = float(execute['best_accuracy'])
    elif execute.get('best_value') is not None:
        metrics['best_value'] = float(execute['best_value'])
        if execute.get('best_metric'):
            metrics['best_metric'] = execute['best_metric']
    if execute.get('mode') == 'eval' and 'accuracy' in metrics:
        metrics['eval_accuracy'] = float(metrics['accuracy'])
    ctx.state['collected'] = {
        'metrics': metrics,
        'observations': observations,
    }
