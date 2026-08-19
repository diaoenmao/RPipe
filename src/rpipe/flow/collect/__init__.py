"""collect: gather execute observations into Result buffer."""

from __future__ import annotations

from rpipe.flow.context import FlowContext


def run(ctx: FlowContext) -> None:
    observations = list(ctx.state.get('observations') or [])
    metrics = {}
    for item in observations:
        results = item.get('results')
        if results is None and 'result' in item:
            results = [item['result']]
        for result in results or []:
            if not isinstance(result, dict):
                continue
            if 'metric' in result:
                metrics.update(result['metric'])
            if 'loss' in result:
                metrics['loss'] = result['loss']
            if 'accuracy' in result:
                metrics['accuracy'] = result['accuracy']
    ctx.state['collected'] = {
        'metrics': metrics,
        'observations': observations,
    }
