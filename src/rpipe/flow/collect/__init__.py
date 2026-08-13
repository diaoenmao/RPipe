"""collect: gather execute observations into Result buffer."""

from __future__ import annotations

from rpipe.flow.context import FlowContext


def run(ctx: FlowContext) -> None:
    observations = list(ctx.state.get('observations') or [])
    metrics = {}
    for item in observations:
        for result in item.get('results') or []:
            if 'metric' in result:
                metrics.update(result['metric'])
            if 'loss' in result:
                metrics['loss'] = result['loss']
    ctx.state['collected'] = {
        'metrics': metrics,
        'observations': observations,
    }
