"""execute: run algorithm semantics declared in Control."""

from __future__ import annotations

from rpipe.artifact.asset import write_text_asset
from rpipe.flow.context import FlowContext
from rpipe.structure.algorithm import eval as eval_algo
from rpipe.structure.algorithm import inference, train


_SEMANTICS = {
    'train': train.run,
    'eval': eval_algo.run,
    'inference': inference.run,
}


def run(ctx: FlowContext) -> None:
    if ctx.control is None:
        raise RuntimeError('prepare must run before execute')
    requested = ctx.control.algorithm.get('semantics') or ['train', 'eval']
    if isinstance(requested, str):
        requested = [requested]
    results = []
    for name in requested:
        fn = _SEMANTICS.get(name)
        if fn is None:
            raise ValueError(f'unknown algorithm semantic: {name}')
        results.append(fn(ctx.control.algorithm, ctx.state))
    ctx.state['execute'] = results
    write_text_asset(ctx.layout, 'execute.log', f'semantics={list(requested)}\n')
    ctx.state.setdefault('observations', []).append({'phase': 'execute', 'results': results})
