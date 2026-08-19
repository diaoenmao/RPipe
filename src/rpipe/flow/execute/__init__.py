"""execute: run algorithm mode declared in Control."""

from __future__ import annotations

from rpipe.artifact.asset import write_text_asset
from rpipe.flow.context import FlowContext
from rpipe.structure.algorithm import eval as eval_algo
from rpipe.structure.algorithm import inference, train


_MODES = {
    'train': train.run,
    'eval': eval_algo.run,
    'inference': inference.run,
}


def run(ctx: FlowContext) -> None:
    if ctx.control is None:
        raise RuntimeError('prepare must run before execute')
    mode = ctx.control.algorithm.get('mode') or 'train'
    fn = _MODES.get(mode)
    if fn is None:
        raise ValueError(f'unknown algorithm mode: {mode}')
    result = fn(ctx.control.algorithm, ctx.state)
    ctx.state['execute'] = [result]
    write_text_asset(ctx.layout, 'execute.log', f'mode={mode}\n')
    ctx.state.setdefault('observations', []).append({'phase': 'execute', 'mode': mode, 'result': result})
