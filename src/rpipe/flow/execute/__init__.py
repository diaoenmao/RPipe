"""execute: run the prepared Algorithm; update AlgorithmTracker."""

from __future__ import annotations

from rpipe.flow.context import FlowContext


def run(ctx: FlowContext) -> None:
    if ctx.control is None:
        raise RuntimeError('prepare must run before execute')
    algorithm = ctx.state.get('algorithm')
    tracker = ctx.state.get('tracker')
    logger = ctx.state.get('logger')
    if algorithm is None or tracker is None:
        raise RuntimeError('prepare must land algorithm and tracker')
    try:
        result = algorithm.run(
            ctx.state.get('data'),
            ctx.state.get('model'),
            ctx.state.get('system'),
            tracker,
        )
        ctx.state['execute'] = result
        ctx.state.setdefault('observations', []).append(
            {'phase': 'execute', 'mode': algorithm.mode}
        )
        if logger is not None:
            logger.info('execute finished')
    except Exception:
        if logger is not None:
            try:
                logger.error('execute failed')
            except Exception:
                pass
        raise
    finally:
        if tracker is not None:
            try:
                tracker.flush_state()
            except Exception:
                pass
