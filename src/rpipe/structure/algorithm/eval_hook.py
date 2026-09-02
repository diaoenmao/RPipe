"""Test-split body for train ``AlgorithmHook.on_eval_period`` (not a Flow mode)."""

from __future__ import annotations

from typing import Any

from rpipe.structure.algorithm.tracker import AlgorithmTracker


def eval_test_split(
    tracker: AlgorithmTracker,
    logger: Any,
    data: Any,
    model: Any,
    system: Any,
    extra: dict[str, Any] | None = None,
) -> dict[str, float]:
    import torch

    extra = extra or {}
    module = getattr(model, 'module', None)
    if module is None or not hasattr(data, 'iter_batches'):
        return {}
    device = torch.device(getattr(system, 'device', 'cpu'))
    module.eval()
    with torch.no_grad():
        for images, targets in data.iter_batches('test'):
            images = images.view(images.size(0), -1).to(device)
            targets = targets.to(device)
            logits = module(images)
            values = tracker.evaluate('test', 'batch', (images, targets), logits)
            tracker.append('test', n=int(images.size(0)), values=values)
    if logger is not None:
        logger.report(tracker, 'test', extra=extra)
    tracker.flush('test')
    tracker.save('test')
    tracker.reset('test')
    tracker.flush_state()
    module.train()
    return tracker.segment_mean('test')


def should_early_stop(
    *,
    accuracy: float | None,
    best: float | None,
    stall: int,
    patience: int | None,
    min_delta: float,
) -> tuple[bool, float | None, int]:
    """Maximize test Accuracy. ``patience`` is consecutive non-improving evals."""
    if patience is None or accuracy is None:
        return False, best if best is not None else accuracy, stall
    if best is None or accuracy > best + min_delta:
        return False, accuracy, 0
    stall += 1
    return stall >= int(patience), best, stall
