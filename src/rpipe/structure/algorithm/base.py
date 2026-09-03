"""Algorithm base class (no factory import)."""

from __future__ import annotations

from typing import Any

from rpipe.structure.algorithm.config import AlgorithmConfig
from rpipe.structure.algorithm.hook import AlgorithmHook
from rpipe.structure.algorithm.optim import make_optimizer as _make_optimizer
from rpipe.structure.algorithm.optim import make_scheduler as _make_scheduler
from rpipe.structure.algorithm.resume import apply_module_state, resume_stem
from rpipe.structure.algorithm.tracker import AlgorithmTracker


class Algorithm(AlgorithmHook):
    """Computation for one mode. Optimizer / scheduler / resume live here."""

    def __init__(self, config: AlgorithmConfig) -> None:
        self.config = config
        self.mode = config.mode or 'train'

    def make_optimizer(self, module: Any) -> Any:
        return _make_optimizer(module, self.config)

    def make_scheduler(self, optimizer: Any, t_max: int) -> Any:
        return _make_scheduler(optimizer, self.config, t_max)

    def resume(
        self,
        data: Any,
        model: Any,
        system: Any,
        tracker: AlgorithmTracker,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        del data, tracker
        extra = extra if extra is not None else {}
        stem = resume_stem(self.config, mode=self.mode)
        extra['resume_stem'] = stem
        if stem is None:
            return None
        loader = getattr(system, 'load_checkpoint', None)
        payload = loader(stem) if callable(loader) else None
        logger = getattr(system, 'logger', None)
        if payload is None:
            if self.mode == 'eval':
                raise FileNotFoundError(f'eval resume missing checkpoint {stem!r}')
            if logger is not None:
                logger.info(f'resume skip (no {stem})')
            return None
        apply_module_state(getattr(model, 'module', None), payload)
        extra['resume_payload'] = payload
        if logger is not None:
            logger.info(f"resume {stem} epoch={payload.get('epoch')} step={payload.get('step')}")
        return payload

    def run(self, data: Any, model: Any, system: Any, tracker: AlgorithmTracker) -> dict[str, Any]:
        raise NotImplementedError
