"""Train budget and checkpoint cadence (epoch or step)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

UNIT_EPOCH = 'epoch'
UNIT_STEP = 'step'
CHECKPOINT_LATEST = 'latest'
CHECKPOINT_PERCENT = 'percent'
DEFAULT_PERCENTS = (0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class ProgressBudget:
    unit: str
    num_epochs: int | None
    num_steps: int
    total: int
    steps_per_epoch: int | None = None
    steps_from_epochs: bool = False

    def progress(self, *, epoch: int, step: int) -> int:
        return epoch if self.unit == UNIT_EPOCH else step

    def scheduler_t_max(self) -> int:
        if self.unit == UNIT_STEP:
            return max(int(self.num_steps), 1)
        if self.num_epochs is not None:
            return max(int(self.num_epochs), 1)
        if self.num_steps is not None:
            return max(int(self.num_steps), 1)
        return 1


def infer_steps_per_epoch(data: Any) -> int | None:
    """Infer train steps/epoch from Data.steps_per_epoch, loader length, or metadata."""
    method = getattr(data, 'steps_per_epoch', None)
    if callable(method):
        try:
            value = method()
        except TypeError:
            value = None
        if value:
            return int(value)
    loaders = getattr(data, '_loaders', None)
    if isinstance(loaders, dict):
        train_loader = loaders.get('train')
        if train_loader is not None:
            try:
                value = int(len(train_loader))
                if value > 0:
                    return value
            except (TypeError, ValueError):
                pass
    meta = getattr(data, 'meta', None)
    if isinstance(meta, dict):
        train_size = meta.get('train_size')
        batch_size = meta.get('batch_size')
        if train_size and batch_size:
            import math

            return max(int(math.ceil(int(train_size) / int(batch_size))), 1)
    return None


def resolve_budget(config: Any, *, steps_per_epoch: int | None = None) -> ProgressBudget:
    raw_epochs = config.setting('num_epochs')
    raw_steps = config.setting('num_steps')
    raw_unit = config.setting('progress_unit')
    epochs = int(raw_epochs) if raw_epochs is not None else None
    steps = int(raw_steps) if raw_steps is not None else None
    from_epochs = False
    inferred_spe = int(steps_per_epoch) if steps_per_epoch is not None else None
    if epochs is not None:
        if inferred_spe is not None and inferred_spe > 0:
            steps = epochs * inferred_spe
            from_epochs = True
        elif steps is None:
            # Fallback for datasets without epoch cardinality.
            steps = epochs
    if raw_unit is not None:
        unit = str(raw_unit).lower()
        if unit not in (UNIT_EPOCH, UNIT_STEP):
            raise ValueError(f'progress_unit must be epoch or step, got {raw_unit!r}')
    else:
        unit = UNIT_STEP
    if steps is None:
        steps = 1
    if unit == UNIT_EPOCH:
        if epochs is None:
            raise ValueError('progress_unit=epoch requires num_epochs')
        total = int(epochs)
    else:
        total = int(steps)
    return ProgressBudget(
        unit=unit,
        num_epochs=epochs,
        num_steps=max(int(steps), 1),
        total=max(total, 1),
        steps_per_epoch=inferred_spe,
        steps_from_epochs=from_epochs,
    )


def due_period(period: int, current: int) -> bool:
    """True when ``current`` hits a positive period. ``period <= 0`` is never due in-loop."""
    return period > 0 and current > 0 and current % period == 0


due_eval_period = due_period


def parse_checkpoint_mode(raw: Any) -> str:
    mode = CHECKPOINT_LATEST if raw is None else str(raw).lower().replace('-', '_')
    if mode not in (CHECKPOINT_LATEST, CHECKPOINT_PERCENT):
        raise ValueError(f'checkpoint must be latest or percent, got {raw!r}')
    return mode


def parse_percents(raw: Any) -> tuple[float, ...]:
    if raw is None:
        return DEFAULT_PERCENTS
    values = tuple(float(x) for x in raw)
    if not values:
        return DEFAULT_PERCENTS
    for value in values:
        if value <= 0 or value > 1:
            raise ValueError(f'checkpoint_percents must be in (0, 1], got {value}')
    return values


def snapshot_name(unit: str, current: int) -> str:
    if unit == UNIT_STEP:
        return f'step_{int(current):06d}'
    return f'epoch_{int(current):04d}'


def crossed_percents(
    current: int,
    total: int,
    percents: Iterable[float],
    already: set[float],
) -> list[float]:
    if total <= 0 or current <= 0:
        return []
    ratio = current / total
    hit: list[float] = []
    for percent in percents:
        if percent in already:
            continue
        if ratio + 1e-12 >= percent:
            hit.append(percent)
    return hit


def checkpoint_names(
    *,
    mode: str,
    save_best: bool,
    period: int,
    percents: Iterable[float],
    current: int,
    total: int,
    unit: str,
    improved: bool,
    is_last: bool,
    already_percent: set[float] | None = None,
) -> list[str]:
    """Return checkpoint file stems to write this tick (order: latest, snapshot, best)."""
    already = already_percent if already_percent is not None else set()
    named = (
        [snapshot_name(unit, current) for _ in crossed_percents(current, total, percents, already)]
        if mode == CHECKPOINT_PERCENT
        else []
    )
    write_latest = is_last or due_period(period, current) or bool(named)
    names: list[str] = []
    if write_latest:
        names.append('latest')
    names.extend(named)
    if save_best and improved:
        names.append('best')
    return list(dict.fromkeys(names))
