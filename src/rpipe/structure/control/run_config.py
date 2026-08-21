"""ExperimentConfig / RunConfig and merge + id hashing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rpipe.structure.control.hashing import compute_run_id
from rpipe.structure.control.layers import (
    AlgorithmConfig,
    DataConfig,
    ModelConfig,
    SystemConfig,
)
from rpipe.structure.control.merge import deep_merge

# Legacy keys that must not affect run id / are not RunConfig fields.
_IGNORE_TOP = frozenset({'slug', 'control_slug', 'raw'})
_LAYER_KEYS = frozenset(
    {'seed', 'experiment', 'description', 'tags', 'data', 'model', 'algorithm', 'system'}
)


def _normalize_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise TypeError(f'tags must be a list of strings, got {type(value)!r}')


@dataclass
class ExperimentConfig:
    """Experiment-level base config (no run id)."""

    seed: int | None = None
    experiment: str | None = None
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> ExperimentConfig:
        body = {k: v for k, v in dict(mapping or {}).items() if k not in _IGNORE_TOP and k != 'id'}
        extras = {k: body.pop(k) for k in list(body) if k not in _LAYER_KEYS}
        return cls(
            seed=body.get('seed'),
            experiment=body.get('experiment'),
            description=body.get('description'),
            tags=_normalize_tags(body.get('tags')),
            data=DataConfig.from_mapping(body.get('data') or {}),
            model=ModelConfig.from_mapping(body.get('model') or {}),
            algorithm=AlgorithmConfig.from_mapping(body.get('algorithm') or {}),
            system=SystemConfig.from_mapping(body.get('system') or {}),
            extras=extras,
        )

    def to_mapping(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.seed is not None:
            out['seed'] = self.seed
        if self.experiment is not None:
            out['experiment'] = self.experiment
        if self.description is not None:
            out['description'] = self.description
        if self.tags:
            out['tags'] = list(self.tags)
        out['data'] = self.data.to_mapping()
        out['model'] = self.model.to_mapping()
        out['algorithm'] = self.algorithm.to_mapping()
        out['system'] = self.system.to_mapping()
        out.update(self.extras)
        return out


@dataclass
class RunConfig:
    """Run-level config; id is content hash of fields except id/description/tags."""

    id: str = ''
    seed: int | None = None
    experiment: str | None = None
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None, *, assign_id: bool = True) -> RunConfig:
        body = {k: v for k, v in dict(mapping or {}).items() if k not in _IGNORE_TOP}
        run_id = str(body.pop('id')) if body.get('id') not in (None, '') else ''
        extras = {k: body.pop(k) for k in list(body) if k not in _LAYER_KEYS}
        run = cls(
            id=run_id,
            seed=body.get('seed'),
            experiment=body.get('experiment'),
            description=body.get('description'),
            tags=_normalize_tags(body.get('tags')),
            data=DataConfig.from_mapping(body.get('data') or {}),
            model=ModelConfig.from_mapping(body.get('model') or {}),
            algorithm=AlgorithmConfig.from_mapping(body.get('algorithm') or {}),
            system=SystemConfig.from_mapping(body.get('system') or {}),
            extras=extras,
        )
        if assign_id and not run.id:
            run.id = compute_run_id(run.to_mapping())
        return run

    def to_mapping(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.id:
            out['id'] = self.id
        if self.seed is not None:
            out['seed'] = self.seed
        if self.experiment is not None:
            out['experiment'] = self.experiment
        if self.description is not None:
            out['description'] = self.description
        if self.tags:
            out['tags'] = list(self.tags)
        out['data'] = self.data.to_mapping()
        out['model'] = self.model.to_mapping()
        out['algorithm'] = self.algorithm.to_mapping()
        out['system'] = self.system.to_mapping()
        out.update(self.extras)
        return out

    def with_computed_id(self) -> RunConfig:
        mapping = self.to_mapping()
        mapping.pop('id', None)
        return RunConfig.from_mapping(mapping, assign_id=True)


def experiment_config_from_mapping(mapping: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig.from_mapping(mapping)


def run_config_from_merge(
    experiment: ExperimentConfig | dict[str, Any],
    patch: dict[str, Any] | None = None,
) -> RunConfig:
    """Merge experiment base ⊕ patch, then hash id."""
    base = (
        experiment.to_mapping()
        if isinstance(experiment, ExperimentConfig)
        else ExperimentConfig.from_mapping(experiment).to_mapping()
    )
    merged = deep_merge(base, dict(patch or {}))
    merged.pop('id', None)
    return RunConfig.from_mapping(merged, assign_id=True)


def run_config_to_mapping(run: RunConfig) -> dict[str, Any]:
    return run.to_mapping()
