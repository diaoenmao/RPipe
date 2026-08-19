"""Control: ExperimentConfig / RunConfig / Control + merge / hash id / contract."""

from __future__ import annotations

from rpipe.structure.control.contract import (
    ContractError,
    validate_control,
    validate_run_config,
)
from rpipe.structure.control.control import (
    Control,
    control_from_config,
    control_from_run_config,
    control_to_config,
)
from rpipe.structure.control.hashing import canonical_json, compute_run_id
from rpipe.structure.control.layers import (
    AlgorithmConfig,
    DataConfig,
    ModelConfig,
    SystemConfig,
)
from rpipe.structure.control.merge import deep_merge
from rpipe.structure.control.run_config import (
    ExperimentConfig,
    RunConfig,
    experiment_config_from_mapping,
    run_config_from_merge,
    run_config_to_mapping,
)

__all__ = [
    'AlgorithmConfig',
    'ContractError',
    'Control',
    'DataConfig',
    'ExperimentConfig',
    'ModelConfig',
    'RunConfig',
    'SystemConfig',
    'canonical_json',
    'compute_run_id',
    'control_from_config',
    'control_from_run_config',
    'control_to_config',
    'deep_merge',
    'experiment_config_from_mapping',
    'run_config_from_merge',
    'run_config_to_mapping',
    'validate_control',
    'validate_run_config',
]
