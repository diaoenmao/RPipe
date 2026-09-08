"""Write Run configs and the Study index."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rpipe.structure.artifact import (
    artifact_layout,
    build_index,
    experiment_entries,
    load_config,
    write_config,
    write_index,
)
from rpipe.structure.control import experiment_config_from_mapping, run_config_from_merge
from rpipe.structure.make.expand import axis_keys_from_study


def write_run_configs(study_dir: Path, patches: list[dict[str, Any]]) -> list[Path]:
    base_path = study_dir / 'experiment_config.yaml'
    if not base_path.is_file():
        raise FileNotFoundError(f'missing experiment_config.yaml: {base_path}')
    base = experiment_config_from_mapping(load_config(base_path))
    written: list[Path] = []
    for patch in patches:
        run = run_config_from_merge(base, patch)
        cfg = run.to_mapping()
        layout = artifact_layout(study_dir, run.id)
        written.append(write_config(layout.config_path, cfg))
    return written


def write_study_index(
    study_dir: Path,
    study: dict[str, Any],
    config_paths: list[Path],
) -> Path:
    loaded = [(path, load_config(path)) for path in config_paths]
    experiments = experiment_entries(
        configs=loaded,
        axis_keys=axis_keys_from_study(study),
        study_dir=study_dir,
    )
    index = build_index(
        study=str(study.get('study') or study_dir.name),
        description=str(study.get('description') or ''),
        experiments=experiments,
    )
    return write_index(study_dir, index)
