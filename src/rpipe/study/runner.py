"""Study runner: study.yaml → Configs + index.json → Flow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from rpipe.artifact import (
    artifact_layout,
    build_index,
    load_config,
    write_config,
    write_index,
)
from rpipe.artifact.layout import ensure_study_layout
from rpipe.flow import FlowContext, FlowRunner
from rpipe.structure.control import experiment_config_from_mapping, run_config_from_merge
from rpipe.study.expand import expand_patches


def load_study_yaml(study_dir: Path) -> dict[str, Any]:
    path = study_dir / 'study.yaml'
    if not path.is_file():
        raise FileNotFoundError(f'missing study.yaml: {path}')
    data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    if not isinstance(data, dict):
        raise ValueError(f'study.yaml must be a mapping: {path}')
    return data


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
    base = load_config(study_dir / 'experiment_config.yaml')
    exp_meta = study.get('experiment') if isinstance(study.get('experiment'), dict) else {}
    name = (exp_meta or {}).get('name') or base.get('experiment') or study_dir.name
    runs = []
    for path in config_paths:
        cfg = load_config(path)
        runs.append(
            {
                'id': cfg.get('id'),
                'description': cfg.get('description'),
                'tags': cfg.get('tags') or [],
                'run_dir': path.parent.name,
                'config': str(path),
            }
        )
    index = build_index(
        study=str(study.get('study') or study_dir.name),
        description=str(study.get('description') or ''),
        experiments=[
            {
                'name': name,
                'description': base.get('description') or '',
                'path': str(study_dir),
                'runs': runs,
            }
        ],
    )
    return write_index(study_dir, index)


def launch_runs(
    study_dir: Path,
    config_paths: list[Path],
    phases: list[str] | None = None,
) -> list[Path]:
    results: list[Path] = []
    runner = FlowRunner(phases=phases)
    for path in config_paths:
        run_dir = path.parent.name
        layout = artifact_layout(study_dir, run_dir)
        cfg = load_config(layout.config_path)
        ctx = FlowContext(study_dir=study_dir, layout=layout, config=cfg)
        results.append(runner.run(ctx))
    return results


def run_study(
    study_dir: Path | str,
    *,
    skip_launch: bool = False,
    phases: list[str] | None = None,
) -> dict[str, Any]:
    """Expand → write index → launch. One entry for humans and CLI."""
    study_dir = ensure_study_layout(Path(study_dir).resolve())
    study = load_study_yaml(study_dir)
    patches = expand_patches(study)
    configs = write_run_configs(study_dir, patches)
    index_path = write_study_index(study_dir, study, configs)
    result_paths: list[Path] = []
    if not skip_launch:
        result_paths = launch_runs(study_dir, configs, phases=phases)
    return {
        'study_dir': study_dir,
        'index': index_path,
        'configs': configs,
        'results': result_paths,
    }
