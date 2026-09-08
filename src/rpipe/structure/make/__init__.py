"""Multi-run configs, index, and launch scripts."""

from pathlib import Path
from typing import Any

from rpipe.structure.artifact.layout import ensure_study_layout
from rpipe.structure.make.expand import expand_patches, load_study_yaml
from rpipe.structure.make.schedule import (
    gpu_ids,
    launch_jobs,
    plan_jobs,
    render_bash,
    run_succeeded,
    write_launch_scripts,
)
from rpipe.structure.make.write import write_run_configs, write_study_index

__all__ = [
    'expand_patches',
    'expand_study',
    'gpu_ids',
    'launch_jobs',
    'load_study_yaml',
    'plan_jobs',
    'render_bash',
    'run_succeeded',
    'write_launch_scripts',
    'write_run_configs',
    'write_study_index',
]


def expand_study(study_dir: Path | str) -> dict[str, Any]:
    """Write Run configs and index from ``study.yaml``."""
    study_dir = ensure_study_layout(Path(study_dir).resolve())
    study = load_study_yaml(study_dir)
    patches = expand_patches(study)
    configs = write_run_configs(study_dir, patches)
    index_file = write_study_index(study_dir, study, configs)
    return {
        'study_dir': study_dir,
        'index': index_file,
        'configs': configs,
        'study': study,
    }
