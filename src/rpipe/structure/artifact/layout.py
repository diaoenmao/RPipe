"""Study Artifact directory layout (shared + one Run subtree)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rpipe.structure.artifact.shared import shared_paths


CONFIG_NAME = 'config.yaml'

RESULT_NAME = 'result.json'

INDEX_NAME = 'index.json'

PROCESS_NAME = 'process.json'

DERIVED_NAME = 'derived.json'

ASSETS_DIRNAME = 'assets'

SHARED_DIRNAME = 'shared'

DOCS_DIRNAME = 'docs'

RUNS_DIRNAME = 'runs'

class ArtifactError(Exception):
    """Base error for artifact path / IO failures."""

class MissingConfigError(ArtifactError, FileNotFoundError):
    """Run config file is missing."""

class CorruptArtifactError(ArtifactError):
    """Config / result / index is not a mapping or cannot be parsed."""


def validate_run_directory(run_dir: str) -> None:
    reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {f'{prefix}{n}' for prefix in ('COM', 'LPT') for n in range(1, 10)}
    if (not isinstance(run_dir, str) or not run_dir or run_dir in ('.', '..')
            or run_dir != run_dir.strip() or run_dir.endswith('.')
            or any(char in run_dir for char in '/\\:<>"|?*') or any(ord(char) < 32 for char in run_dir)
            or run_dir.split('.')[0].upper() in reserved):
        raise ValueError(f'invalid Run directory: {run_dir!r}')


@dataclass(frozen=True)
class ArtifactLayout:
    """Paths for one Run under ``<study>/runs/<run_dir>/``."""

    root: Path
    study_dir: Path

    @property
    def config_path(self) -> Path:
        return self.root / CONFIG_NAME

    @property
    def result_path(self) -> Path:
        return self.root / RESULT_NAME

    @property
    def assets_dir(self) -> Path:
        return self.root / ASSETS_DIRNAME

    @property
    def shared_dir(self) -> Path:
        return shared_paths(self.study_dir)['root']

    @property
    def shared_data_dir(self) -> Path:
        return shared_paths(self.study_dir)['data']

    @property
    def shared_model_dir(self) -> Path:
        return shared_paths(self.study_dir)['model']

    @property
    def docs_dir(self) -> Path:
        return self.study_dir / DOCS_DIRNAME

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.shared_data_dir.mkdir(parents=True, exist_ok=True)
        self.shared_model_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)


def ensure_study_layout(study_dir: Path | str) -> Path:
    """Ensure Study outputs and project shared input roots."""
    study = Path(study_dir)
    shared = shared_paths(study)
    (study / DOCS_DIRNAME).mkdir(parents=True, exist_ok=True)
    shared['data'].mkdir(parents=True, exist_ok=True)
    shared['model'].mkdir(parents=True, exist_ok=True)
    (study / RUNS_DIRNAME).mkdir(parents=True, exist_ok=True)
    return study


def artifact_layout(study_dir: Path | str, run_dir: str) -> ArtifactLayout:
    """Build layout under ``study_dir/runs/<run_dir>/``.

    ``run_dir`` is the content-derived Config ``id``.
    """
    validate_run_directory(run_dir)
    study = ensure_study_layout(study_dir)
    layout = ArtifactLayout(root=study / RUNS_DIRNAME / run_dir, study_dir=study)
    layout.ensure()
    return layout
