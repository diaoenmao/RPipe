"""Study Artifact directory layout (shared + one Run subtree)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rpipe.structure.artifact import paths


@dataclass(frozen=True)
class ArtifactLayout:
    """Paths for one Run under ``<study>/runs/<run_dir>/``."""

    root: Path
    study_dir: Path

    @property
    def config_path(self) -> Path:
        return self.root / paths.CONFIG_NAME

    @property
    def result_path(self) -> Path:
        return self.root / paths.RESULT_NAME

    @property
    def assets_dir(self) -> Path:
        return self.root / paths.ASSETS_DIRNAME

    @property
    def shared_dir(self) -> Path:
        return self.study_dir / paths.SHARED_DIRNAME

    @property
    def shared_data_dir(self) -> Path:
        return self.shared_dir / 'data'

    @property
    def shared_model_dir(self) -> Path:
        return self.shared_dir / 'model'

    @property
    def docs_dir(self) -> Path:
        return self.study_dir / paths.DOCS_DIRNAME

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.shared_data_dir.mkdir(parents=True, exist_ok=True)
        self.shared_model_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)


def ensure_study_layout(study_dir: Path | str) -> Path:
    """Ensure Study skeleton: ``docs/``, ``shared/{data,model}/``, ``runs/``."""
    study = Path(study_dir)
    (study / paths.DOCS_DIRNAME).mkdir(parents=True, exist_ok=True)
    (study / paths.SHARED_DIRNAME / 'data').mkdir(parents=True, exist_ok=True)
    (study / paths.SHARED_DIRNAME / 'model').mkdir(parents=True, exist_ok=True)
    (study / paths.RUNS_DIRNAME).mkdir(parents=True, exist_ok=True)
    return study


def make_run_dir(run_id: str, timestamp: str | None = None) -> str:
    if timestamp:
        return f'{run_id}_{timestamp}'
    return run_id


def artifact_layout(study_dir: Path | str, run_dir: str) -> ArtifactLayout:
    """Build layout under ``study_dir/runs/<run_dir>/``.

    ``run_dir`` is typically Config ``id`` or ``id_<timestamp>``.
    """
    study = ensure_study_layout(study_dir)
    layout = ArtifactLayout(root=study / paths.RUNS_DIRNAME / run_dir, study_dir=study)
    layout.ensure()
    return layout
