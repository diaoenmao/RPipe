"""Study Artifact directory layout (shared + one Run subtree)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactLayout:
    """Paths for one Run under ``<study>/runs/<run_dir>/``."""

    root: Path
    study_dir: Path

    @property
    def config_path(self) -> Path:
        return self.root / 'config.yaml'

    @property
    def result_path(self) -> Path:
        return self.root / 'result.json'

    @property
    def assets_dir(self) -> Path:
        return self.root / 'assets'

    @property
    def shared_dir(self) -> Path:
        return self.study_dir / 'shared'

    @property
    def shared_data_dir(self) -> Path:
        return self.shared_dir / 'data'

    @property
    def shared_model_dir(self) -> Path:
        return self.shared_dir / 'model'

    @property
    def docs_dir(self) -> Path:
        return self.study_dir / 'docs'

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.shared_data_dir.mkdir(parents=True, exist_ok=True)
        self.shared_model_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)


def ensure_study_layout(study_dir: Path | str) -> Path:
    """Ensure Study skeleton: ``docs/``, ``shared/{data,model}/``, ``runs/``."""
    study = Path(study_dir)
    (study / 'docs').mkdir(parents=True, exist_ok=True)
    (study / 'shared' / 'data').mkdir(parents=True, exist_ok=True)
    (study / 'shared' / 'model').mkdir(parents=True, exist_ok=True)
    (study / 'runs').mkdir(parents=True, exist_ok=True)
    return study


def artifact_layout(study_dir: Path | str, run_dir: str) -> ArtifactLayout:
    """Build layout under ``study_dir/runs/<run_dir>/``.

    ``run_dir`` is typically Config ``id`` or ``id_<timestamp>``.
    Also ensures ``shared/{data,model}/`` and ``docs/``.
    """
    study = ensure_study_layout(study_dir)
    layout = ArtifactLayout(root=study / 'runs' / run_dir, study_dir=study)
    layout.ensure()
    return layout
