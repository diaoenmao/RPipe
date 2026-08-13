"""Artifact directory layout for one Control run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactLayout:
    root: Path

    @property
    def config_path(self) -> Path:
        return self.root / 'config.yaml'

    @property
    def result_path(self) -> Path:
        return self.root / 'result.json'

    @property
    def assets_dir(self) -> Path:
        return self.root / 'assets'

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)


def artifact_layout(experiment_dir: Path | str, slug: str) -> ArtifactLayout:
    root = Path(experiment_dir) / 'artifact' / slug
    layout = ArtifactLayout(root=root)
    layout.ensure()
    return layout
