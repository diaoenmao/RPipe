"""Runtime context passed through Flow phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rpipe.artifact.layout import ArtifactLayout
from rpipe.structure.control import Control


@dataclass
class FlowContext:
    """``study_dir`` is the Study root (owns ``docs/``, ``shared/``, ``runs/``)."""

    study_dir: Path
    layout: ArtifactLayout
    config: dict[str, Any]
    control: Control | None = None
    state: dict[str, Any] = field(default_factory=dict)

    @property
    def experiment_dir(self) -> Path:
        """Deprecated alias for ``study_dir`` (recipe lives inside the Study)."""
        return self.study_dir
