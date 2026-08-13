"""Runtime context passed through Flow phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rpipe.artifact.layout import ArtifactLayout
from rpipe.structure.control import Control


@dataclass
class FlowContext:
    experiment_dir: Path
    layout: ArtifactLayout
    config: dict[str, Any]
    control: Control | None = None
    state: dict[str, Any] = field(default_factory=dict)
