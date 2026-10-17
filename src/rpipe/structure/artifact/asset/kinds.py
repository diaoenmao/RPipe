"""Relative asset paths under ``runs/<id>/assets/``."""

from __future__ import annotations

from pathlib import Path
from rpipe.structure.artifact.layout import ArtifactLayout

TRACKER = 'tracker'
TRACKER_STATE = 'tracker/tracker_state.json'
TRACKER_JSONL = 'tracker/scalars.jsonl'
LOGS = 'logs'
RUN_LOG = 'logs/run.log'
CHECKPOINTS = 'checkpoints'
SAMPLES = 'samples'
CACHE = 'cache'
WEIGHTS = 'weights'


def list_asset_files(assets_dir: Path | str) -> list[str]:
    root = Path(assets_dir)
    if not root.is_dir():
        return []
    files = [p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()]
    return sorted(files)

def ensure_assets(layout: ArtifactLayout) -> Path:
    layout.assets_dir.mkdir(parents=True, exist_ok=True)
    (layout.assets_dir / TRACKER).mkdir(parents=True, exist_ok=True)
    (layout.assets_dir / LOGS).mkdir(parents=True, exist_ok=True)
    (layout.assets_dir / CHECKPOINTS).mkdir(parents=True, exist_ok=True)
    return layout.assets_dir
