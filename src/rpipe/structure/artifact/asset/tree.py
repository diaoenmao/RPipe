"""Create and list files under a Run assets directory."""

from __future__ import annotations

from pathlib import Path

from rpipe.structure.artifact.asset import kinds
from rpipe.structure.artifact.layout import ArtifactLayout


def list_asset_files(assets_dir: Path | str) -> list[str]:
    root = Path(assets_dir)
    if not root.is_dir():
        return []
    files = [p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()]
    return sorted(files)


def ensure_assets(layout: ArtifactLayout) -> Path:
    layout.assets_dir.mkdir(parents=True, exist_ok=True)
    (layout.assets_dir / kinds.TRACKER).mkdir(parents=True, exist_ok=True)
    (layout.assets_dir / kinds.LOGS).mkdir(parents=True, exist_ok=True)
    (layout.assets_dir / kinds.CHECKPOINTS).mkdir(parents=True, exist_ok=True)
    return layout.assets_dir
