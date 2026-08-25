"""Asset path helpers."""

from __future__ import annotations

from pathlib import Path

from rpipe.structure.artifact.layout import ArtifactLayout


def ensure_assets(layout: ArtifactLayout) -> Path:
    layout.assets_dir.mkdir(parents=True, exist_ok=True)
    return layout.assets_dir


def write_text_asset(layout: ArtifactLayout, name: str, text: str) -> Path:
    path = ensure_assets(layout) / name
    path.write_text(text, encoding='utf-8')
    return path
