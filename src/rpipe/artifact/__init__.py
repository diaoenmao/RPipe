"""Artifact IO: Config / Result / Asset paths and helpers."""

from rpipe.artifact.asset import ensure_assets
from rpipe.artifact.config import load_config, write_config
from rpipe.artifact.result import load_result, validate_result, write_result
from rpipe.artifact.layout import ArtifactLayout, artifact_layout, ensure_study_layout
from rpipe.artifact.index import build_index, index_path, load_index, write_index

__all__ = [
    'ArtifactLayout',
    'artifact_layout',
    'ensure_study_layout',
    'ensure_assets',
    'load_config',
    'write_config',
    'load_result',
    'write_result',
    'validate_result',
    'build_index',
    'write_index',
    'load_index',
    'index_path',
]
