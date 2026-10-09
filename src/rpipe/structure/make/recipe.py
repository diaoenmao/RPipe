"""Load a Study recipe and call its ``register(ctx)`` before Data / Model are built."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

STUDY_PHASES = ('prepare', 'execute', 'collect', 'summarize', 'write', 'process')


def study_phases_enabled(study: dict[str, Any]) -> bool:
    block = study.get('flow', {})
    if not isinstance(block, dict):
        raise ValueError('flow must be a mapping')
    enabled = block.get('study_phases', False)
    if not isinstance(enabled, bool):
        raise ValueError('flow.study_phases must be a boolean')
    return enabled


def study_phase_files(study_dir: Path | str, study: dict[str, Any]) -> list[Path]:
    if not study_phases_enabled(study):
        return []
    root = Path(study_dir).resolve()
    files = []
    for phase in STUDY_PHASES:
        directory = root / phase
        if not directory.exists():
            continue
        if not directory.is_dir() or not (directory / '__init__.py').is_file():
            raise ValueError(f'Study phase must be a package: {directory}')
        for path in sorted(directory.rglob('*.py')):
            if not path.resolve().is_relative_to(root):
                raise ValueError(f'Study phase source must stay inside the Study: {path}')
            files.append(path)
    return files


@dataclass(frozen=True)
class RecipeContext:
    study_dir: Path
    run_id: str
    seed: int
    config: dict[str, Any] = field(repr=False)
    shared_data_dir: Path
    shared_model_dir: Path
    assets_dir: Path


def recipe_path(study_dir: Path | str, study: dict[str, Any]) -> Path | None:
    raw = study.get('recipe')
    if raw in (None, ''):
        return None
    root = Path(study_dir).resolve()
    path = (root / str(raw)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f'recipe must stay inside the Study: {raw}') from exc
    if not path.is_file():
        raise FileNotFoundError(f'missing recipe: {path}')
    return path


def load_recipe(study_dir: Path | str, study: dict[str, Any]) -> ModuleType | None:
    path = recipe_path(study_dir, study)
    if path is None:
        return None
    root = str(path.parent)
    name = f'rpipe_recipe_{Path(study_dir).resolve().name}'
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load recipe: {path}')
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, root)
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(root)
        except ValueError:
            pass
    if not callable(getattr(module, 'register', None)):
        raise AttributeError(f'recipe must define register(ctx): {path}')
    return module


def apply_recipe(study_dir: Path | str, study: dict[str, Any], ctx: RecipeContext) -> bool:
    """Return True when a recipe was registered for this Run."""
    module = load_recipe(study_dir, study)
    if module is None:
        return False
    root = str(Path(study_dir).resolve())
    sys.path.insert(0, root)
    try:
        module.register(ctx)
    finally:
        try:
            sys.path.remove(root)
        except ValueError:
            pass
    return True
