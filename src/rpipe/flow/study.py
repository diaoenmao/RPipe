"""Invoke optional Study stages in isolated package namespaces."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from uuid import uuid4

from rpipe.structure.make.recipe import STUDY_PHASES, study_phase_files, study_phases_enabled


def run_study_phase(study_dir: Path | str, phase: str, ctx: Any, study: dict[str, Any]) -> bool:
    if phase not in STUDY_PHASES:
        raise ValueError(f'unknown Study phase: {phase}')
    if not study_phases_enabled(study):
        return False
    root = Path(study_dir).resolve()
    study_phase_files(root, study)
    path = root / phase / '__init__.py'
    if not path.is_file():
        return False
    name = f'_rpipe_study_{uuid4().hex}'
    package = ModuleType(name)
    package.__path__ = [str(root)]
    package.__package__ = name
    sys.modules[name] = package
    try:
        spec = importlib.util.spec_from_file_location(
            f'{name}.{phase}', path, submodule_search_locations=[str(path.parent)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f'cannot load Study phase: {path}')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        hook = getattr(module, 'run', None)
        if not callable(hook):
            raise AttributeError(f'Study phase must define run(ctx): {path}')
        hook(ctx)
    finally:
        for key in list(sys.modules):
            if key == name or key.startswith(name + '.'):
                del sys.modules[key]
    return True
