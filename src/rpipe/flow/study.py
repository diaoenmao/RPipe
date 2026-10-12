"""Invoke optional Study stages in isolated package namespaces."""

from __future__ import annotations

from importlib import import_module
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from uuid import uuid4

from rpipe.structure.make.recipe import STUDY_PHASES, study_phase_files, study_phases_enabled


def run_study_phase(study_dir: Path | str, phase: str, ctx: Any, study: dict[str, Any], *, entry: str = 'run') -> bool:
    if entry not in ('run', 'before') or (entry == 'before' and phase != 'prepare'):
        raise ValueError(f'unsupported Study phase entry: {phase}.{entry}')
    if phase not in STUDY_PHASES:
        raise ValueError(f'unknown Study phase: {phase}')
    if not study_phases_enabled(study):
        return False
    root = Path(study_dir).resolve()
    study_phase_files(root, study)
    path = root / 'flow' / phase / '__init__.py'
    if not path.is_file():
        return False
    name = f'_rpipe_study_{uuid4().hex}'
    package = ModuleType(name)
    package.__path__ = [str(root)]
    package.__package__ = name
    sys.modules[name] = package
    try:
        module = import_module(f'{name}.flow.{phase}')
        hook = getattr(module, entry, None)
        if entry == 'before' and hook is None:
            return False
        if not callable(hook):
            raise AttributeError(f'Study phase must define {entry}(ctx): {path}')
        hook(ctx)
    finally:
        for key in list(sys.modules):
            if key == name or key.startswith(name + '.'):
                del sys.modules[key]
    return True
