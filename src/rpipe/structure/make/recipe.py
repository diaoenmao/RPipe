"""Load a Study recipe and call its ``register(ctx)`` before Data / Model are built."""

from __future__ import annotations

import builtins
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from threading import RLock
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
    flow = root / 'flow'
    if flow.exists() and (not flow.is_dir() or not flow.resolve().is_relative_to(root)):
        raise ValueError(f'Study flow must be a directory inside the Study: {flow}')
    files = []
    for phase in STUDY_PHASES:
        if (root / phase / '__init__.py').is_file():
            raise ValueError(f'move legacy Study phase to flow/{phase}/: {root / phase}')
        directory = flow / phase
        if not directory.exists():
            continue
        if not directory.is_dir() or not directory.resolve().is_relative_to(root):
            raise ValueError(f'Study phase must be a directory inside the Study: {directory}')
        if not (directory / '__init__.py').is_file():
            continue
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


_recipe_import_lock = RLock()


class _RecipeLoader(importlib.machinery.SourceFileLoader):
    def exec_module(self, module):
        package = sys.modules[module.__name__.partition('.')[0]]
        module.__dict__['__builtins__'] = dict(vars(builtins), __import__=package._recipe_import)
        super().exec_module(module)


class _RecipeFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if not fullname.startswith('_rpipe_recipe_'):
            return None
        package = sys.modules.get(fullname.partition('.')[0])
        if package is None or not getattr(package, '_recipe_namespace', False):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and isinstance(spec.loader, importlib.machinery.SourceFileLoader):
            spec.loader = _RecipeLoader(fullname, spec.origin)
        return spec


if not any(type(finder).__module__ == __name__ and type(finder).__name__ == '_RecipeFinder'
           for finder in sys.meta_path):
    sys.meta_path.insert(0, _RecipeFinder())


def _recipe_namespace(root: Path) -> str:
    return '_rpipe_recipe_' + hashlib.sha256(os.path.normcase(str(root.resolve())).encode()).hexdigest()


def _local_module(package: ModuleType, name: str) -> bool:
    top = name.partition('.')[0]
    qualified = package.__name__ + '.' + top
    return (top not in sys.stdlib_module_names
            and (qualified in sys.modules
                 or importlib.machinery.PathFinder.find_spec(qualified, package.__path__) is not None))


def _recipe_package(root: Path, recipe_root: Path) -> ModuleType:
    name = _recipe_namespace(root)
    package = sys.modules.get(name)
    if package is None:
        package = ModuleType(name)
        package.__package__ = name
        package.__path__ = []
        package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        package.__spec__.submodule_search_locations = package.__path__
        package._recipe_namespace = True

        def local_import(module_name, globals=None, locals=None, fromlist=(), level=0):
            if level == 0 and _local_module(package, module_name):
                imported = builtins.__import__(name + '.' + module_name, globals, locals, fromlist, 0)
                return imported if fromlist else sys.modules[name + '.' + module_name.partition('.')[0]]
            return builtins.__import__(module_name, globals, locals, fromlist, level)

        package._recipe_import = local_import
        sys.modules[name] = package
    package.__path__[:] = list(dict.fromkeys([str(recipe_root), str(root)]))
    return package


def _recipe_pickle_module(study_dir: Path | str) -> ModuleType | None:
    package = sys.modules.get(_recipe_namespace(Path(study_dir)))
    if package is None or not getattr(package, '_recipe_namespace', False):
        return None

    class RecipeUnpickler(pickle.Unpickler):
        def find_class(self, module_name, name):
            prefix, separator, local = module_name.partition('.')
            private = (separator and prefix.startswith('_rpipe_recipe_')
                       and len(prefix) == len('_rpipe_recipe_') + 64
                       and all(char in '0123456789abcdef' for char in prefix[len('_rpipe_recipe_'):]))
            qualified = package.__name__ + '.' + (local if private else module_name)
            if not private:
                owned_top = package.__name__ + '.' + prefix
                if ((prefix in sys.modules and owned_top not in sys.modules)
                        or not _local_module(package, module_name)):
                    return super().find_class(module_name, name)
                try:
                    spec = importlib.util.find_spec(qualified)
                except (ModuleNotFoundError, AttributeError, ValueError):
                    spec = None
                if spec is None:
                    return super().find_class(module_name, name)
            value = importlib.import_module(qualified)
            try:
                for part in name.split('.'):
                    value = getattr(value, part)
            except AttributeError:
                if private:
                    raise
                return super().find_class(module_name, name)
            return value

    reader = ModuleType('_rpipe_recipe_pickle')
    reader.__dict__.update({name: value for name, value in vars(pickle).items() if name != '__name__'})
    reader.Unpickler = RecipeUnpickler
    return reader


def load_recipe(study_dir: Path | str, study: dict[str, Any]) -> ModuleType | None:
    path = recipe_path(study_dir, study)
    if path is None:
        return None
    root = Path(study_dir).resolve()
    with _recipe_import_lock:
        package = _recipe_package(root, path.parent)
        name = package.__name__ + '.<recipe>'
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f'cannot load recipe: {path}')
        module = importlib.util.module_from_spec(spec)
        previous, existed = sys.modules.get(name), name in sys.modules
        sys.modules[name] = module
        module.__dict__['__builtins__'] = dict(vars(builtins), __import__=package._recipe_import)
        try:
            spec.loader.exec_module(module)
            if not callable(getattr(module, 'register', None)):
                raise AttributeError(f'recipe must define register(ctx): {path}')
        except BaseException:
            if existed:
                sys.modules[name] = previous
            else:
                sys.modules.pop(name, None)
            raise
        return module


def apply_recipe(study_dir: Path | str, study: dict[str, Any], ctx: RecipeContext) -> bool:
    """Return True when a recipe was registered for this Run."""
    module = load_recipe(study_dir, study)
    if module is None:
        return False
    with _recipe_import_lock:
        package = sys.modules[module.__package__]
        package.__path__[:] = list(dict.fromkeys([str(Path(study_dir).resolve()), *package.__path__]))
        module.register(ctx)
    return True
