"""Structure constraints from docs/code/structure.md §2.1."""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3] / 'src/rpipe/structure'
pytestmark = [pytest.mark.unit, pytest.mark.location, pytest.mark.p1,
              pytest.mark.structure_layer, pytest.mark.module_api,
              pytest.mark.cost(cost_class='c1'),
              pytest.mark.result_type('categorical', detail='summary')]


@pytest.mark.parametrize('path', sorted(ROOT.rglob('__init__.py')),
                         ids=lambda path: path.relative_to(ROOT).as_posix())
def test_package_initializers_only_export(path):
    for node in ast.parse(path.read_text(encoding='utf-8-sig')).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        assert isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == '__all__', (path, node.lineno, type(node).__name__)
        names = ast.literal_eval(node.value)
        assert isinstance(names, (list, tuple)) and all(isinstance(name, str) for name in names), path


def test_every_structure_file_is_registered():
    document = (ROOT.parents[2] / 'docs/code/structure.md').read_text(encoding='utf-8-sig')
    registered = re.findall(r'^\| `([^`]+\.py)` \|', document, re.MULTILINE)
    actual = {path.relative_to(ROOT).as_posix() for path in ROOT.rglob('*.py') if path.name != '__init__.py'}
    assert len(registered) == len(set(registered)), 'duplicate file registration'
    assert set(registered) == actual


@pytest.mark.parametrize('name', ['artifact_api', 'make_api'])
def test_new_facades_are_pure_reexports(name):
    import importlib
    module = importlib.import_module('rpipe.structure.api.' + name)
    for node in ast.parse(Path(module.__file__).read_text(encoding='utf-8-sig')).body:
        if isinstance(node, ast.Assign):
            assert len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == '__all__'
            assert isinstance(ast.literal_eval(node.value), list)
        else:
            assert isinstance(node, (ast.Import, ast.ImportFrom)), (name, type(node).__name__)
        if isinstance(node, ast.ImportFrom):
            original = importlib.import_module(node.module)
            for alias in node.names:
                assert getattr(module, alias.asname or alias.name) is getattr(original, alias.name)


def test_structure_has_exactly_eight_packages():
    assert {p.name for p in ROOT.iterdir() if p.name != '__pycache__'} == {
        '__init__.py', 'api', 'control', 'data', 'model', 'algorithm', 'system', 'artifact', 'make'}


STUDIES = ROOT.parents[2] / 'studies'
CURRENT_STUDY_CODE = sorted(p for study in STUDIES.iterdir() if study.is_dir()
                           for area in ('flow', 'structure') for p in (study / area).rglob('*.py'))
CURRENT_STUDY_CODE += sorted(STUDIES.glob('*/recipe.py'))


@pytest.mark.parametrize('path', sorted(ROOT.parent.rglob('*.py')) + CURRENT_STUDY_CODE,
                         ids=lambda path: path.relative_to(ROOT.parents[2]).as_posix())
def test_cross_layer_imports_use_facades(path):
    parts = path.relative_to(ROOT.parent).parts if path.is_relative_to(ROOT.parent) else ()
    layer = parts[1] if len(parts) > 2 and parts[0] == 'structure' else None
    if layer in ('api', 'control'):
        return
    from importlib.util import resolve_name
    from rpipe.structure.control import __all__ as control_exports
    package = ('rpipe.' + '.'.join(path.parent.relative_to(ROOT.parent).parts)).rstrip('.') if parts else '.'.join(path.parent.relative_to(ROOT.parents[2]).parts)
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
        if isinstance(node, ast.ImportFrom):
            module = resolve_name('.' * node.level + (node.module or ''), package) if node.level else node.module
            modules = [module + '.' + a.name for a in node.names] if module == 'rpipe.structure' else [module]
        else:
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else []
        for module in modules:
            if module and module.startswith('rpipe.structure.'):
                target = module.split('.')[2]
                assert target in (layer, 'api', 'control') or (module.rsplit('.', 1)[0] == 'rpipe.structure' and target in control_exports), (path, node.lineno, module)


def test_origin_builders_keep_existing_public_names():
    from rpipe.structure.api import data_api, model_api, system_api
    from rpipe.structure.system import origin
    assert data_api.load_hf_dataset is system_api.load_hf_dataset is origin.load_hf_dataset
    assert model_api.download_model is system_api.download_model is origin.download_model
