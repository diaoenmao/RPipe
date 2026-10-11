"""Flow package entries export implementations without defining business logic."""

import ast
from importlib import import_module
from pathlib import Path

import pytest
import rpipe.flow

pytestmark = [pytest.mark.unit, pytest.mark.location, pytest.mark.p1,
              pytest.mark.flow_layer, pytest.mark.module_runner,
              pytest.mark.cost(cost_class='c1'), pytest.mark.result_type('categorical', detail='summary')]


@pytest.mark.parametrize('package', ['', 'prepare', 'execute', 'collect', 'summarize', 'write', 'process'])
def test_flow_package_entry_only_contains_exports(package):
    root = Path(rpipe.flow.__file__).parent
    entry = root / package / '__init__.py'
    body = ast.parse(entry.read_text(encoding='utf-8')).body
    for node in body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        assert isinstance(node, ast.Assign) and all(isinstance(target, ast.Name) and target.id == '__all__'
                                                  for target in node.targets), entry
        assert isinstance(node.value, (ast.List, ast.Tuple)) and all(
            isinstance(item, ast.Constant) and isinstance(item.value, str) for item in node.value.elts), entry


@pytest.mark.parametrize('phase', ['prepare', 'execute', 'collect', 'summarize', 'write', 'process'])
def test_stage_package_preserves_callable_import(phase):
    package = import_module(f'rpipe.flow.{phase}')
    implementation = import_module(f'rpipe.flow.{phase}.{"individual" if phase == "process" else "run"}')
    assert callable(package.run)
    assert package.run is implementation.run
