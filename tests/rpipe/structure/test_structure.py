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
