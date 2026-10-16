"""Study-local recipe imports must match fresh-process imports."""

import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

from rpipe.structure.make.recipe import apply_recipe

pytestmark = [pytest.mark.unit, pytest.mark.content, pytest.mark.p1,
              pytest.mark.structure_layer, pytest.mark.module_data,
              pytest.mark.cost(cost_class='c1'),
              pytest.mark.result_type('categorical', detail='summary')]


def _recipe(root, helper, value, package, at_register, fail=False):
    root.mkdir()
    if package:
        directory = root / helper
        directory.mkdir()
        (directory / '__init__.py').write_text('from .value import VALUE\n', encoding='utf-8')
        (directory / 'value.py').write_text(f'VALUE = {value!r}\n', encoding='utf-8')
    else:
        (root / f'{helper}.py').write_text(f'VALUE = {value!r}\n', encoding='utf-8')
    prefix = '' if at_register else f'from {helper} import VALUE\n'
    body = f'    from {helper} import VALUE\n' if at_register else ''
    body += '    ctx.observed.append(VALUE)\n'
    if fail:
        body += '    raise RuntimeError("registration refused")\n'
    (root / 'recipe.py').write_text(prefix + 'def register(ctx):\n' + body, encoding='utf-8')


@pytest.mark.parametrize('package', [False, True], ids=['module', 'package'])
@pytest.mark.parametrize('at_register', [False, True], ids=['import-on-load', 'import-on-register'])
@pytest.mark.parametrize('first_failure', [False, True], ids=['first-success', 'first-error'])
def test_sequential_studies_import_their_own_recipe_helpers(tmp_path, package, at_register, first_failure):
    helper = '_recipe_isolation_' + uuid4().hex
    paths = list(sys.path)
    observed = []
    try:
        for index, value in enumerate(('first', 'second')):
            study = tmp_path / value
            fail = first_failure and index == 0
            _recipe(study, helper, value, package, at_register, fail)
            if fail:
                with pytest.raises(RuntimeError, match='registration refused'):
                    apply_recipe(study, {'recipe': 'recipe.py'}, SimpleNamespace(observed=observed))
            else:
                assert apply_recipe(study, {'recipe': 'recipe.py'}, SimpleNamespace(observed=observed))
            assert sys.path == paths
        assert observed == ['first', 'second']
    finally:
        for name in list(sys.modules):
            if name == helper or name.startswith(helper + '.'):
                sys.modules.pop(name)


@pytest.mark.parametrize('package', [False, True], ids=['module', 'package'])
@pytest.mark.parametrize('at_register', [False, True], ids=['import-on-load', 'import-on-register'])
def test_single_study_recipe_helper_control(tmp_path, package, at_register):
    helper = '_recipe_isolation_' + uuid4().hex
    study = tmp_path / 'single'
    paths = list(sys.path)
    observed = []
    try:
        _recipe(study, helper, 'single', package, at_register)
        assert apply_recipe(study, {'recipe': 'recipe.py'}, SimpleNamespace(observed=observed))
        assert observed == ['single']
        assert sys.path == paths
    finally:
        for name in list(sys.modules):
            if name == helper or name.startswith(helper + '.'):
                sys.modules.pop(name)
