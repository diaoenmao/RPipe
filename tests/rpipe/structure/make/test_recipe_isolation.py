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


@pytest.mark.parametrize('package', [False, True], ids=['module', 'package'])
@pytest.mark.parametrize('at_register', [False, True], ids=['import-on-load', 'import-on-register'])
def test_local_recipe_import_preserves_preexisting_foreign_module(tmp_path, package, at_register):
    from types import ModuleType
    import json

    helper = '_recipe_isolation_' + uuid4().hex
    previous = ModuleType(helper)
    previous.VALUE = 'foreign'
    sys.modules[helper] = previous
    json_module = sys.modules['json']
    paths = list(sys.path)
    observed = []
    try:
        study = tmp_path / 'local'
        _recipe(study, helper, 'local', package, at_register)
        assert apply_recipe(study, {'recipe': 'recipe.py'}, SimpleNamespace(observed=observed))
        assert observed == ['local']
        assert sys.modules[helper] is previous
        assert sys.modules['json'] is json_module is json
        assert sys.path == paths
    finally:
        for name in list(sys.modules):
            if name == helper or name.startswith(helper + '.'):
                sys.modules.pop(name)


@pytest.mark.parametrize('package', [False, True], ids=['module', 'package'])
def test_registered_functions_keep_each_studies_helper_globals(tmp_path, package):
    helper = '_recipe_isolation_' + uuid4().hex
    registered = []
    try:
        for value in ('first', 'second'):
            study = tmp_path / value
            _recipe(study, helper, value, package, False)
            (study / 'recipe.py').write_text(
                f'from {helper} import VALUE\n'
                'def build():\n    return VALUE\n'
                'def register(ctx):\n    ctx.registered.append(build)\n', encoding='utf-8')
            assert apply_recipe(study, {'recipe': 'recipe.py'}, SimpleNamespace(registered=registered))
        assert [build() for build in registered] == ['first', 'second']
    finally:
        for name in list(sys.modules):
            if name == helper or name.startswith(helper + '.'):
                sys.modules.pop(name)


@pytest.mark.parametrize('nested', [False, True], ids=['root-recipe', 'nested-recipe'])
@pytest.mark.parametrize('alias', [False, True], ids=['direct-dataclass', 'module-alias'])
@pytest.mark.parametrize('postponed', [False, True], ids=['eager-annotations', 'postponed-annotations'])
def test_recipe_dataclass_resolves_annotations_in_defining_module(tmp_path, nested, alias, postponed):
    from dataclasses import fields
    from rpipe.structure.make.recipe import load_recipe

    study = tmp_path / 'dataclass-study'
    folder = study / 'recipes' if nested else study
    folder.mkdir(parents=True)
    prefix = 'from __future__ import annotations\n' if postponed else ''
    prefix += 'import dataclasses as dc\n' if alias else 'from dataclasses import dataclass\n'
    decorator = '@dc.dataclass' if alias else '@dataclass'
    (folder / 'recipe.py').write_text(
        prefix + 'from typing import ClassVar\n' + decorator + '\n'
        'class Payload:\n    value: int\n    label: ClassVar[str] = "tag"\n'
        'def register(ctx):\n    pass\n', encoding='utf-8')
    paths = list(sys.path)
    name = 'rpipe_recipe_' + study.resolve().name
    previous = sys.modules.get(name)
    module = load_recipe(study, {'recipe': ('recipes/' if nested else '') + 'recipe.py'})
    assert module.Payload(7).value == 7
    assert module.Payload.label == 'tag'
    assert [field.name for field in fields(module.Payload)] == ['value']
    assert sys.path == paths
    assert sys.modules.get(name) is previous


@pytest.mark.parametrize('nested', [False, True], ids=['root-recipe', 'nested-recipe'])
@pytest.mark.parametrize('postponed', [False, True], ids=['eager-annotations', 'postponed-annotations'])
def test_dataclass_defined_during_register_has_defining_module(tmp_path, nested, postponed):
    study = tmp_path / 'registration-dataclass'
    folder = study / 'recipes' if nested else study
    folder.mkdir(parents=True)
    prefix = 'from __future__ import annotations\n' if postponed else ''
    (folder / 'recipe.py').write_text(
        prefix + 'from dataclasses import dataclass, fields\nfrom typing import ClassVar\n'
        'def register(ctx):\n'
        '    @dataclass\n    class Payload:\n        value: int\n        label: ClassVar[str] = "tag"\n'
        '    ctx.observed.append((Payload(7).value, [f.name for f in fields(Payload)]))\n',
        encoding='utf-8')
    observed = []
    assert apply_recipe(study, {'recipe': ('recipes/' if nested else '') + 'recipe.py'},
                        SimpleNamespace(observed=observed))
    assert observed == [(7, ['value'])]
