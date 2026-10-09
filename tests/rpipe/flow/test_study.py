from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from rpipe.flow.runner import FlowRunner
from rpipe.flow.study import run_study_phase
from rpipe.structure.make.recipe import study_phase_files, study_phases_enabled

pytestmark = [pytest.mark.unit, pytest.mark.content, pytest.mark.p1,
              pytest.mark.flow_layer, pytest.mark.module_runner,
              pytest.mark.cost(cost_class='c1'), pytest.mark.result_type('categorical', detail='summary')]


def test_relative_helpers_are_isolated_between_studies(tmp_path):
    before = set(sys.modules)
    ctx = SimpleNamespace(state={})
    for label in ('left', 'right'):
        root = tmp_path / label
        phase = root / 'prepare'
        phase.mkdir(parents=True)
        (root / 'helper.py').write_text(f'MARK = {label!r}\n', encoding='utf-8')
        (phase / '__init__.py').write_text(
            'from ..helper import MARK\ndef run(ctx):\n    ctx.state["mark"] = MARK\n', encoding='utf-8')
        assert run_study_phase(root, 'prepare', ctx, {'flow': {'study_phases': True}})
        assert ctx.state['mark'] == label
    assert not {name for name in set(sys.modules) - before if name.startswith('_rpipe_study_')}


def test_disabled_study_directories_do_not_execute(tmp_path):
    phase = tmp_path / 'prepare'
    phase.mkdir()
    (phase / '__init__.py').write_text('raise RuntimeError("must not import")\n', encoding='utf-8')
    assert not run_study_phase(tmp_path, 'prepare', None, {})
    assert study_phase_files(tmp_path, {}) == []


@pytest.mark.parametrize('declaration', [{'flow': None}, {'flow': []}, {'flow': {'study_phases': 'true'}},
                                      {'flow': {'study_phases': 1}}])
def test_invalid_phase_declaration_is_rejected(declaration):
    with pytest.raises(ValueError):
        study_phases_enabled(declaration)


def test_invalid_hook_does_not_leak_modules(tmp_path):
    directory = tmp_path / 'prepare'
    directory.mkdir()
    with pytest.raises(ValueError, match='package'):
        study_phase_files(tmp_path, {'flow': {'study_phases': True}})
    (directory / '__init__.py').write_text('x = 1\n', encoding='utf-8')
    before = set(sys.modules)
    with pytest.raises(AttributeError, match='run'):
        run_study_phase(tmp_path, 'prepare', None, {'flow': {'study_phases': True}})
    assert not {name for name in set(sys.modules) - before if name.startswith('_rpipe_study_')}


@pytest.mark.parametrize('phases', [['execute', 'prepare'], ['prepare', 'prepare'], ['write', 'persist']])
def test_stage_order_and_duplicates_are_rejected(phases):
    with pytest.raises(ValueError, match='canonical order'):
        FlowRunner(phases)


def test_recipe_owned_shared_preparation_is_not_called(tmp_path, monkeypatch):
    from rpipe.flow.cli import _prepare_shared
    (tmp_path / 'study.yaml').write_text('flow:\n  prepare_shared: false\n', encoding='utf-8')
    def forbidden(*args):
        raise AssertionError('must not build custom data before recipe registration')
    monkeypatch.setattr('rpipe.flow.cli.data_api.prepare_shared', forbidden)
    _prepare_shared(tmp_path, [])


@pytest.mark.parametrize('value', ['null', '1', '"false"'])
def test_shared_preparation_requires_boolean(tmp_path, value):
    from rpipe.flow.cli import _prepare_shared
    (tmp_path / 'study.yaml').write_text(f'flow:\n  prepare_shared: {value}\n', encoding='utf-8')
    with pytest.raises(TypeError, match='prepare_shared'):
        _prepare_shared(tmp_path, [])
