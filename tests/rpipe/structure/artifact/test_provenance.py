from pathlib import Path

import pytest

from rpipe.structure.artifact.provenance import (
    ProvenanceChangedError, check_frozen, source_files, write_provenance,
)

pytestmark = [pytest.mark.unit, pytest.mark.content, pytest.mark.p1,
              pytest.mark.structure_layer, pytest.mark.module_artifact,
              pytest.mark.cost(cost_class='c1'), pytest.mark.result_type('categorical', detail='summary')]


def _source(root: Path, relative: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('VALUE = 1\n', encoding='utf-8')
    return path


@pytest.mark.parametrize('enabled', [False, True])
def test_two_column_sources_are_frozen_without_include_or_hooks(tmp_path, enabled):
    sources = ('flow/__init__.py', 'flow/prepare_data.py', 'flow/execute/speed.py',
               'structure/__init__.py', 'structure/data/helper.py')
    for name in sources:
        _source(tmp_path, name)
    _source(tmp_path, 'runs/old/assets/source/model.py')
    _source(tmp_path, 'shared/generated.py')
    _source(tmp_path, 'flow/__pycache__/cached.pyc')
    files = source_files(tmp_path, {'flow': {'study_phases': enabled}})
    assert set(sources) <= set(files)
    assert not any(name.startswith(('runs/', 'shared/')) or '__pycache__' in name for name in files)


@pytest.mark.parametrize('folder', ['flow', 'structure'])
@pytest.mark.parametrize('change', ['add', 'edit', 'delete'])
def test_freeze_detects_two_column_helper_changes(tmp_path, monkeypatch, folder, change):
    """Two-column source → make provenance → changed helper blocks freeze without explicit include."""
    monkeypatch.setattr('rpipe.structure.artifact.provenance.capture_environment', lambda: {'check': 'CPU'})
    monkeypatch.setattr('rpipe.structure.artifact.provenance.capture_git', lambda root: {})
    path = tmp_path / folder / 'helper.py'
    if change != 'add':
        _source(tmp_path, f'{folder}/helper.py')
    study = {'freeze': True}
    write_provenance(tmp_path, study)
    check_frozen(tmp_path, study)
    if change == 'delete':
        path.unlink()
    else:
        _source(tmp_path, f'{folder}/helper.py').write_text('VALUE = 2\n', encoding='utf-8')
    with pytest.raises(ProvenanceChangedError, match=f'{folder}/helper.py'):
        check_frozen(tmp_path, study)


@pytest.mark.parametrize('folder', ['flow', 'structure'])
def test_two_column_source_cannot_escape_study(tmp_path, folder):
    outside = tmp_path / 'outside'
    outside.mkdir()
    _source(outside, 'helper.py')
    study = tmp_path / 'study'
    study.mkdir()
    try:
        (study / folder).symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip('host does not allow directory symlinks')
    with pytest.raises(ValueError, match='inside the Study'):
        source_files(study, {})
