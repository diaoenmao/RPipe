"""Overlapping source declarations preserve keys, changes, and error boundaries."""
import hashlib
from pathlib import Path

import pytest

from rpipe.structure.artifact import provenance

pytestmark = [pytest.mark.unit, pytest.mark.content, pytest.mark.p1,
              pytest.mark.structure_layer, pytest.mark.module_artifact,
              pytest.mark.cost(cost_class='c1'),
              pytest.mark.result_type('categorical', detail='summary')]


@pytest.mark.parametrize('enabled', [False, True])
def test_overlapping_sources_keep_keys_and_detect_later_edits(tmp_path, enabled):
    names = ['flow/execute/__init__.py', 'flow/execute/run.py', 'structure/helper.py']
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'VALUE = 1\n')
    study = {'flow': {'study_phases': enabled}, 'recipe': './structure/helper.py',
             'provenance': {'include': ['flow/**/*.py', 'structure/*.py', 'structure/helper.py']}}
    first = provenance.source_files(tmp_path, study)
    assert {name: first[name] for name in names} == {name: hashlib.sha256(b'VALUE = 1\n').hexdigest() for name in names}
    assert set(names) <= set(first)
    assert './structure/helper.py' not in first
    (tmp_path / 'structure/helper.py').write_bytes(b'VALUE = 2\n')
    second = provenance.source_files(tmp_path, study)
    assert set(second) == set(first)
    assert first['structure/helper.py'] != second['structure/helper.py']


def test_overlapping_discovery_preserves_read_failure(tmp_path, monkeypatch):
    target = tmp_path / 'structure/helper.py'
    target.parent.mkdir()
    target.write_bytes(b'input')
    original = Path.read_bytes
    def denied(path):
        if path == target:
            raise PermissionError('source denied')
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', denied)
    with pytest.raises(PermissionError, match='source denied'):
        provenance.source_files(tmp_path, {'provenance': {'include': ['structure/*.py']}})


@pytest.mark.parametrize('extra,error', [('missing/*.py', FileNotFoundError), ('../outside.py', ValueError)])
def test_overlap_does_not_hide_invalid_extra_input(tmp_path, extra, error):
    study = tmp_path / 'study'
    (study / 'structure').mkdir(parents=True)
    (study / 'structure/helper.py').write_bytes(b'input')
    (tmp_path / 'outside.py').write_bytes(b'outside')
    with pytest.raises(error):
        provenance.source_files(study, {'provenance': {'include': ['structure/*.py', extra]}})
