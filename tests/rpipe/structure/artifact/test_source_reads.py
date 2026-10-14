"""Source discovery overlaps must not repeatedly read a resolved input per call."""
from collections import Counter
from pathlib import Path

import pytest

from rpipe.structure.artifact import provenance

pytestmark = [pytest.mark.unit, pytest.mark.content, pytest.mark.p1,
              pytest.mark.structure_layer, pytest.mark.module_artifact,
              pytest.mark.cost(cost_class='c1'),
              pytest.mark.result_type('categorical', detail='summary')]


@pytest.mark.parametrize('enabled', [False, True])
def test_overlapping_sources_read_each_file_once_per_call(tmp_path, monkeypatch, enabled):
    names = ['flow/execute/__init__.py', 'flow/execute/run.py', 'structure/helper.py']
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'VALUE = 1\n')
    reads, original = Counter(), Path.read_bytes
    def tracked(path):
        if path.is_relative_to(tmp_path):
            reads[path.relative_to(tmp_path).as_posix()] += 1
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', tracked)
    study = {'flow': {'study_phases': enabled}, 'recipe': './structure/helper.py',
             'provenance': {'include': ['flow/**/*.py', 'structure/*.py', 'structure/helper.py']}}
    first = provenance.source_files(tmp_path, study)
    assert reads == Counter({name: 1 for name in names})
    assert set(names) <= set(first)
    assert './structure/helper.py' not in first
    reads.clear()
    (tmp_path / 'structure/helper.py').write_bytes(b'VALUE = 2\n')
    second = provenance.source_files(tmp_path, study)
    assert reads == Counter({name: 1 for name in names})
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
