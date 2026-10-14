"""Native status falls back to index/results when the optional activity hint is unusable."""
from pathlib import Path

import pytest

from rpipe.flow.cli import main
from rpipe.structure.artifact.index import write_index
from rpipe.structure.artifact.result import write_result

pytestmark = [pytest.mark.integration, pytest.mark.content, pytest.mark.p1,
              pytest.mark.flow_layer, pytest.mark.module_cli,
              pytest.mark.cost(cost_class='c1'),
              pytest.mark.result_type('categorical', detail='summary')]


@pytest.mark.parametrize('fault', ['encoding', 'stat-denied'])
@pytest.mark.parametrize('indexed', [False, True])
def test_status_ignores_unusable_hint_and_keeps_index_contract(tmp_path, monkeypatch, capsys, fault, indexed):
    """CLI status -> optional unreadable activity -> real index/result listing or missing-index exit2; all bytes preserved."""
    study = tmp_path / 'study'
    study.mkdir()
    activity = study / 'activity.json'
    activity.write_bytes(b'\xff' if fault == 'encoding' else b'{"phase": "make"}')
    if indexed:
        write_index(study, {'study': 'status', 'experiments': [
            {'factors': {'algorithm.mode': 'train'}, 'runs': [{'id': 'old', 'seed': 0}]}]})
        write_result(study / 'runs/old/result.json',
                     {'status': 'succeeded', 'control': {}, 'metrics': {'accuracy': 42}, 'paths': {}})
    entries = set(study.rglob('*'))
    files = [p for p in entries if p.is_file()]
    before = {p: p.read_bytes() for p in files}
    if fault == 'stat-denied':
        original = Path.is_file
        def denied(path):
            if path == activity:
                raise PermissionError('activity denied')
            return original(path)
        monkeypatch.setattr(Path, 'is_file', denied)
    assert main(['status', str(study)]) == (0 if indexed else 2)
    captured = capsys.readouterr()
    if indexed:
        assert 'planned=1' in captured.out and 'succeeded=1' in captured.out and 'old' in captured.out
        assert 'make:' not in captured.out and not captured.err
    else:
        assert 'missing index:' in captured.err and not captured.out
    assert {p: p.read_bytes() for p in files} == before
    assert set(study.rglob('*')) == entries
