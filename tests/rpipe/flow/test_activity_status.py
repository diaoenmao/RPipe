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


@pytest.mark.parametrize('indexed', [False, True], ids=['missing-index', 'planned-peer'])
@pytest.mark.parametrize('field', ['phase', 'detail'])
@pytest.mark.parametrize('separator', ['\n', '\t'])
def test_status_hint_cannot_add_rows_or_columns_and_keeps_original_artifacts(tmp_path, capsys, indexed, field, separator):
    import json
    study = tmp_path / 'study'
    study.mkdir()
    body = {'phase': 'make', 'detail': 'working'}
    body[field] += separator + 'foreign'
    if field == 'detail':
        body[field] += '\ttrain\t999\tghost\t-\taccuracy=1\t-\t-\t-'
    activity = study / 'activity.json'
    activity.write_bytes(json.dumps(body, ensure_ascii=False).encode())
    if indexed:
        write_index(study, {'study': 'hint', 'experiments': [
            {'factors': {'algorithm.mode': 'train'}, 'runs': [{'id': 'old', 'seed': 0}]}]})
        write_result(study / 'runs/old/result.json',
                     {'status': 'succeeded', 'control': {}, 'metrics': {'accuracy': 42}, 'paths': {}})
    entries = set(study.rglob('*'))
    before = {path: path.read_bytes() for path in entries if path.is_file()}
    assert main(['status', str(study)]) == 0
    captured = capsys.readouterr()
    assert not captured.err
    lines = captured.out.splitlines()
    assert len(lines) == (4 if indexed else 1)
    assert '\t' not in lines[0] and all(word in lines[0] for word in ('make', 'working', 'foreign'))
    if indexed:
        assert 'planned=1 succeeded=1' in lines[1]
        assert lines[2].startswith('status\tmode\tseed\tid\t')
        fields = lines[3].split('\t')
        assert len(fields) == 9 and fields[0] == 'succeeded' and fields[3] == 'old'
    assert {path: path.read_bytes() for path in before} == before
    assert set(study.rglob('*')) == entries
