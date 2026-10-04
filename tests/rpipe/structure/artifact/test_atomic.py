from pathlib import Path
import pytest

from rpipe.structure.artifact._atomic import atomic_replace, atomic_write_text

pytestmark = [pytest.mark.unit, pytest.mark.content, pytest.mark.p1,
              pytest.mark.structure_layer, pytest.mark.module_artifact,
              pytest.mark.cost(cost_class='c1'),
              pytest.mark.result_type('categorical', detail='summary')]


@pytest.mark.parametrize('winerror', [5, 32, 33])
def test_atomic_replace_transient_windows_denial_eventually_commits(tmp_path, monkeypatch, winerror):
    source, target = tmp_path / 'new', tmp_path / 'old'
    source.write_text('new', encoding='utf-8')
    target.write_text('old', encoding='utf-8')
    replace, pauses, calls = Path.replace, [], []

    def held(path, destination):
        calls.append(path)
        if len(calls) <= 2:
            assert target.read_text(encoding='utf-8') == 'old'
            error = PermissionError('file temporarily held')
            error.winerror = winerror
            raise error
        return replace(path, destination)

    monkeypatch.setattr(Path, 'replace', held)
    monkeypatch.setattr('rpipe.structure.artifact._atomic.time.sleep', pauses.append)
    assert atomic_replace(source, target) == target
    assert target.read_text(encoding='utf-8') == 'new' and not source.exists()
    assert len(calls) == 3 and pauses == [0.05, 0.1]


@pytest.mark.parametrize('winerror,expected_calls', [(5, 6), (None, 1), (112, 1)])
def test_atomic_write_permanent_failure_preserves_target_and_cleans_stage(tmp_path, monkeypatch, winerror, expected_calls):
    target, calls = tmp_path / 'old', []
    target.write_text('old', encoding='utf-8')
    error = PermissionError('permanent failure')
    error.winerror = winerror

    def denied(path, destination):
        calls.append(path)
        raise error

    monkeypatch.setattr(Path, 'replace', denied)
    monkeypatch.setattr('rpipe.structure.artifact._atomic.time.sleep', lambda _: None)
    with pytest.raises(PermissionError) as caught:
        atomic_write_text(target, 'new')
    assert caught.value is error and len(calls) == expected_calls
    assert target.read_text(encoding='utf-8') == 'old'
    assert list(tmp_path.iterdir()) == [target]
