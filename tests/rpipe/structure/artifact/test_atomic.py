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


@pytest.mark.parametrize('existing', [False, True])
@pytest.mark.parametrize('error_type', [OSError, KeyboardInterrupt])
def test_partial_text_write_failure_cleans_only_its_stage(tmp_path, monkeypatch, existing, error_type):
    target = tmp_path / 'result.txt'
    if existing:
        target.write_text('old', encoding='utf-8')
    peer = tmp_path / 'result.txt.other.tmp'
    peer.write_text('another writer', encoding='utf-8')
    original_write = Path.write_text
    error = error_type('injected partial text write failure')
    def partial_then_fail(path, text, *args, **kwargs):
        original_write(path, 'partial', *args, **kwargs)
        raise error
    monkeypatch.setattr(Path, 'write_text', partial_then_fail)
    with pytest.raises(error_type) as caught:
        atomic_write_text(target, 'new')
    assert caught.value is error
    assert peer.read_text(encoding='utf-8') == 'another writer'
    assert set(tmp_path.iterdir()) == ({target, peer} if existing else {peer})
    if existing:
        assert target.read_text(encoding='utf-8') == 'old'


@pytest.mark.parametrize('existing', [False, True])
def test_text_encoding_failure_preserves_target_and_peer(tmp_path, existing):
    target, peer = tmp_path / 'result.txt', tmp_path / 'peer.tmp'
    if existing:
        target.write_bytes(b'original')
    peer.write_bytes(b'peer staging')
    with pytest.raises(UnicodeEncodeError):
        atomic_write_text(target, 'plain\u4e2d\u6587', encoding='ascii')
    assert set(tmp_path.iterdir()) == ({target, peer} if existing else {peer})
    assert peer.read_bytes() == b'peer staging'
    if existing:
        assert target.read_bytes() == b'original'


def test_atomic_text_success_replaces_target_and_keeps_peer_stage(tmp_path):
    target = tmp_path / 'result.txt'
    target.write_text('old', encoding='utf-8')
    peer = tmp_path / 'result.txt.other.tmp'
    peer.write_text('another writer', encoding='utf-8')
    assert atomic_write_text(target, 'new') == target
    assert target.read_text(encoding='utf-8') == 'new'
    assert peer.read_text(encoding='utf-8') == 'another writer'
    assert set(tmp_path.iterdir()) == {target, peer}
