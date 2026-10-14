"""Immutable input promotion owns each temporary reference, including concurrent calls."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from rpipe.structure.artifact import _atomic, shared

pytestmark = [pytest.mark.unit, pytest.mark.content, pytest.mark.p1,
              pytest.mark.structure_layer, pytest.mark.module_artifact,
              pytest.mark.cost(cost_class='c2'),
              pytest.mark.result_type('categorical', detail='summary')]


@pytest.mark.parametrize('clock', ['real', 'frozen'])
@pytest.mark.parametrize('backend', ['default', 'hardlink'])
def test_concurrent_identical_promotions_own_distinct_staging(tmp_path, monkeypatch, clock, backend):
    source, target = tmp_path / 'source', tmp_path / 'duplicate'
    source.write_bytes(b'immutable payload')
    target.write_bytes(source.read_bytes())
    peer = tmp_path / 'duplicate.peer.link'
    peer.write_bytes(b'peer writer')
    if clock == 'frozen':
        monkeypatch.setattr(shared, 'time', SimpleNamespace(time_ns=lambda: 123, sleep=shared.time.sleep))
    if backend == 'hardlink':
        def unavailable(*args, **kwargs):
            raise OSError('symlinks unavailable')
        monkeypatch.setattr(Path, 'symlink_to', unavailable)
    checked, ready = threading.Barrier(2), threading.Barrier(2)
    digest, replace = shared._digest, _atomic.atomic_replace
    staged = []
    def verify(path):
        value = digest(path)
        if path == target:
            checked.wait(timeout=5)
        return value
    def publish(path, destination):
        staged.append(path)
        ready.wait(timeout=2)
        return replace(path, destination)
    monkeypatch.setattr(shared, '_digest', verify)
    monkeypatch.setattr(_atomic, 'atomic_replace', publish)
    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(shared.link_input, source, target) for _ in range(2)]
        errors = [future.exception(timeout=10) for future in futures]
    assert errors == [None, None], [repr(error) for error in errors]
    assert [future.result() for future in futures] == [target, target]
    assert len(staged) == len(set(staged)) == 2
    assert all(not path.exists() for path in staged)
    assert source.read_bytes() == target.read_bytes() == b'immutable payload'
    assert target.samefile(source)
    assert peer.read_bytes() == b'peer writer'
    assert set(tmp_path.iterdir()) == {source, target, peer}


def test_failed_reference_publication_preserves_original_and_peer(tmp_path, monkeypatch):
    source, target = tmp_path / 'source', tmp_path / 'duplicate'
    source.write_bytes(b'immutable payload')
    target.write_bytes(source.read_bytes())
    peer = tmp_path / 'duplicate.peer.link'
    peer.write_bytes(b'peer writer')
    def denied(*args, **kwargs):
        raise PermissionError('publication denied')
    monkeypatch.setattr(_atomic, 'atomic_replace', denied)
    with pytest.raises(PermissionError, match='publication denied'):
        shared.link_input(source, target)
    assert source.read_bytes() == target.read_bytes() == b'immutable payload'
    assert not target.samefile(source)
    assert peer.read_bytes() == b'peer writer'
    assert set(tmp_path.iterdir()) == {source, target, peer}
