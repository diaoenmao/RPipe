"""Valid destination basenames must not overflow private staging components."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import threading

import pytest

from rpipe.structure.artifact import _atomic, shared

pytestmark = [pytest.mark.unit, pytest.mark.content, pytest.mark.p1,
              pytest.mark.structure_layer, pytest.mark.module_artifact,
              pytest.mark.cost(cost_class='c2'),
              pytest.mark.result_type('categorical', detail='summary')]


@pytest.mark.parametrize('existing', [False, True])
def test_atomic_text_accepts_valid_long_basename(tmp_path, existing):
    root = Path('\\\\?\\' + str(tmp_path.resolve())) if os.name == 'nt' else tmp_path
    target = root / ('x' * 245)
    if existing:
        target.write_bytes(b'old')
    assert _atomic.atomic_write_text(target, 'complete') == target
    assert target.read_bytes() == b'complete'
    assert set(root.iterdir()) == {target}


@pytest.mark.parametrize('kind', ['data', 'model'])
def test_duplicate_migration_accepts_valid_long_basename(tmp_path, kind):
    root = Path('\\\\?\\' + str(tmp_path.resolve())) if os.name == 'nt' else tmp_path
    study = root / 'studies' / 'a'
    old, public = study / 'shared' / kind, root / 'shared' / kind
    for folder in (old, public):
        folder.mkdir(parents=True)
        (folder / ('x' * 245)).write_bytes(b'input')
    source, target = old / ('x' * 245), public / ('x' * 245)
    assert not source.samefile(target)
    report = shared.migrate_shared(study)
    assert len(report['files']) == 1 and report['files'][0]['same_file']
    assert source.samefile(target)
    assert source.read_bytes() == target.read_bytes() == b'input'
    assert set(old.iterdir()) == {source, old / '.rpipe.lock'}
    assert shared.migrate_shared(study)['files'][0]['same_file']


def test_failed_migration_publication_keeps_copies_and_peer(tmp_path, monkeypatch):
    study = tmp_path / 'studies' / 'a'
    old, public = study / 'shared' / 'data', tmp_path / 'shared' / 'data'
    for folder in (old, public):
        folder.mkdir(parents=True)
        (folder / 'input').write_bytes(b'input')
    peer = old / 'peer.link'
    peer.write_bytes(b'peer')
    def denied(*args, **kwargs):
        raise PermissionError('publish denied')
    monkeypatch.setattr(_atomic, 'atomic_replace', denied)
    with pytest.raises(PermissionError, match='publish denied'):
        shared.migrate_shared(study)
    assert (old / 'input').read_bytes() == (public / 'input').read_bytes() == b'input'
    assert not (old / 'input').samefile(public / 'input')
    assert peer.read_bytes() == b'peer'
    assert set(old.iterdir()) == {old / 'input', peer, old / '.rpipe.lock'}


def test_two_preparers_publish_one_complete_resource(tmp_path):
    ready = threading.Barrier(2)
    built = []
    def build(staging):
        built.append(staging)
        (staging / 'data').write_bytes(b'complete')
    def prepare():
        ready.wait(timeout=5)
        return shared.prepared_resource(tmp_path, {'revision': 'one'}, ['data'], build)
    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(prepare) for _ in range(2)]
        values = [future.result(timeout=10) for future in futures]
    assert len(built) == 1
    assert values[0] == values[1]
    assert (values[0][0] / 'data').read_bytes() == b'complete'
    assert not list(tmp_path.glob('*/.building-*'))
