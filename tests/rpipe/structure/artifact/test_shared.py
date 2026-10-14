from pathlib import Path
import os
import subprocess
import sys

import pytest

from rpipe.structure.artifact.layout import artifact_layout
from rpipe.structure.artifact.shared import cache_lock, link_input, migrate_shared, shared_paths
from rpipe.structure.artifact.shared import prepared_resource

pytestmark = [pytest.mark.unit, pytest.mark.content, pytest.mark.p1,
              pytest.mark.structure_layer, pytest.mark.module_artifact,
              pytest.mark.cost(cost_class='c2'),
              pytest.mark.result_type('categorical', detail='summary')]


def test_studies_share_inputs_but_not_run_outputs(tmp_path):
    a = artifact_layout(tmp_path / 'studies' / 'a', 'same-id')
    b = artifact_layout(tmp_path / 'studies' / 'b', 'same-id')
    assert a.shared_data_dir == b.shared_data_dir == tmp_path / 'shared' / 'data'
    assert a.shared_model_dir == b.shared_model_dir == tmp_path / 'shared' / 'model'
    assert a.assets_dir != b.assets_dir


def test_private_data_does_not_disable_shared_model(tmp_path):
    study = tmp_path / 'studies' / 'private'
    study.mkdir(parents=True)
    (study / 'study.yaml').write_text('shared: {data: study}\n')
    paths = shared_paths(study)
    assert paths['data'] == study / 'shared' / 'data'
    assert paths['model'] == tmp_path / 'shared' / 'model'


def test_explicit_root_is_resolved_relative_to_study(tmp_path):
    study = tmp_path / 'experiments' / 'a'
    study.mkdir(parents=True)
    (study / 'study.yaml').write_text('shared: {root: ../../input-cache}\n')
    assert shared_paths(study)['model'] == tmp_path / 'input-cache' / 'model'


@pytest.mark.parametrize('block', ['null', '[]', '{root: ""}', '{data: false}', '{model: local}', '{typo: project}'])
def test_bad_shared_config_creates_no_layout(tmp_path, block):
    study = tmp_path / 'studies' / 'a'
    study.mkdir(parents=True)
    (study / 'study.yaml').write_text(f'shared: {block}\n')
    with pytest.raises(ValueError):
        artifact_layout(study, 'r')
    assert not (study / 'runs').exists()
    assert not (tmp_path / 'shared').exists()


def test_migration_deduplicates_identical_bytes_and_keeps_profiles_private(tmp_path):
    studies = [tmp_path / 'studies' / name for name in ('a', 'b')]
    for study in studies:
        for kind in ('data', 'model'):
            folder = study / 'shared' / kind
            folder.mkdir(parents=True)
            (folder / 'input.bin').write_bytes(b'one immutable input')
        (study / 'shared' / 'data' / 'stats.yaml').write_text('private stats')
    for study in studies:
        report = migrate_shared(study)
        assert all(row['same_file'] for row in report['files'])
    for kind in ('data', 'model'):
        shared = tmp_path / 'shared' / kind / 'input.bin'
        assert all((study / 'shared' / kind / 'input.bin').samefile(shared) for study in studies)
        assert shared.read_bytes() == b'one immutable input'
    assert not (tmp_path / 'shared' / 'data' / 'stats.yaml').exists()
    assert migrate_shared(studies[0])['files']  # idempotent, preserved references


def test_migration_checks_all_conflicts_before_linking(tmp_path):
    study = tmp_path / 'studies' / 'a'
    data = study / 'shared' / 'data'
    model = study / 'shared' / 'model'
    data.mkdir(parents=True)
    model.mkdir(parents=True)
    (data / 'raw.bin').write_bytes(b'data')
    (model / 'weights.bin').write_bytes(b'old')
    target = tmp_path / 'shared' / 'model'
    target.mkdir(parents=True)
    (target / 'weights.bin').write_bytes(b'different revision')
    with pytest.raises(ValueError, match='conflicting'):
        migrate_shared(study)
    assert not (tmp_path / 'shared' / 'data' / 'raw.bin').exists()
    assert (model / 'weights.bin').read_bytes() == b'old'
    assert (target / 'weights.bin').read_bytes() == b'different revision'


def test_input_reference_never_copies_or_overwrites(tmp_path):
    source = tmp_path / 'raw'
    source.write_bytes(b'official raw')
    target = link_input(source, tmp_path / 'run' / 'raw')
    assert target.samefile(source)
    assert link_input(source, target) == target
    unrelated = tmp_path / 'other'
    unrelated.write_bytes(b'other')
    with pytest.raises(FileExistsError):
        link_input(unrelated, target)
    assert source.read_bytes() == b'official raw'


def test_identical_raw_duplicates_become_one_file(tmp_path):
    from rpipe.structure.artifact.shared import reference_tree
    source, target = tmp_path / 'raw', tmp_path / 'backend'
    source.mkdir()
    target.mkdir()
    (source / 'input.bin').write_bytes(b'official')
    (target / 'input.bin').write_bytes(b'official')
    (source / 'stats.yaml').write_text('must not be a shared raw input')
    reference_tree(source, target)
    assert (target / 'input.bin').samefile(source / 'input.bin')
    assert not (target / 'stats.yaml').exists()


def test_failed_reference_creation_keeps_duplicate_payload(tmp_path, monkeypatch):
    source, target = tmp_path / 'raw', tmp_path / 'duplicate'
    source.write_bytes(b'complete')
    target.write_bytes(b'complete')
    def refuse(*args, **kwargs):
        raise OSError('injected reference failure')
    monkeypatch.setattr(Path, 'symlink_to', refuse)
    monkeypatch.setattr(os, 'link', refuse)
    with pytest.raises(OSError, match='injected'):
        link_input(source, target)
    assert target.read_bytes() == b'complete'
    assert source.read_bytes() == b'complete'
    assert not list(tmp_path.glob('*.link'))


def test_interprocess_prepare_has_one_writer_and_recovers_after_exit(tmp_path):
    script = '''
from pathlib import Path
import sys, time
from rpipe.structure.artifact.shared import cache_lock
root = Path(sys.argv[1])
with cache_lock(root, timeout=20):
    target = root / 'download.bin'
    if not target.exists():
        with (root / 'writers.txt').open('a') as log:
            log.write('writer\\n')
        time.sleep(.15)
        target.write_bytes(b'complete')
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[4] / 'src'))
    processes = [subprocess.Popen([sys.executable, '-c', script, str(tmp_path)], env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(2)]
    for process in processes:
        stdout, stderr = process.communicate(timeout=40)
        assert process.returncode == 0, (stdout, stderr)
    assert (tmp_path / 'writers.txt').read_text() == 'writer\n'
    with cache_lock(tmp_path, timeout=0):
        assert (tmp_path / 'download.bin').read_bytes() == b'complete'


def test_terminated_writer_does_not_leave_an_owned_lock(tmp_path):
    script = '''
import sys, time
from rpipe.structure.artifact.shared import cache_lock
with cache_lock(sys.argv[1]):
    print('locked', flush=True)
    time.sleep(30)
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[4] / 'src'))
    process = subprocess.Popen([sys.executable, '-c', script, str(tmp_path)], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert process.stdout.readline().strip() == b'locked'
        with pytest.raises(TimeoutError):
            with cache_lock(tmp_path, timeout=0):
                pass
    finally:
        process.kill()
        process.communicate(timeout=10)
    with cache_lock(tmp_path, timeout=0):
        assert (tmp_path / '.rpipe.lock').is_file()


def test_same_definition_reuses_input_and_changed_revision_builds_separately(tmp_path):
    built = []
    def build(staging):
        built.append(staging)
        (staging / 'weights').write_bytes(b'immutable')
    first, digest = prepared_resource(tmp_path, {'revision': 'one'}, ['weights'], build)
    second, same_digest = prepared_resource(tmp_path, {'revision': 'one'}, ['weights'], build)
    other, _ = prepared_resource(tmp_path, {'revision': 'two'}, ['weights'], build)
    assert first == second and digest == same_digest
    assert other != first and len(built) == 2


def test_partial_build_is_retained_without_publishing_ready(tmp_path):
    def fail(staging):
        (staging / 'data').write_bytes(b'partial')
        raise OSError('injected prepare failure')
    with pytest.raises(OSError, match='injected'):
        prepared_resource(tmp_path, {'revision': 'one'}, ['data'], fail)
    assert not list(tmp_path.glob('*/inputs'))
    assert len(list(tmp_path.glob('*/.building-*/data'))) == 1


def test_corrupt_payload_and_incomplete_manifest_are_not_rebuilt_or_accepted(tmp_path):
    import json
    def build(staging):
        (staging / 'data').write_bytes(b'complete')
    cache, _ = prepared_resource(tmp_path, {}, ['data'], build)
    manifest = cache / 'manifest.json'
    original = manifest.read_bytes()
    (cache / 'data').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='cache differs'):
        prepared_resource(tmp_path, {}, ['data'], build)
    assert (cache / 'data').read_bytes() == b'corrupt'
    (cache / 'data').write_bytes(b'complete')
    body = json.loads(original)
    body['files'] = {}
    manifest.write_text(json.dumps(body))
    with pytest.raises(ValueError, match='cache differs'):
        prepared_resource(tmp_path, {}, ['data'], build)


@pytest.mark.parametrize('files', [[], ['../outside'], ['manifest.json'], ['data', 'data']])
def test_invalid_resource_contract_is_rejected_before_writes(tmp_path, files):
    with pytest.raises(ValueError, match='relative'):
        prepared_resource(tmp_path, {}, files, lambda staging: None)
    assert not list(tmp_path.iterdir())

@pytest.mark.parametrize('timeout', [float('nan'), float('inf'), float('-inf'), -1.0])
def test_invalid_cache_lock_budget_is_rejected_before_creating_root(tmp_path, timeout):
    root = tmp_path / 'invalid-budget'
    with pytest.raises(ValueError, match='timeout'):
        with cache_lock(root, timeout=timeout):
            pass
    assert not root.exists()


def test_nan_cache_lock_budget_cannot_disable_contention_deadline(tmp_path, monkeypatch):
    """A bounded busy-lock fixture exposes the old NaN loop without hanging pytest."""
    import errno
    import time
    from types import SimpleNamespace
    import rpipe.structure.artifact.shared as shared

    sleeps = []
    def busy(*args):
        raise BlockingIOError(errno.EACCES, 'fixture lock held by another writer')
    def watchdog(delay):
        sleeps.append(delay)
        if len(sleeps) == 3:
            raise RuntimeError('bounded watchdog: NaN disabled the contention deadline')
    if os.name == 'nt':
        import msvcrt
        monkeypatch.setattr(msvcrt, 'locking', busy)
    else:
        import fcntl
        monkeypatch.setattr(fcntl, 'flock', busy)
    monkeypatch.setattr(shared, 'time', SimpleNamespace(monotonic=time.monotonic, sleep=watchdog))
    root = tmp_path / 'nan-budget'
    with pytest.raises(ValueError, match='timeout'):
        with cache_lock(root, timeout=float('nan')):
            pytest.fail('invalid budget acquired a lock')
    assert sleeps == [] and not root.exists()


def test_interrupted_resource_retry_preserves_failed_staging_and_publishes_fresh(tmp_path):
    def interrupted(staging):
        (staging / 'data').write_bytes(b'partial')
        raise OSError('fixture interrupted preparation')
    with pytest.raises(OSError, match='interrupted'):
        prepared_resource(tmp_path, {'revision': 'retry'}, ['data'], interrupted)
    failed = list(tmp_path.glob('*/.building-*/data'))
    assert len(failed) == 1 and failed[0].read_bytes() == b'partial'

    def complete(staging):
        (staging / 'data').write_bytes(b'complete')
    cache, digest = prepared_resource(tmp_path, {'revision': 'retry'}, ['data'], complete)
    assert (cache / 'data').read_bytes() == b'complete'
    assert failed[0].read_bytes() == b'partial'
    def no_rebuild(staging):
        pytest.fail('published inputs must be reused')
    assert prepared_resource(tmp_path, {'revision': 'retry'}, ['data'], no_rebuild) == (cache, digest)



def test_zero_wait_busy_lock_preserves_cause_and_releases_handle(tmp_path, monkeypatch):
    import errno
    from types import SimpleNamespace
    import rpipe.structure.artifact.shared as shared

    error = BlockingIOError(errno.EACCES, 'fixture occupied lock')
    def busy(*args):
        raise error
    def no_sleep(delay):
        pytest.fail('zero budget must not sleep')
    with monkeypatch.context() as patch:
        if os.name == 'nt':
            import msvcrt
            patch.setattr(msvcrt, 'locking', busy)
        else:
            import fcntl
            patch.setattr(fcntl, 'flock', busy)
        patch.setattr(shared, 'time', SimpleNamespace(monotonic=lambda: 100, sleep=no_sleep))
        with pytest.raises(TimeoutError, match='shared input preparation locked') as caught:
            with cache_lock(tmp_path, timeout=0):
                pytest.fail('occupied lock was acquired')
        assert caught.value.__cause__ is error
    with cache_lock(tmp_path, timeout=0):
        pass


def test_non_contention_lock_error_is_not_retried_or_wrapped(tmp_path, monkeypatch):
    import errno
    from types import SimpleNamespace
    import rpipe.structure.artifact.shared as shared

    error = OSError(errno.EBADF, 'fixture invalid descriptor')
    calls = []
    def broken(*args):
        calls.append(args)
        raise error
    def no_sleep(delay):
        pytest.fail('descriptor errors must not retry')
    with monkeypatch.context() as patch:
        if os.name == 'nt':
            import msvcrt
            patch.setattr(msvcrt, 'locking', broken)
        else:
            import fcntl
            patch.setattr(fcntl, 'flock', broken)
        patch.setattr(shared, 'time', SimpleNamespace(monotonic=lambda: 100, sleep=no_sleep))
        with pytest.raises(OSError) as caught:
            with cache_lock(tmp_path):
                pytest.fail('broken descriptor was accepted')
        assert caught.value is error and len(calls) == 1
    with cache_lock(tmp_path, timeout=0):
        pass


def test_nested_lock_exception_keeps_outer_owner_then_releases_for_other_thread(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    def acquire():
        with cache_lock(tmp_path, timeout=0):
            return 'acquired'
    with ThreadPoolExecutor(max_workers=1) as worker:
        with pytest.raises(RuntimeError, match='outer interrupted'):
            with cache_lock(tmp_path, timeout=0):
                with pytest.raises(ValueError, match='inner interrupted'):
                    with cache_lock(tmp_path / '.', timeout=0):
                        raise ValueError('inner interrupted')
                with pytest.raises(TimeoutError):
                    worker.submit(acquire).result(timeout=10)
                raise RuntimeError('outer interrupted')
        assert worker.submit(acquire).result(timeout=10) == 'acquired'



def test_finite_lock_deadline_has_bounded_retry_timeline(tmp_path, monkeypatch):
    import errno
    from types import SimpleNamespace
    import rpipe.structure.artifact.shared as shared

    clock = iter([0.0, 0.0, 0.05, 0.1])
    attempts, sleeps = [], []
    error = BlockingIOError(errno.EAGAIN, 'fixture contention')
    def busy(*args):
        attempts.append(args)
        raise error
    with monkeypatch.context() as patch:
        if os.name == 'nt':
            import msvcrt
            patch.setattr(msvcrt, 'locking', busy)
        else:
            import fcntl
            patch.setattr(fcntl, 'flock', busy)
        patch.setattr(shared, 'time', SimpleNamespace(monotonic=lambda: next(clock), sleep=sleeps.append))
        with pytest.raises(TimeoutError) as caught:
            with cache_lock(tmp_path, timeout=0.1):
                pytest.fail('held lock was acquired')
        assert caught.value.__cause__ is error
        assert len(attempts) == 3 and sleeps == [0.05, 0.05]
    with cache_lock(tmp_path, timeout=0):
        pass


def test_cancelled_wait_cleans_handle_before_next_acquisition(tmp_path, monkeypatch):
    import errno
    from types import SimpleNamespace
    import rpipe.structure.artifact.shared as shared

    cancelled = KeyboardInterrupt('fixture cancelled wait')
    attempts = []
    def busy(*args):
        attempts.append(args)
        raise BlockingIOError(errno.EACCES, 'fixture contention')
    def cancel(delay):
        assert delay == 0.05
        raise cancelled
    with monkeypatch.context() as patch:
        if os.name == 'nt':
            import msvcrt
            patch.setattr(msvcrt, 'locking', busy)
        else:
            import fcntl
            patch.setattr(fcntl, 'flock', busy)
        patch.setattr(shared, 'time', SimpleNamespace(monotonic=lambda: 0, sleep=cancel))
        with pytest.raises(KeyboardInterrupt) as caught:
            with cache_lock(tmp_path):
                pytest.fail('cancelled wait entered critical section')
        assert caught.value is cancelled and len(attempts) == 1
    with cache_lock(tmp_path, timeout=0):
        pass


def test_repeated_failed_acquisitions_do_not_release_other_thread_owner(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    def acquire():
        with cache_lock(tmp_path, timeout=0):
            return 'acquired'
    with ThreadPoolExecutor(max_workers=1) as worker:
        with pytest.raises(KeyboardInterrupt, match='owner cancelled'):
            with cache_lock(tmp_path, timeout=0):
                for _ in range(3):
                    with pytest.raises(TimeoutError):
                        worker.submit(acquire).result(timeout=10)
                raise KeyboardInterrupt('owner cancelled')
        assert worker.submit(acquire).result(timeout=10) == 'acquired'
