from pathlib import Path
import os
import subprocess
import sys

import pytest

from rpipe.structure.artifact.layout import artifact_layout
from rpipe.structure.artifact.shared import cache_lock, link_input, migrate_shared, shared_paths

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
