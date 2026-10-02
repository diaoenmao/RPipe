import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.content,
    pytest.mark.p2,
    pytest.mark.structure_layer,
    pytest.mark.module_system,
    pytest.mark.cost(cost_class='c1'),
    pytest.mark.result_type('categorical', detail='summary'),
]

from pathlib import Path

from rpipe.structure.system.config import SystemConfig
from rpipe.structure.system.factory import SystemFactory


def test_system_save_checkpoint_writes_pt(tmp_path: Path):
    system = SystemFactory.build(SystemConfig.from_mapping({'device': 'cpu'}), tmp_path)
    path = system.save_checkpoint({'step': 3, 'model': {'w': 1}}, 'latest')
    assert path == tmp_path / 'checkpoints' / 'latest.pt'
    assert path.is_file()
    assert (tmp_path / 'checkpoints' / 'latest' / 'meta.json').is_file()
    import torch

    loaded = torch.load(path, weights_only=False)
    assert loaded['step'] == 3
    assert loaded['model'] == {'w': 1}
    again = system.load_checkpoint('latest')
    assert again['step'] == 3
    assert system.load_checkpoint('missing') is None


@pytest.mark.parametrize('failure', ['serialize', 'mirror', 'bundle'])
@pytest.mark.parametrize('previous', ['missing', 'bundle', 'legacy', 'legacy_payload'])
def test_checkpoint_publication_failure_preserves_committed_bundle(tmp_path, monkeypatch, failure, previous):
    import torch

    system = SystemFactory.build(SystemConfig.from_mapping({'device': 'cpu'}), tmp_path)
    old = {'step': 3, 'model': {'w': 1}, 'optimizer': {'lr': 0.1}}
    new = {'step': 4, 'model': {'w': 2}}
    root = system.checkpoint_dir()
    bundle = root / 'latest.pt'
    folder = root / 'latest'
    if previous == 'legacy_payload':
        folder.mkdir()
        torch.save(old, folder / 'payload.pt')
    elif previous != 'missing':
        system.save_checkpoint(old, 'latest')
    if previous == 'legacy':
        bundle.unlink()
    original_bytes = bundle.read_bytes() if previous == 'bundle' else None
    save = torch.save
    replace = Path.replace
    rename = Path.rename
    publishing_new = False

    def fail_save(value, path, *args, **kwargs):
        nonlocal publishing_new
        publishing_new |= value == new
        if publishing_new and failure == 'serialize' and Path(path).name in ('.latest.pt.writing', 'latest.pt'):
            Path(path).write_bytes(b'partial bundle')
            raise OSError('injected serialize failure')
        return save(value, path, *args, **kwargs)

    def fail_replace(path, target):
        if (
            failure == 'mirror' and path.name == 'meta.json' and path.parent.name == '.latest.writing'
            or failure == 'bundle' and publishing_new and path.name == '.latest.pt.writing'
        ):
            raise PermissionError(f'injected {failure} failure')
        return replace(path, target)

    def fail_rename(path, target):
        if failure == 'mirror' and path.name == '.latest.writing':
            raise PermissionError('injected mirror failure')
        return rename(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(torch, 'save', fail_save)
        patch.setattr(Path, 'replace', fail_replace)
        patch.setattr(Path, 'rename', fail_rename)
        with pytest.raises(OSError, match='injected'):
            system.save_checkpoint(new, 'latest')

    if previous == 'bundle':
        assert bundle.read_bytes() == original_bytes
    elif previous == 'missing':
        assert not bundle.exists()
    else:
        assert bundle.is_file()  # Preserve the legacy snapshot before touching its only copy.
    for ref in ('latest', str(bundle), str(folder)):
        if previous != 'missing':
            assert system.load_checkpoint(ref) == old
        elif failure == 'serialize':
            assert system.load_checkpoint(ref) is None
        else:
            with pytest.raises(OSError, match='incomplete checkpoint'):
                system.load_checkpoint(ref)
    assert (root / '.latest.incomplete').exists() == (failure != 'serialize')

    system.save_checkpoint(new, 'latest')
    assert system.load_checkpoint('latest') == new
    assert system.load_checkpoint(str(folder)) == new
    assert not (root / '.latest.incomplete').exists()
    assert not (folder / 'optimizer.pt').exists()
    assert not (folder / 'payload.pt').exists()
    if previous == 'legacy_payload':
        import shutil

        exported = Path(shutil.copytree(folder, tmp_path / 'exported'))
        assert system.load_checkpoint(str(exported)) == new
    bundle.unlink()
    assert system.load_checkpoint(str(folder)) == new  # Legacy pieces-only reader stays supported.


def test_committed_checkpoint_survives_marker_cleanup_failure(tmp_path, monkeypatch):
    system = SystemFactory.build(SystemConfig.from_mapping({'device': 'cpu'}), tmp_path)
    unlink = Path.unlink

    def fail_marker_cleanup(path, *args, **kwargs):
        if path.name == '.latest.incomplete':
            raise PermissionError('injected marker cleanup failure')
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'unlink', fail_marker_cleanup)
    payload = {'step': 4, 'model': {'w': 2}}
    bundle = system.save_checkpoint(payload, 'latest')
    assert system.load_checkpoint('latest') == payload
    assert system.load_checkpoint(str(bundle.with_suffix(''))) == payload
    assert (bundle.parent / '.latest.incomplete').exists()


def test_checkpoint_mirror_removes_stale_meta_format(tmp_path):
    system = SystemFactory.build(SystemConfig.from_mapping({'device': 'cpu'}), tmp_path)
    bundle = system.save_checkpoint({'step': 1, 'extra': {1, 2}}, 'latest')
    folder = bundle.with_suffix('')
    assert (folder / 'meta.pt').is_file()
    payload = {'step': 2, 'model': {'w': 3}, 'tracker': {'step': 2}, 'logger': {'path': 'run.log'}}
    system.save_checkpoint(payload, 'latest')
    assert not (folder / 'meta.pt').exists()
    bundle.unlink()
    assert system.load_checkpoint('latest') == payload


@pytest.mark.parametrize('name', ['', '.', '..'])
def test_checkpoint_rejects_unsafe_stem(tmp_path, name):
    system = SystemFactory.build(SystemConfig.from_mapping({'device': 'cpu'}), tmp_path)
    with pytest.raises(ValueError, match='checkpoint name'):
        system.save_checkpoint({'step': 1}, name)
    assert system.logger.path.is_file()


@pytest.mark.parametrize('stage', ['mirror', 'bundle'])
@pytest.mark.parametrize('clears', [True, False])
def test_checkpoint_windows_denial_commits_or_preserves_old_bundle(tmp_path, monkeypatch, stage, clears):
    system = SystemFactory.build(SystemConfig.from_mapping({'device': 'cpu'}), tmp_path)
    old, new = {'step': 3, 'model': {'w': 1}}, {'step': 4, 'model': {'w': 2}}
    bundle = system.save_checkpoint(old, 'latest')
    old_bytes = bundle.read_bytes()
    replace, calls = Path.replace, []

    def denied(path, target):
        affected = (stage == 'mirror' and path.name == 'meta.json' and path.parent.name == '.latest.writing'
                    or stage == 'bundle' and path.name == '.latest.pt.writing')
        if affected:
            calls.append(path)
            if not clears or len(calls) <= 2:
                error = PermissionError('Windows file held')
                error.winerror = 5
                raise error
        return replace(path, target)

    monkeypatch.setattr(Path, 'replace', denied)
    monkeypatch.setattr('rpipe.structure.artifact._atomic.time.sleep', lambda _: None)
    if clears:
        assert system.save_checkpoint(new, 'latest') == bundle
        assert len(calls) == 3
        assert system.load_checkpoint('latest') == new
        assert system.load_checkpoint(str(bundle.with_suffix(''))) == new
    else:
        with pytest.raises(PermissionError):
            system.save_checkpoint(new, 'latest')
        assert len(calls) == 6 and bundle.read_bytes() == old_bytes
        assert system.load_checkpoint('latest') == old
        assert system.load_checkpoint(str(bundle.with_suffix(''))) == old
