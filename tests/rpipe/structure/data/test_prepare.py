import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.content,
    pytest.mark.p1,
    pytest.mark.structure_layer,
    pytest.mark.module_data,
]

from pathlib import Path

from rpipe.structure.artifact import artifact_layout, write_config
from rpipe.structure.data.prepare import prepare_shared_data


def test_prepare_skips_stub(tmp_path: Path, monkeypatch):
    calls: list[str] = []

    def fake_build(cfg, root, seed=None, origin=None):
        calls.append(str(cfg.name))
        return None

    monkeypatch.setattr('rpipe.structure.data.prepare.DataFactory.build', fake_build)
    layout = artifact_layout(tmp_path, 'r0')
    write_config(layout.config_path, {'id': 'r0', 'data': {'name': 'Toy', 'source': 'stub'}})
    assert prepare_shared_data(tmp_path, [layout.config_path]) == []
    assert calls == []


def test_prepare_skips_existing_cache(tmp_path: Path, monkeypatch):
    calls: list[str] = []

    def fake_build(cfg, root, seed=None, origin=None):
        calls.append(str(cfg.name))
        return None

    monkeypatch.setattr('rpipe.structure.data.prepare.DataFactory.build', fake_build)
    layout = artifact_layout(tmp_path, 'r0')
    write_config(
        layout.config_path,
        {'id': 'r0', 'data': {'name': 'MNIST', 'source': 'torch', 'config': {'batch_size': 64}}},
    )
    cached = tmp_path / 'shared' / 'data' / 'mnist'
    cached.mkdir(parents=True)
    (cached / '.ready').write_text('foreign\n', encoding='utf-8')
    assert prepare_shared_data(tmp_path, [layout.config_path]) == []
    assert calls == []


def test_prepare_once_per_name_source(tmp_path: Path, monkeypatch):
    calls: list[tuple] = []

    def fake_build(cfg, root, seed=None, origin=None):
        calls.append((cfg.name, cfg.source, cfg.config.get('train_size'), Path(root).name))
        return None

    monkeypatch.setattr('rpipe.structure.data.prepare.DataFactory.build', fake_build)
    paths = []
    for i, size in enumerate((500, 2000, 500)):
        layout = artifact_layout(tmp_path, f'r{i}')
        write_config(
            layout.config_path,
            {
                'id': f'r{i}',
                'data': {
                    'name': 'MNIST',
                    'source': 'torch',
                    'config': {'train_size': size, 'batch_size': 64},
                },
            },
        )
        paths.append(layout.config_path)
    names = prepare_shared_data(tmp_path, paths)
    assert names == ['MNIST']
    assert len(calls) == 1
    assert calls[0] == ('MNIST', 'torch', None, 'data')


def test_prepare_announces_download_before_build(tmp_path: Path, monkeypatch, capsys):
    seen: dict[str, str] = {}

    def fake_build(cfg, root, seed=None, origin=None):
        seen['during'] = (tmp_path / 'activity.json').read_text(encoding='utf-8')
        return None

    monkeypatch.setattr('rpipe.structure.data.prepare.DataFactory.build', fake_build)
    layout = artifact_layout(tmp_path, 'r0')
    write_config(
        layout.config_path,
        {'id': 'r0', 'data': {'name': 'MNIST', 'source': 'torch', 'config': {'batch_size': 64}}},
    )
    assert prepare_shared_data(tmp_path, [layout.config_path]) == ['MNIST']
    out = capsys.readouterr().out
    assert 'make: shared MNIST download' in out
    assert 'foreign' in out
    assert 'make: shared MNIST ready' in out
    assert 'download' in seen['during']
    assert (tmp_path / 'shared' / 'data' / 'mnist' / '.ready').read_text(encoding='utf-8').strip() == 'foreign'


def test_prepare_retries_partial_archive(tmp_path: Path, monkeypatch, capsys):
    calls: list[str] = []

    def fake_build(cfg, root, seed=None, origin=None):
        calls.append(origin or '')
        return None

    monkeypatch.setattr('rpipe.structure.data.prepare.DataFactory.build', fake_build)
    layout = artifact_layout(tmp_path, 'r0')
    write_config(
        layout.config_path,
        {
            'id': 'r0',
            'origin': 'domestic',
            'data': {
                'name': 'CIFAR10',
                'source': 'torch',
                'config': {'batch_size': 64},
            },
        },
    )
    partial = tmp_path / 'shared' / 'data' / 'cifar10'
    partial.mkdir(parents=True)
    (partial / 'cifar-10-python.tar.gz').write_bytes(b'not-a-full-archive')
    assert prepare_shared_data(tmp_path, [layout.config_path]) == ['CIFAR10']
    assert calls == ['domestic']
    out = capsys.readouterr().out
    assert 'download domestic' in out
    assert 'dataset.bj.bcebos.com' in out
    assert 'cached' not in out
