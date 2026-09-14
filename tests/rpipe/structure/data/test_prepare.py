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

    def fake_build(cfg, root, seed=None):
        calls.append(str(cfg.name))
        return None

    monkeypatch.setattr('rpipe.structure.data.prepare.DataFactory.build', fake_build)
    layout = artifact_layout(tmp_path, 'r0')
    write_config(layout.config_path, {'id': 'r0', 'data': {'name': 'Toy', 'source': 'stub'}})
    assert prepare_shared_data(tmp_path, [layout.config_path]) == []
    assert calls == []


def test_prepare_skips_existing_cache(tmp_path: Path, monkeypatch):
    calls: list[str] = []

    def fake_build(cfg, root, seed=None):
        calls.append(str(cfg.name))
        return None

    monkeypatch.setattr('rpipe.structure.data.prepare.DataFactory.build', fake_build)
    layout = artifact_layout(tmp_path, 'r0')
    write_config(
        layout.config_path,
        {'id': 'r0', 'data': {'name': 'MNIST', 'source': 'torch', 'config': {'batch_size': 64}}},
    )
    cached = tmp_path / 'shared' / 'data' / 'MNIST'
    cached.mkdir(parents=True)
    (cached / 'ready').write_text('1', encoding='utf-8')
    assert prepare_shared_data(tmp_path, [layout.config_path]) == []
    assert calls == []


def test_prepare_once_per_name_source(tmp_path: Path, monkeypatch):
    calls: list[tuple] = []

    def fake_build(cfg, root, seed=None):
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
