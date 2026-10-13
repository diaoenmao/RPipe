"""Two Studies reuse input files while complete native Runs retain independent state."""
from pathlib import Path
import hashlib
import json

import pytest

from rpipe.flow.cli import main
from rpipe.structure.artifact.shared import shared_paths
from rpipe.structure.data import Data, DataRegistry
from rpipe.structure.model import Model, ModelRegistry

pytestmark = [pytest.mark.integration, pytest.mark.content, pytest.mark.p1,
              pytest.mark.flow_layer, pytest.mark.module_runner,
              pytest.mark.cost(cost_class='c2'),
              pytest.mark.result_type('categorical', detail='summary')]


def test_two_native_studies_share_downloads_not_training_state(tmp_path, monkeypatch):
    """CLI make/run-one → shared builders → checkpoints: one raw/weight, two mutable states."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    def data_builder(cfg, root, seed=None):
        raw = root / 'offline-v1.bin'
        if not raw.exists():
            raw.write_bytes(b'fixed offline data')
        dataset = TensorDataset(torch.ones(4, 2), torch.zeros(4, dtype=torch.long))
        return Data(name=cfg.name, source=cfg.source,
                    loaders={'train': DataLoader(dataset, batch_size=2),
                             'test': DataLoader(dataset, batch_size=2)},
                    meta={'data_size': [2], 'target_size': 2, 'config': cfg.config})

    def model_builder(cfg, root, data_meta=None):
        path = root / 'offline-v1.weights'
        module = torch.nn.Linear(2, 2)
        if not path.exists():
            with torch.no_grad():
                module.weight.zero_()
                module.bias.zero_()
            torch.save(module.state_dict(), path)
        module.load_state_dict(torch.load(path, weights_only=True))
        return Model(name=cfg.name, source=cfg.source, module=module, meta={'input': str(path)})

    monkeypatch.setitem(DataRegistry._items, ('shared-offline', 'fixture'), data_builder)
    monkeypatch.setitem(ModelRegistry._items, ('shared-offline', 'fixture'), model_builder)
    results = []
    first_checkpoint = None
    for name, steps in [('a', 1), ('b', 2)]:
        study = tmp_path / 'studies' / name
        study.mkdir(parents=True)
        # Custom builders register during prepare, so make must not preconstruct them.
        (study / 'study.yaml').write_text('study: ' + name + '\nseeds: [0]\nflow: {prepare_shared: false}\n')
        (study / 'experiment_config.yaml').write_text(
            f'version: shared-flow-{name}\ndata: {{name: shared-offline, source: fixture}}\n'
            'model: {name: shared-offline, source: fixture}\n'
            f'algorithm: {{mode: train, num_steps: {steps}, checkpoint: latest, checkpoint_period: 1}}\n'
            'system: {device: cpu, deterministic: true}\n')
        assert main(['make', str(study), '--num-gpus', '0']) == 0
        index = json.loads((study / 'index.json').read_text())
        run = study / 'runs' / index['experiments'][0]['runs'][0]['id']
        assert main(['run-one', str(study), run.name]) == 0
        assert main(['process', str(study)]) == 0
        assert main(['report', str(study)]) == 0
        result = json.loads((run / 'result.json').read_text())
        assert result['status'] == 'succeeded'
        results.append(run)
        if first_checkpoint is None:
            first_checkpoint = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in run.glob('assets/checkpoints/*') if p.is_file()}
        else:
            assert all(hashlib.sha256(p.read_bytes()).hexdigest() == value
                       for p, value in first_checkpoint.items())
    assert first_checkpoint
    roots = shared_paths(tmp_path / 'studies' / 'a')
    assert roots == shared_paths(tmp_path / 'studies' / 'b')
    assert len(list(roots['data'].glob('*.bin'))) == 1
    assert len(list(roots['model'].glob('*.weights'))) == 1
    weights = torch.load(roots['model'] / 'offline-v1.weights', weights_only=True)
    assert weights['weight'].count_nonzero() == 0
    for run in results:
        checkpoint = torch.load(run / 'assets/checkpoints/latest.pt', weights_only=False)
        assert checkpoint['model']['weight'].count_nonzero() > 0


def test_shared_raw_does_not_share_study_normalization(tmp_path, monkeypatch):
    """prepare → actual vision Factory → result: private stats cannot contaminate a sibling."""
    from PIL import Image
    import torchvision.datasets

    class FakeMNIST:
        def __init__(self, root, train=True, download=False, transform=None):
            self.transform = transform
        def __len__(self):
            return 4
        def __getitem__(self, index):
            return self.transform(Image.new('L', (28, 28))), 0

    monkeypatch.setattr(torchvision.datasets, 'MNIST', FakeMNIST)
    foreign_stats = tmp_path / 'shared' / 'data' / 'mnist'
    foreign_stats.mkdir(parents=True)
    (foreign_stats / 'stats.yaml').write_text('mean: [9]\nstd: [1]\n')
    means = []
    for name in ('a', 'b'):
        study = tmp_path / 'studies' / name
        study.mkdir(parents=True)
        (study / 'study.yaml').write_text(f'study: {name}\nflow: {{prepare_shared: false}}\n')
        (study / 'experiment_config.yaml').write_text(
            f'version: profile-{name}\ndata: {{name: MNIST, source: torch}}\n'
            'model: {name: linear}\nalgorithm: {num_steps: 1}\nsystem: {device: cpu}\n')
        if name == 'a':
            local = study / 'shared' / 'data' / 'mnist'
            local.mkdir(parents=True)
            (local / 'stats.yaml').write_text('mean: [0.2]\nstd: [0.3]\n')
        assert main(['make', str(study), '--num-gpus', '0']) == 0
        index = json.loads((study / 'index.json').read_text())
        run = study / 'runs' / index['experiments'][0]['runs'][0]['id']
        assert main(['run-one', str(study), run.name]) == 0
        means.append(json.loads((run / 'result.json').read_text())['structure']['data']['mean'])
    assert means == [[0.2], [0.1307]]
    # A resolved input directory is an IO concern, not a new sampler identity.
    # The implicit profile must not alter declared config or old restore contracts.
    snapshot = json.loads((run / 'result.json').read_text())['structure']['data']
    assert 'stats_path' not in snapshot['config']


def test_profile_directory_move_keeps_native_sampler_restore_identity(tmp_path, monkeypatch):
    """Shared vision Factory → saved cursor → new profile path: unseen batches stay equal."""
    from PIL import Image
    import torch
    import torchvision.datasets
    from rpipe.structure.data import DataConfig, DataFactory

    class FakeMNIST:
        def __init__(self, root, train=True, download=False, transform=None):
            self.transform = transform
        def __len__(self):
            return 4
        def __getitem__(self, index):
            return self.transform(Image.new('L', (28, 28), color=index * 50)), 0

    monkeypatch.setattr(torchvision.datasets, 'MNIST', FakeMNIST)
    paths = [tmp_path / 'a' / 'stats.yaml', tmp_path / 'b' / 'stats.yaml']
    for path in paths:
        path.parent.mkdir()
        path.write_text('mean: [0.2]\nstd: [0.3]\n')
    config = DataConfig.from_mapping({'name': 'MNIST', 'source': 'torch',
                                     'config': {'batch_size': 2, 'augment': False}})
    first = DataFactory.build(config, tmp_path / 'shared' / 'data', seed=11, profile_path=paths[0])
    first.rebind_train_steps(step=0, num_steps=3)
    iterator = first.iter_batches('train')
    next(iterator)
    state = first.training_state()
    expected = list(iterator)
    second = DataFactory.build(config, tmp_path / 'shared' / 'data', seed=11, profile_path=paths[1])
    second.rebind_train_steps(step=0, num_steps=3)
    assert second.restore_training_state(state)
    actual = list(second.iter_batches('train'))
    assert len(actual) == len(expected) == 2
    assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(actual, expected))
    assert first.meta['config'] == second.meta['config'] == config.config
