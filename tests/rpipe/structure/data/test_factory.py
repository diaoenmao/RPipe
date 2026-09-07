from pathlib import Path

from rpipe.structure.api import data_api
from rpipe.structure.data import DataConfig


def test_stub_source_does_not_download(tmp_path: Path):
    data = data_api.build(
        DataConfig.from_mapping({'name': 'MNIST', 'source': 'stub'}),
        tmp_path,
    )
    assert data.source == 'stub'
    assert data.meta.get('stub') is True
    assert not (tmp_path / 'mnist').exists()


def test_rebind_train_steps_uses_remaining_prefix():
    import torch
    from torch.utils.data import TensorDataset

    from rpipe.structure.data.factory import Data

    dataset = TensorDataset(torch.arange(20), torch.zeros(20, dtype=torch.long))
    data = Data(name='toy', source='x', loaders={}, meta={'batch_size': 2, 'seed': 0})
    data._train_set = dataset
    data.rebind_train_steps(step=0, num_steps=4, step_period=1)
    assert len(data._loaders['train']) == 4
    full = [batch[0].clone() for batch in data.iter_batches('train')]
    data.rebind_train_steps(step=2, num_steps=4, step_period=1)
    rest = [batch[0].clone() for batch in data.iter_batches('train')]
    assert len(rest) == 2
    assert torch.equal(rest[0], full[0])
    assert torch.equal(rest[1], full[1])
    data.rebind_train_steps(step=4, num_steps=4, step_period=1)
    assert len(data._loaders['train']) == 0


def test_cifar10_builder_uses_fake_dataset(tmp_path: Path, monkeypatch):
    from PIL import Image

    from rpipe.structure.api import data_api
    from rpipe.structure.data import DataConfig

    class _Fake:
        def __init__(self, root, train=True, download=False, transform=None):
            del root, download
            self.train = train
            self.transform = transform
            self._n = 8 if train else 4

        def __len__(self):
            return self._n

        def __getitem__(self, index):
            img = Image.new('RGB', (32, 32), color=(index % 256, 0, 0))
            if self.transform is not None:
                img = self.transform(img)
            return img, 0

    import torchvision.datasets as tv_datasets

    monkeypatch.setattr(tv_datasets, 'CIFAR10', _Fake)
    data = data_api.build(
        DataConfig.from_mapping(
            {
                'name': 'CIFAR10',
                'source': 'torch',
                'config': {'batch_size': 4, 'train_size': 8, 'augment': False},
            }
        ),
        tmp_path,
        seed=0,
    )
    assert data.meta['data_size'] == [3, 32, 32]
    assert data.meta['train_size'] == 8
    images, targets = next(iter(data.iter_batches('train')))
    assert tuple(images.shape) == (4, 3, 32, 32)
    assert tuple(targets.shape) == (4,)


def test_svhn_builder_uses_fake_dataset(tmp_path: Path, monkeypatch):
    from PIL import Image

    from rpipe.structure.api import data_api
    from rpipe.structure.data import DataConfig

    class _Fake:
        def __init__(self, root, split='train', download=False, transform=None):
            del root, download
            self.transform = transform
            self._n = 6 if split == 'train' else 3

        def __len__(self):
            return self._n

        def __getitem__(self, index):
            img = Image.new('RGB', (32, 32))
            if self.transform is not None:
                img = self.transform(img)
            return img, 1

    import torchvision.datasets as tv_datasets

    monkeypatch.setattr(tv_datasets, 'SVHN', _Fake)
    data = data_api.build(
        DataConfig.from_mapping({'name': 'SVHN', 'source': 'torch', 'config': {'batch_size': 3, 'augment': False}}),
        tmp_path,
        seed=0,
    )
    assert data.name == 'SVHN'
    images, _ = next(iter(data.iter_batches('test')))
    assert tuple(images.shape[1:]) == (3, 32, 32)

    import torch
    from torch.utils.data import TensorDataset

    from rpipe.structure.data.factory import Data

    dataset = TensorDataset(torch.arange(20), torch.zeros(20, dtype=torch.long))
    data = Data(name='toy', source='x', loaders={}, meta={'batch_size': 2, 'seed': 0})
    data._train_set = dataset
    data.rebind_train_steps(step=0, num_steps=4, step_period=1)
    assert len(data._loaders['train']) == 4
    full = [batch[0].clone() for batch in data.iter_batches('train')]
    data.rebind_train_steps(step=2, num_steps=4, step_period=1)
    rest = [batch[0].clone() for batch in data.iter_batches('train')]
    assert len(rest) == 2
    assert torch.equal(rest[0], full[0])
    assert torch.equal(rest[1], full[1])
    data.rebind_train_steps(step=4, num_steps=4, step_period=1)
    assert len(data._loaders['train']) == 0
