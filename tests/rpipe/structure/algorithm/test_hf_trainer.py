import pytest

from rpipe.structure.algorithm.config import AlgorithmConfig
from rpipe.structure.algorithm.tracker import AlgorithmTracker
from rpipe.structure.system.config import SystemConfig
from rpipe.structure.system.factory import SystemFactory

pytest.importorskip('transformers')


def test_hf_trainer_runs_one_epoch(tmp_path):
    import torch

    from rpipe.structure.algorithm.transformers_trainer import HfTrainAlgorithm

    train_ds = torch.utils.data.TensorDataset(
        torch.randn(8, 1, 28, 28),
        torch.zeros(8, dtype=torch.long),
    )
    test_ds = torch.utils.data.TensorDataset(
        torch.randn(4, 1, 28, 28),
        torch.zeros(4, dtype=torch.long),
    )

    class _Loader:
        def __init__(self, dataset, steps: int) -> None:
            self.dataset = dataset
            self._steps = steps

        def __len__(self) -> int:
            return self._steps

        def __iter__(self):
            for i in range(self._steps):
                x, y = self.dataset[i % len(self.dataset)]
                yield x.unsqueeze(0), y.unsqueeze(0)

    class _Data:
        name = 'CIFAR'
        meta = {'train_size': 8, 'batch_size': 4, 'seed': 0}

        def __init__(self) -> None:
            self._loaders = {'train': _Loader(train_ds, 2), 'test': _Loader(test_ds, 1)}

        def iter_batches(self, split: str):
            yield from self._loaders[split]

        def steps_per_epoch(self, split: str = 'train'):
            del split
            return 2

    class _Model:
        def __init__(self) -> None:
            self.module = torch.nn.Linear(784, 10)

    algo = HfTrainAlgorithm(
        AlgorithmConfig.from_mapping(
            {
                'mode': 'train',
                'source': 'transformers_trainer',
                'num_epochs': 1,
                'progress_unit': 'epoch',
                'eval_period': 1,
                'optimizer': 'SGD',
                'lr': 0.1,
                'scheduler': 'cosine',
            }
        )
    )
    system = SystemFactory.build(SystemConfig.from_mapping({'device': 'cpu'}), tmp_path)
    out = algo.run(_Data(), _Model(), system, AlgorithmTracker(tmp_path))
    assert out['source'] == 'transformers_trainer'
    assert out['epochs'] >= 1
    assert (tmp_path / 'checkpoints' / 'latest.pt').is_file()
