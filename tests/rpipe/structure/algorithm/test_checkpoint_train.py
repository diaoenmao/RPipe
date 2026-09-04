from pathlib import Path

from rpipe.structure.algorithm.config import AlgorithmConfig
from rpipe.structure.algorithm.tracker import AlgorithmTracker
from rpipe.structure.algorithm.train import TrainAlgorithm
from rpipe.structure.system.config import SystemConfig
from rpipe.structure.system.factory import SystemFactory


def _run(tmp_path: Path, mapping: dict) -> tuple[dict, Path]:
    import torch

    class _Data:
        name = 'CIFAR'
        meta = {'train_size': 8, 'batch_size': 4}

        def iter_batches(self, split: str):
            del split
            images = torch.randn(4, 1, 28, 28)
            targets = torch.zeros(4, dtype=torch.long)
            yield images, targets
            yield images, targets

    class _Model:
        def __init__(self) -> None:
            self.module = torch.nn.Linear(784, 10)

    algo = TrainAlgorithm(AlgorithmConfig.from_mapping(mapping))
    tracker = AlgorithmTracker(tmp_path)
    system = SystemFactory.build(SystemConfig.from_mapping({'device': 'cpu'}), tmp_path)
    out = algo.run(_Data(), _Model(), system, tracker)
    return out, tmp_path / 'checkpoints'


def test_epoch_train_writes_latest_only_by_default(tmp_path: Path):
    out, ckpt = _run(tmp_path, {'mode': 'train', 'num_epochs': 2, 'eval_period': 1})
    assert out['epochs'] == 2
    assert out['progress_unit'] == 'step'
    assert out['steps'] == 4
    assert (ckpt / 'latest.pt').is_file()
    assert (ckpt / 'latest' / 'model.pt').is_file()
    assert (ckpt / 'latest' / 'optimizer.pt').is_file()
    assert (ckpt / 'latest' / 'tracker.json').is_file()
    assert not (ckpt / 'best.pt').is_file()
    assert list(ckpt.glob('epoch_*.pt')) == []


def test_save_best_writes_best_pt(tmp_path: Path):
    out, ckpt = _run(
        tmp_path,
        {'mode': 'train', 'num_epochs': 2, 'eval_period': 1, 'save_best': True},
    )
    assert out['best_accuracy'] is not None
    assert (ckpt / 'latest.pt').is_file()
    assert (ckpt / 'best.pt').is_file()


def test_percent_keeps_named_epoch_snapshots(tmp_path: Path):
    _out, ckpt = _run(
        tmp_path,
        {
            'mode': 'train',
            'num_epochs': 4,
            'eval_period': 1,
            'checkpoint': 'percent',
            'checkpoint_percents': [0.5, 1.0],
        },
    )
    assert (ckpt / 'latest.pt').is_file()
    assert (ckpt / 'step_000004.pt').is_file()
    assert (ckpt / 'step_000008.pt').is_file()
    assert not (ckpt / 'step_000002.pt').is_file()


def test_num_steps_stops_and_names_step_snapshots(tmp_path: Path):
    out, ckpt = _run(
        tmp_path,
        {
            'mode': 'train',
            'num_steps': 3,
            'eval_period': 3,
            'checkpoint': 'percent',
            'checkpoint_percents': [1.0],
        },
    )
    assert out['progress_unit'] == 'step'
    assert out['steps'] == 3
    assert out['epochs'] == 2
    assert (ckpt / 'latest.pt').is_file()
    assert (ckpt / 'step_000003.pt').is_file()


def test_resume_restores_tracker_history(tmp_path: Path):
    mapping = {
        'mode': 'train',
        'num_epochs': 1,
        'progress_unit': 'epoch',
        'eval_period': 1,
        'checkpoint_period': 1,
    }
    _run(tmp_path, mapping)
    import json

    body = json.loads((tmp_path / 'checkpoints' / 'latest' / 'tracker.json').read_text(encoding='utf-8'))
    assert 'splits' in body
    second, _ = _run(tmp_path, mapping)
    assert second['epochs'] == 1


def test_step_period_accumulates_before_optimizer_step(tmp_path: Path):
    out, _ = _run(
        tmp_path,
        {
            'mode': 'train',
            'num_epochs': 1,
            'progress_unit': 'epoch',
            'eval_period': 1,
            'step_period': 2,
        },
    )
    assert out['steps'] == 1


def test_resume_latest_skips_when_budget_already_done(tmp_path: Path):
    mapping = {
        'mode': 'train',
        'num_epochs': 1,
        'progress_unit': 'epoch',
        'eval_period': 1,
        'checkpoint': 'latest',
        'checkpoint_period': 1,
    }
    first, _ = _run(tmp_path, mapping)
    assert first['epochs'] == 1
    second, _ = _run(tmp_path, mapping)
    assert second['epochs'] == 1
    assert second['steps'] == first['steps']


def test_eval_algorithm_loads_best(tmp_path: Path):
    from rpipe.structure.algorithm.eval import EvalAlgorithm

    mapping = {
        'mode': 'train',
        'num_epochs': 1,
        'progress_unit': 'epoch',
        'eval_period': 1,
        'save_best': True,
    }
    trained, ckpt = _run(tmp_path, mapping)
    assert (ckpt / 'best.pt').is_file()
    import torch

    class _Data:
        name = 'CIFAR'
        meta = {'train_size': 8, 'batch_size': 4}

        def iter_batches(self, split: str):
            del split
            images = torch.randn(4, 1, 28, 28)
            targets = torch.zeros(4, dtype=torch.long)
            yield images, targets

    class _Model:
        def __init__(self) -> None:
            self.module = torch.nn.Linear(784, 10)

    system = SystemFactory.build(SystemConfig.from_mapping({'device': 'cpu'}), tmp_path)
    tracker = AlgorithmTracker(tmp_path / 'eval_tracker')
    out = EvalAlgorithm(AlgorithmConfig.from_mapping({'mode': 'eval'})).run(
        _Data(), _Model(), system, tracker
    )
    assert out['mode'] == 'eval'
    assert 'accuracy' in out
    assert out['resume_stem'] == 'best'


def test_eval_algorithm_missing_checkpoint_fails(tmp_path: Path):
    from rpipe.structure.algorithm.eval import EvalAlgorithm
    import pytest
    import torch

    class _Data:
        name = 'CIFAR'
        meta = {}

        def iter_batches(self, split: str):
            del split
            yield torch.randn(2, 1, 28, 28), torch.zeros(2, dtype=torch.long)

    class _Model:
        module = torch.nn.Linear(784, 10)

    system = SystemFactory.build(SystemConfig.from_mapping({'device': 'cpu'}), tmp_path)
    tracker = AlgorithmTracker(tmp_path)
    with pytest.raises(FileNotFoundError, match='eval resume missing'):
        EvalAlgorithm(AlgorithmConfig.from_mapping({'mode': 'eval'})).run(
            _Data(), _Model(), system, tracker
        )
