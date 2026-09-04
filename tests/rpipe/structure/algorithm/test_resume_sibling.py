from pathlib import Path

from rpipe.structure.algorithm.config import AlgorithmConfig
from rpipe.structure.algorithm.eval import EvalAlgorithm
from rpipe.structure.algorithm.resume import sibling_train_checkpoint
from rpipe.structure.algorithm.tracker import AlgorithmTracker
from rpipe.structure.artifact import build_index, write_index
from rpipe.structure.system.config import SystemConfig
from rpipe.structure.system.factory import SystemFactory


def test_eval_run_loads_sibling_train_best(tmp_path: Path):
    import torch

    study = tmp_path
    train_id = 'trainrun'
    eval_id = 'evalrun'
    train_assets = study / 'runs' / train_id / 'assets'
    eval_assets = study / 'runs' / eval_id / 'assets'
    train_assets.mkdir(parents=True)
    eval_assets.mkdir(parents=True)
    train_system = SystemFactory.build(SystemConfig.from_mapping({'device': 'cpu'}), train_assets)
    module = torch.nn.Linear(784, 10)
    with torch.no_grad():
        module.weight.fill_(0.02)
        module.bias.zero_()
    train_system.save_checkpoint(
        {'model': module.state_dict(), 'epoch': 2, 'step': 4, 'best_accuracy': 0.77},
        'best',
    )
    write_index(
        study,
        build_index(
            study='demo',
            description='',
            experiments=[
                {
                    'factors': {'data.config.train_size': 8, 'algorithm.mode': 'train'},
                    'runs': [{'id': train_id, 'seed': 0, 'run_dir': train_id}],
                },
                {
                    'factors': {'data.config.train_size': 8, 'algorithm.mode': 'eval'},
                    'runs': [{'id': eval_id, 'seed': 0, 'run_dir': eval_id}],
                },
            ],
        ),
    )
    eval_system = SystemFactory.build(SystemConfig.from_mapping({'device': 'cpu'}), eval_assets)
    found = sibling_train_checkpoint(eval_system, stem='best')
    assert found is not None
    assert found.is_file()

    class _Data:
        name = 'Other'
        meta = {'train_size': 8, 'batch_size': 4}

        def iter_batches(self, split: str):
            del split
            yield torch.randn(4, 1, 28, 28), torch.zeros(4, dtype=torch.long)

    class _Model:
        def __init__(self) -> None:
            self.module = torch.nn.Linear(784, 10)

    tracker = AlgorithmTracker(eval_assets)
    out = EvalAlgorithm(AlgorithmConfig.from_mapping({'mode': 'eval'})).run(
        _Data(), _Model(), eval_system, tracker
    )
    assert out['mode'] == 'eval'
    assert 'accuracy' in out
