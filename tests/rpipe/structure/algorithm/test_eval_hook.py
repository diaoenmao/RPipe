import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.content,
    pytest.mark.p1,
    pytest.mark.structure_layer,
    pytest.mark.module_algorithm,
]

from rpipe.structure.algorithm.base import Algorithm
from rpipe.structure.algorithm.config import AlgorithmConfig
from rpipe.structure.algorithm.hook import AlgorithmHook
from rpipe.structure.algorithm.eval_hook import should_early_stop
from rpipe.structure.algorithm.tracker import AlgorithmTracker
from rpipe.structure.algorithm.train import TrainAlgorithm, due_eval_period


class _Model:
    module = object()


def test_algorithm_is_algorithm_hook():
    assert issubclass(Algorithm, AlgorithmHook)
    algo = Algorithm(AlgorithmConfig.from_mapping({'mode': 'train'}))
    assert isinstance(algo, AlgorithmHook)


def test_eval_period_default_every_epoch():
    algo = Algorithm(AlgorithmConfig.from_mapping({'mode': 'train'}))
    assert algo.eval_period() == 1
    assert algo.progress_unit() == 'step'
    assert algo.checkpoint_mode() == 'latest'
    assert algo.save_best() is False
    algo = Algorithm(AlgorithmConfig.from_mapping({'mode': 'train', 'eval_period': 0}))
    assert algo.eval_period() == 0


def test_due_eval_period_cadence():
    assert due_eval_period(1, 1) is True
    assert due_eval_period(2, 1) is False
    assert due_eval_period(2, 2) is True
    assert due_eval_period(0, 1) is False


def test_base_on_eval_period_is_noop(tmp_path):
    algo = Algorithm(AlgorithmConfig.from_mapping({'mode': 'train'}))
    tracker = AlgorithmTracker(tmp_path)
    assert algo.on_eval_period(tracker, None, None, None, None, {'epoch': 1}) is False


def test_train_hook_early_stop_patience():
    stop, best, stall = should_early_stop(
        accuracy=0.5, best=0.8, stall=1, patience=2, min_delta=0.0
    )
    assert stop is True
    assert best == 0.8
    assert stall == 2


def test_min_mode_improves_on_lower_loss():
    stop, best, stall = should_early_stop(
        value=0.4, best=0.8, stall=0, patience=None, min_delta=0.0, mode='min'
    )
    assert stop is False
    assert best == 0.4
    assert stall == 0


def test_best_updates_without_patience():
    stop, best, stall = should_early_stop(
        accuracy=0.9, best=0.8, stall=0, patience=None, min_delta=0.0
    )
    assert stop is False
    assert best == 0.9
    assert stall == 0


def test_eval_num_steps_caps_batches(tmp_path):
    import torch

    from rpipe.structure.algorithm.config import AlgorithmConfig
    from rpipe.structure.algorithm.eval_hook import eval_batch_limit, eval_test_split
    from rpipe.structure.algorithm.tracker import AlgorithmTracker

    class _Data:
        def iter_batches(self, split):
            del split
            for _ in range(4):
                yield torch.zeros(2, 3), torch.zeros(2, dtype=torch.long)

    class _Model:
        def __init__(self):
            self.module = torch.nn.Linear(3, 2)
            self.calls = 0
            inner = self.module.forward

            def counted(x):
                self.calls += 1
                return inner(x)

            self.module.forward = counted

        def eval(self):
            return self.module.eval()

        def train(self):
            return self.module.train()

    class _System:
        device = 'cpu'

    assert eval_batch_limit(AlgorithmConfig.from_mapping({})) is None
    assert eval_batch_limit(AlgorithmConfig.from_mapping({'eval_num_steps': -1})) is None
    assert eval_batch_limit(AlgorithmConfig.from_mapping({'eval_num_steps': 2})) == 2

    model = _Model()
    eval_test_split(AlgorithmTracker(tmp_path), None, _Data(), model, _System(), num_steps=2)
    assert model.calls == 2


def test_on_eval_period_runs_eval_and_can_stop(tmp_path, monkeypatch):
    calls: list[int] = []

    def fake_eval(tracker, logger, data, model, system, extra=None, **_kwargs):
        calls.append(1)
        return {'Accuracy': 0.4}

    monkeypatch.setattr('rpipe.structure.algorithm.train.eval_test_split', fake_eval)
    algo = TrainAlgorithm(
        AlgorithmConfig.from_mapping(
            {'mode': 'train', 'early_stop_patience': 1, 'early_stop_min_delta': 1.0}
        )
    )
    tracker = AlgorithmTracker(tmp_path)
    first = algo.on_eval_period(tracker, None, None, _Model(), None, {'epoch': 1})
    second = algo.on_eval_period(tracker, None, None, _Model(), None, {'epoch': 2})
    assert calls == [1, 1]
    assert first is False
    assert second is True
