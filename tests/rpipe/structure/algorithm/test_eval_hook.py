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


def test_on_eval_period_runs_eval_and_can_stop(tmp_path, monkeypatch):
    calls: list[int] = []

    def fake_eval(tracker, logger, data, model, system, extra=None):
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
