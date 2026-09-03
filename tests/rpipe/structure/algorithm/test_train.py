import pytest

from rpipe.structure.algorithm.config import AlgorithmConfig
from rpipe.structure.algorithm.train import TrainAlgorithm, make_scheduler
from rpipe.structure.algorithm.tracker import AlgorithmTracker


class _Data:
    name = 'Toy'


def test_two_epoch_stub_train_does_not_raise(tmp_path):
    algo = TrainAlgorithm(AlgorithmConfig.from_mapping({'mode': 'train', 'num_epochs': 2}))
    tracker = AlgorithmTracker(tmp_path)
    out = algo.run(_Data(), None, None, tracker)
    assert out['mode'] == 'train'
    assert tracker.segment_mean('train').get('Loss') == 0.0


def test_make_scheduler_none_or_constant():
    cfg = AlgorithmConfig.from_mapping({'mode': 'train'})
    assert make_scheduler(object(), cfg, 20) is None
    cfg = AlgorithmConfig.from_mapping({'mode': 'train', 'scheduler': 'constant'})
    assert make_scheduler(object(), cfg, 20) is None


def test_make_scheduler_unknown_raises():
    cfg = AlgorithmConfig.from_mapping({'mode': 'train', 'scheduler': 'nope'})
    with pytest.raises(ValueError, match='unknown scheduler'):
        make_scheduler(object(), cfg, 20)


def test_cosine_scheduler_decays_to_eta_min():
    import torch

    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=0.1)
    cfg = AlgorithmConfig.from_mapping(
        {'mode': 'train', 'scheduler': 'cosine', 'eta_min': 0.0}
    )
    sched = make_scheduler(opt, cfg, 20)
    assert sched is not None
    first = opt.param_groups[0]['lr']
    for _ in range(20):
        sched.step()
    last = opt.param_groups[0]['lr']
    assert first == pytest.approx(0.1)
    assert last == pytest.approx(0.0, abs=1e-6)
