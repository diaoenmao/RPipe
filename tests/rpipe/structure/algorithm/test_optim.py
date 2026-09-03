import pytest

from rpipe.structure.algorithm.config import AlgorithmConfig
from rpipe.structure.algorithm.optim import make_optimizer, make_scheduler
from rpipe.structure.algorithm.resume import resume_stem


def test_resume_stem_defaults():
    train = AlgorithmConfig.from_mapping({'mode': 'train'})
    eval_cfg = AlgorithmConfig.from_mapping({'mode': 'eval'})
    assert resume_stem(train, mode='train') == 'latest'
    assert resume_stem(eval_cfg, mode='eval') == 'best'
    off = AlgorithmConfig.from_mapping({'mode': 'train', 'resume': False})
    assert resume_stem(off, mode='train') is None
    named = AlgorithmConfig.from_mapping({'mode': 'eval', 'resume_from': 'step_000003'})
    assert resume_stem(named, mode='eval') == 'step_000003'


def test_make_optimizer_sgd_and_unknown():
    import torch

    module = torch.nn.Linear(1, 1)
    cfg = AlgorithmConfig.from_mapping({'mode': 'train', 'optimizer': 'SGD', 'lr': 0.2, 'momentum': 0.9})
    opt = make_optimizer(module, cfg)
    assert opt.param_groups[0]['lr'] == pytest.approx(0.2)
    assert opt.param_groups[0]['momentum'] == pytest.approx(0.9)
    with pytest.raises(ValueError, match='unknown optimizer'):
        make_optimizer(module, AlgorithmConfig.from_mapping({'mode': 'train', 'optimizer': 'nope'}))


def test_linear_scheduler_with_warmup():
    import torch

    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=0.1)
    cfg = AlgorithmConfig.from_mapping(
        {'mode': 'train', 'scheduler': 'linear', 'warmup_steps': 2}
    )
    sched = make_scheduler(opt, cfg, 10)
    assert sched is not None
    sched.step()
    assert opt.param_groups[0]['lr'] > 0
