import pytest

from rpipe.structure.algorithm.config import AlgorithmConfig
from rpipe.structure.algorithm.progress import (
    checkpoint_names,
    crossed_percents,
    due_period,
    infer_steps_per_epoch,
    parse_checkpoint_mode,
    resolve_budget,
    snapshot_name,
)


def test_budget_defaults_to_one_step():
    budget = resolve_budget(AlgorithmConfig.from_mapping({'mode': 'train'}))
    assert budget.unit == 'step'
    assert budget.num_epochs is None
    assert budget.num_steps == 1
    assert budget.total == 1


def test_budget_infers_step_when_only_num_steps():
    budget = resolve_budget(AlgorithmConfig.from_mapping({'mode': 'train', 'num_steps': 100}))
    assert budget.unit == 'step'
    assert budget.num_steps == 100
    assert budget.num_epochs is None
    assert budget.total == 100
    assert budget.scheduler_t_max() == 100


def test_budget_epochs_overwrite_steps_when_inferred():
    budget = resolve_budget(
        AlgorithmConfig.from_mapping({'mode': 'train', 'num_epochs': 20, 'num_steps': 1000}),
        steps_per_epoch=5,
    )
    assert budget.unit == 'step'
    assert budget.num_steps == 100
    assert budget.total == 100
    assert budget.steps_from_epochs is True
    assert budget.steps_per_epoch == 5
    assert budget.progress(epoch=3, step=50) == 50


def test_budget_can_still_use_epoch_unit():
    cfg = AlgorithmConfig.from_mapping({'mode': 'train', 'num_epochs': 3, 'progress_unit': 'epoch'})
    budget = resolve_budget(cfg, steps_per_epoch=4)
    assert budget.unit == 'epoch'
    assert budget.total == 3
    assert budget.num_steps == 12


def test_infer_steps_per_epoch_from_meta():
    data = type('D', (), {'meta': {'train_size': 130, 'batch_size': 64}})()
    assert infer_steps_per_epoch(data) == 3


def test_infer_steps_per_epoch_prefers_data_method():
    data = type(
        'D',
        (),
        {'meta': {'train_size': 130, 'batch_size': 64}, 'steps_per_epoch': lambda self=None: 9},
    )()
    assert infer_steps_per_epoch(data) == 9


def test_due_period_and_percent_crossing():
    assert due_period(1, 1) is True
    assert due_period(2, 1) is False
    assert due_period(0, 4) is False
    assert crossed_percents(5, 20, (0.25, 0.5, 1.0), set()) == [0.25]
    assert crossed_percents(5, 20, (0.25, 0.5), {0.25}) == []
    assert snapshot_name('epoch', 5) == 'epoch_0005'
    assert snapshot_name('step', 100) == 'step_000100'


def test_checkpoint_names_default_latest_only():
    assert checkpoint_names(
        mode='latest',
        save_best=False,
        period=1,
        percents=(0.5, 1.0),
        current=1,
        total=4,
        unit='epoch',
        improved=True,
        is_last=False,
    ) == ['latest']


def test_checkpoint_names_best_and_percent():
    already: set[float] = set()
    names = checkpoint_names(
        mode='percent',
        save_best=True,
        period=2,
        percents=(0.5, 1.0),
        current=2,
        total=4,
        unit='epoch',
        improved=True,
        is_last=False,
        already_percent=already,
    )
    assert names == ['latest', 'epoch_0002', 'best']
    parse_checkpoint_mode('percent')
    with pytest.raises(ValueError):
        parse_checkpoint_mode('all')
