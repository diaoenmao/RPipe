import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.content,
    pytest.mark.p2,
    pytest.mark.structure_layer,
    pytest.mark.module_make,
    pytest.mark.cost(cost_class='c1'),
    pytest.mark.result_type('categorical', detail='summary'),
]

from rpipe.structure.make.estimate import estimate_job_bytes, estimate_job_seconds


def test_estimate_resnet_heavier_than_linear():
    linear = estimate_job_bytes(
        {
            'data': {'name': 'MNIST', 'config': {'batch_size': 64}},
            'model': {'name': 'linear'},
            'algorithm': {'mode': 'train'},
        }
    )
    resnet = estimate_job_bytes(
        {
            'data': {'name': 'CIFAR10', 'config': {'batch_size': 250}},
            'model': {'name': 'resnet18'},
            'algorithm': {'mode': 'train'},
        }
    )
    eval_resnet = estimate_job_bytes(
        {
            'data': {'name': 'CIFAR10', 'config': {'batch_size': 250}},
            'model': {'name': 'resnet18'},
            'algorithm': {'mode': 'eval'},
        }
    )
    assert resnet > linear * 5
    assert eval_resnet < resnet
    assert eval_resnet > linear

def test_estimate_seconds_resnet_heavier_and_eval_lighter():
    linear = estimate_job_seconds(
        {
            'data': {'name': 'MNIST', 'config': {'batch_size': 250, 'train_size': 500}},
            'model': {'name': 'linear'},
            'algorithm': {'mode': 'train', 'num_epochs': 20},
        }
    )
    resnet = estimate_job_seconds(
        {
            'data': {'name': 'CIFAR10', 'config': {'batch_size': 250, 'train_size': 50000}},
            'model': {'name': 'resnet18'},
            'algorithm': {'mode': 'train', 'num_epochs': 20},
        }
    )
    eval_resnet = estimate_job_seconds(
        {
            'data': {'name': 'CIFAR10', 'config': {'batch_size': 250, 'train_size': 50000}},
            'model': {'name': 'resnet18'},
            'algorithm': {'mode': 'eval', 'num_epochs': 20},
        }
    )
    assert resnet > linear
    assert eval_resnet < resnet
    assert eval_resnet >= 1
    short = estimate_job_seconds(
        {
            'data': {'name': 'MNIST', 'config': {'batch_size': 64}},
            'model': {'name': 'linear'},
            'algorithm': {'mode': 'train', 'num_steps': 1000, 'progress_unit': 'step', 'eval_period': 0},
        }
    )
    longer = estimate_job_seconds(
        {
            'data': {'name': 'MNIST', 'config': {'batch_size': 64}},
            'model': {'name': 'linear'},
            'algorithm': {'mode': 'train', 'num_steps': 2000, 'progress_unit': 'step', 'eval_period': 0},
        }
    )
    assert longer - short == 6
    eval_short = estimate_job_seconds(
        {
            'data': {'name': 'MNIST', 'config': {'batch_size': 64}},
            'model': {'name': 'linear'},
            'algorithm': {'mode': 'eval', 'num_epochs': 1},
        }
    )
    eval_long = estimate_job_seconds(
        {
            'data': {'name': 'MNIST', 'config': {'batch_size': 64}},
            'model': {'name': 'linear'},
            'algorithm': {'mode': 'eval', 'num_epochs': 20},
        }
    )
    assert eval_short == eval_long

def test_estimate_nested_algorithm_settings_matches_runtime_precedence():
    cfg = {'data': {'name': 'MNIST', 'config': {'batch_size': 64}}, 'model': {'name': 'linear'},
           'algorithm': {'mode': 'train', 'num_steps': 1, 'config': {'num_steps': 1000, 'eval_period': 0}}}
    equivalent = {**cfg, 'algorithm': {'mode': 'train', 'num_steps': 1000, 'eval_period': 0}}
    assert estimate_job_seconds(cfg) == estimate_job_seconds(equivalent)
