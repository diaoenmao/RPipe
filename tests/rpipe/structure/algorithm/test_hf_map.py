import pytest

from rpipe.structure.algorithm.config import AlgorithmConfig
from rpipe.structure.algorithm.hf_map import (
    hf_optim_name,
    hf_scheduler_name,
    training_arguments_kwargs,
)


def test_hf_maps_sgd_cosine_and_resume_key():
    cfg = AlgorithmConfig.from_mapping(
        {
            'mode': 'train',
            'source': 'transformers_trainer',
            'optimizer': 'SGD',
            'lr': 0.1,
            'scheduler': 'cosine',
            'num_steps': 10,
            'warmup_ratio': 0.1,
        }
    )
    kwargs = training_arguments_kwargs(
        cfg, output_dir='/tmp/hf', resume_from_checkpoint='/tmp/ckpt'
    )
    assert kwargs['optim'] == 'sgd'
    assert kwargs['learning_rate'] == 0.1
    assert kwargs['lr_scheduler_type'] == 'cosine'
    assert kwargs['max_steps'] == 10
    assert kwargs['report_to'] == 'none'
    assert kwargs['warmup_ratio'] == 0.1
    assert kwargs['resume_from_checkpoint'] == '/tmp/ckpt'
    assert kwargs['max_grad_norm'] == 0.0
    stepped = training_arguments_kwargs(
        AlgorithmConfig.from_mapping({'mode': 'train', 'num_steps': 8, 'step_period': 2}),
        output_dir='/tmp/hf',
    )
    assert stepped.get('gradient_accumulation_steps') == 2
    assert hf_optim_name('AdamW') == 'adamw_torch'
    assert hf_scheduler_name(None) == 'constant'
    clipped = training_arguments_kwargs(
        AlgorithmConfig.from_mapping({'mode': 'train', 'num_steps': 1, 'max_grad_norm': 1.0}),
        output_dir='/tmp/hf',
    )
    assert clipped['max_grad_norm'] == pytest.approx(1.0)
