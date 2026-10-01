import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.content,
    pytest.mark.p2,
    pytest.mark.structure_layer,
    pytest.mark.module_model,
    pytest.mark.cost(cost_class='c1'),
    pytest.mark.result_type('categorical', detail='summary'),
]

from pathlib import Path

from rpipe.structure.api import model_api
from rpipe.structure.model import ModelConfig, ModelRegistry


def test_registry_lists_main_custom_torch_models():
    names = {name for name, source in ModelRegistry.list() if source == 'custom_torch'}
    assert {'linear', 'mlp', 'cnn', 'resnet18', 'resnet10', 'resnet', 'wresnet28x2', 'wresnet28x8', 'wresnet'} <= names


def test_linear_uses_data_meta_shape(tmp_path: Path):
    import torch

    model = model_api.build(
        ModelConfig.from_mapping({'name': 'linear'}),
        tmp_path,
        data_meta={'data_size': [3, 32, 32], 'target_size': 10},
    )
    x = torch.randn(2, 3, 32, 32)
    out = model.module(x)
    assert tuple(out.shape) == (2, 10)
    assert model.module.output_proj.bias.eq(0).all()


def test_input_norm_sits_in_front_of_the_network(tmp_path: Path):
    import torch

    model = model_api.build(
        ModelConfig.from_mapping({'name': 'linear'}),
        tmp_path,
        data_meta={
            'data_size': [1, 2, 2],
            'target_size': 2,
            'mean': (0.5,),
            'std': (0.5,),
        },
    )
    x = torch.zeros(1, 1, 2, 2)
    out = model.module(x)
    bare = model.module.net(x)
    assert tuple(out.shape) == (1, 2)
    assert not torch.allclose(out, bare)


def test_kornia_crop_and_flip_run_only_while_training(tmp_path: Path):
    import torch

    model = model_api.build(
        ModelConfig.from_mapping({'name': 'linear'}),
        tmp_path,
        data_meta={
            'data_size': [3, 8, 8],
            'target_size': 10,
            'mean': (0.0, 0.0, 0.0),
            'std': (1.0, 1.0, 1.0),
            'augment': True,
            'train_aug': 'cifar',
        },
    )
    assert model.module.train_aug is not None
    x = torch.zeros(2, 3, 8, 8)
    x[:, :, :, 4:] = 1
    model.module.eval()
    first = model.module(x)
    second = model.module(x)
    assert torch.allclose(first, second)
    model.module.train()
    trained = model.module(x)
    assert tuple(trained.shape) == (2, 10)


def test_mlp_cnn_resnet_forward_cifar_shape(tmp_path: Path):
    import torch

    meta = {'data_size': [3, 32, 32], 'target_size': 10}
    x = torch.randn(2, 3, 32, 32)
    for name in ('mlp', 'cnn', 'resnet18', 'resnet10', 'wresnet28x2', 'wresnet28x8'):
        model = model_api.build(ModelConfig.from_mapping({'name': name}), tmp_path, data_meta=meta)
        out = model.module(x)
        assert tuple(out.shape) == (2, 10), name
