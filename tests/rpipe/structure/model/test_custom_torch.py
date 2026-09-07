from pathlib import Path

from rpipe.structure.api import model_api
from rpipe.structure.model import ModelConfig, ModelRegistry


def test_registry_lists_main_custom_torch_models():
    names = {name for name, source in ModelRegistry.list() if source == 'custom_torch'}
    assert {'linear', 'mlp', 'cnn', 'resnet18', 'resnet10', 'resnet'} <= names


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


def test_mlp_cnn_resnet_forward_cifar_shape(tmp_path: Path):
    import torch

    meta = {'data_size': [3, 32, 32], 'target_size': 10}
    x = torch.randn(2, 3, 32, 32)
    for name in ('mlp', 'cnn', 'resnet18', 'resnet10'):
        model = model_api.build(ModelConfig.from_mapping({'name': name}), tmp_path, data_meta=meta)
        out = model.module(x)
        assert tuple(out.shape) == (2, 10), name
