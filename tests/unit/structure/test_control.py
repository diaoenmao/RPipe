from rpipe.structure.control import Control, control_from_config


def test_control_from_config_uses_slug_and_layers():
    cfg = {
        'slug': 'seed_0',
        'seed': 0,
        'data': {'name': 'MNIST'},
        'model': {'name': 'linear'},
        'algorithm': {'semantics': ['train']},
        'system': {'device': 'cpu'},
    }
    control = control_from_config(cfg)
    assert isinstance(control, Control)
    assert control.slug == 'seed_0'
    assert control.seed == 0
    assert control.data['name'] == 'MNIST'
    assert control.to_dict()['model']['name'] == 'linear'
