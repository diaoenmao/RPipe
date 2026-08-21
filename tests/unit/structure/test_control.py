from rpipe.structure.control import (
    ExperimentConfig,
    control_from_config,
    control_to_config,
    run_config_from_merge,
    validate_run_config,
)


def test_control_from_config_hashes_id_and_layers():
    cfg = {
        'seed': 0,
        'experiment': 'mnist_linear',
        'data': {'name': 'MNIST'},
        'model': {'name': 'linear'},
        'algorithm': {'mode': 'train'},
        'system': {'device': 'cpu'},
    }
    control = control_from_config(cfg)
    assert control.id
    assert len(control.id) == 16
    assert control.seed == 0
    assert control.data['name'] == 'MNIST'
    assert control.to_dict()['model']['name'] == 'linear'
    assert control.to_dict()['id'] == control.id


def test_same_content_same_id_different_content_different_id():
    a = control_from_config({'seed': 0, 'data': {'name': 'MNIST'}})
    b = control_from_config({'seed': 0, 'data': {'name': 'MNIST'}})
    c = control_from_config({'seed': 1, 'data': {'name': 'MNIST'}})
    assert a.id == b.id
    assert a.id != c.id


def test_run_config_from_merge_patches_experiment_base():
    base = ExperimentConfig.from_mapping(
        {
            'experiment': 'mnist_linear',
            'data': {'name': 'MNIST', 'source': 'torch'},
            'algorithm': {'mode': 'train'},
        }
    )
    run = run_config_from_merge(base, {'seed': 0, 'data': {'path': '/data/mnist'}})
    assert run.seed == 0
    assert run.experiment == 'mnist_linear'
    assert run.data.name == 'MNIST'
    assert run.data.source == 'torch'
    assert run.data.path == '/data/mnist'
    assert run.id
    validate_run_config(run)


def test_control_to_config_roundtrip():
    control = control_from_config(
        {
            'seed': 2,
            'data': {'name': 'CIFAR'},
            'algorithm': {'mode': 'eval'},
        }
    )
    mapping = control_to_config(control)
    again = control_from_config(mapping)
    assert again.id == control.id
    assert again.seed == 2
    assert again.algorithm['mode'] == 'eval'


def test_legacy_slug_ignored_for_id():
    a = control_from_config({'slug': 'seed_0', 'seed': 0, 'data': {'name': 'X'}})
    b = control_from_config({'seed': 0, 'data': {'name': 'X'}})
    assert a.id == b.id


def test_description_and_tags_ignored_for_run_id():
    a = control_from_config(
        {'seed': 0, 'data': {'name': 'X'}, 'description': 'one', 'tags': ['baseline']}
    )
    b = control_from_config(
        {'seed': 0, 'data': {'name': 'X'}, 'description': 'two', 'tags': ['smoke']}
    )
    assert a.id == b.id
    assert a.to_dict()['description'] == 'one'
    assert a.to_dict()['tags'] == ['baseline']
    assert b.to_dict()['tags'] == ['smoke']
