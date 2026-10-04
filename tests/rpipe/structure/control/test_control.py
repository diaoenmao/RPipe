import pytest

from rpipe.structure.control import (
    ExperimentConfig,
    control_from_config,
    control_to_config,
    run_config_from_merge,
    validate_run_config,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.content,
    pytest.mark.p1,
    pytest.mark.structure_layer,
    pytest.mark.module_control,
    pytest.mark.cost(cost_class='c1'),
    pytest.mark.result_type('categorical', detail='summary'),
]


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


def test_description_ignored_tags_affect_run_id():
    a = control_from_config(
        {'seed': 0, 'data': {'name': 'X'}, 'description': 'one', 'tags': ['baseline']}
    )
    b = control_from_config(
        {'seed': 0, 'data': {'name': 'X'}, 'description': 'two', 'tags': ['baseline']}
    )
    c = control_from_config(
        {'seed': 0, 'data': {'name': 'X'}, 'description': 'one', 'tags': ['smoke']}
    )
    assert a.id == b.id  # description excluded from hash
    assert a.id != c.id  # tags included in hash
    assert a.to_dict()['tags'] == ['baseline']
    assert c.to_dict()['tags'] == ['smoke']


def test_omitted_version_preserves_existing_run_id():
    control = control_from_config({'seed': 0, 'data': {'name': 'MNIST'}})
    # Existing content identity, before adding any version declaration.
    assert control.id == '96a1993415ec5eae'
    assert 'version' not in control_to_config(control)


@pytest.mark.parametrize('version', [7, 'repeat-1', '2026-10-02T10:00:00+08:00'])
def test_version_same_value_reuses_id_new_value_changes_id(version):
    config = {'seed': 0, 'data': {'name': 'MNIST'}, 'version': version}
    first = control_from_config(config)
    same = control_from_config(dict(config))
    different = control_from_config({**config, 'version': 'another-repeat'})
    assert first.id == same.id
    assert first.id != different.id
    assert control_to_config(first)['version'] == version


@pytest.mark.parametrize(
    ('patch', 'expected_version'),
    [({}, 'base'), ({'version': 'repeat'}, 'repeat')],
    ids=['inherit-base', 'override-base'],
)
def test_version_merge_and_roundtrip_preserve_value_and_id(patch, expected_version):
    base = ExperimentConfig.from_mapping({'version': 'base', 'data': {'name': 'MNIST'}})
    run = run_config_from_merge(base, {'seed': 0, **patch})
    mapping = run.to_mapping()
    # Merge recomputes the ID instead of trusting a serialized id field.
    again = run_config_from_merge(mapping)
    assert base.to_mapping()['version'] == 'base'
    assert mapping['version'] == expected_version
    assert again.to_mapping() == mapping
