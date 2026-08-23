from rpipe.study.expand import expand_patches


def test_expand_patches_train_size_baseline():
    study = {
        'study': 'demo',
        'experiment': {'name': 'mnist_linear'},
        'fixed': {'seed': 0, 'algorithm': {'mode': 'train', 'lr': 0.1}},
        'axes': {'data.config.train_size': [500, 2000]},
        'tags': [{'when': {'data.config.train_size': 500}, 'tags': ['baseline']}],
        'run_description': 'size={train_size}',
    }
    patches = expand_patches(study)
    assert len(patches) == 2
    assert patches[0]['tags'] == ['baseline']
    assert patches[0]['data']['config']['train_size'] == 500
    assert 'tags' not in patches[1]
    assert patches[0]['description'] == 'size=500'
