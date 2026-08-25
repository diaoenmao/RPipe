from pathlib import Path

from rpipe.cli import expand_patches, run_study


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


def test_run_study_index_groups_experiments(tmp_path: Path):
    import shutil

    repo = Path(__file__).resolve().parents[3]
    src = repo / 'studies' / 'mnist_train_size'
    study = tmp_path / 'mnist_train_size'
    shutil.copytree(
        src,
        study,
        ignore=shutil.ignore_patterns('runs', 'shared', 'index.json', '__pycache__'),
    )
    out = run_study(study, skip_launch=True)
    from rpipe.structure.artifact import load_index

    index = load_index(study)
    assert out['index'].is_file()
    assert len(out['configs']) == 9
    assert len(index['experiments']) == 3
    sizes = [exp['factors']['data.config.train_size'] for exp in index['experiments']]
    assert sizes == [500, 2000, 8000]
    for exp in index['experiments']:
        assert [r['seed'] for r in exp['runs']] == [0, 1, 2]
    assert (study / 'docs').is_dir()
    assert (study / 'shared' / 'data').is_dir()
    assert (study / 'runs').is_dir()
