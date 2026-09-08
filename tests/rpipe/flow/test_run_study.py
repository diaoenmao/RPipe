from pathlib import Path

from rpipe.flow.cli import run_study
from rpipe.structure.artifact import load_index


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
    index = load_index(study)
    assert out['index'].is_file()
    assert len(out['configs']) == 18
    assert len(index['experiments']) == 6
    sizes = [exp['factors']['data.config.train_size'] for exp in index['experiments']]
    modes = [exp['factors']['algorithm.mode'] for exp in index['experiments']]
    assert sizes == [500, 500, 2000, 2000, 8000, 8000]
    assert modes == ['train', 'eval', 'train', 'eval', 'train', 'eval']
    for exp in index['experiments']:
        assert [r['seed'] for r in exp['runs']] == [0, 1, 2]
    assert (study / 'docs').is_dir()
    assert (study / 'shared' / 'data').is_dir()
    assert (study / 'runs').is_dir()
