from pathlib import Path

from rpipe.cli import run_study
from rpipe.structure.artifact import load_index, load_result


def test_mnist_train_size_study_runner_smoke(tmp_path: Path):
    """Copy the real study into tmp; run one Experiment so CI stays fast."""
    import shutil

    repo = Path(__file__).resolve().parents[2]
    src = repo / 'studies' / 'mnist_train_size'
    study = tmp_path / 'mnist_train_size'
    shutil.copytree(
        src,
        study,
        ignore=shutil.ignore_patterns('runs', 'shared', 'index.json', '__pycache__'),
    )
    yaml_path = study / 'study.yaml'
    text = yaml_path.read_text(encoding='utf-8')
    text = text.replace(
        'data.config.train_size: [500, 2000, 8000]',
        'data.config.train_size: [500]',
    )
    text = text.replace('seeds: [0, 1, 2]', 'seeds: [0]')
    text = text.replace('num_epochs: 20', 'num_epochs: 2')
    yaml_path.write_text(text, encoding='utf-8')

    out = run_study(study)
    assert out['index'].is_file()
    assert len(out['configs']) == 1
    assert len(out['results']) == 1
    assert out['results'][0].is_file()
    result = load_result(out['results'][0])
    assert result['status'] == 'succeeded'
    assert 'accuracy' in result['metrics']
    assert 'train_loss' in result['metrics']
    log_path = out['results'][0].parent / 'assets' / 'logs' / 'run.log'
    assert log_path.is_file()
    log_text = log_path.read_text(encoding='utf-8')
    assert 'Loss' in log_text
    tracker_state = out['results'][0].parent / 'assets' / 'tracker' / 'tracker_state.json'
    assert tracker_state.is_file()
    index = load_index(study)
    assert len(index['experiments']) == 1
    assert index['experiments'][0]['factors'] == {'data.config.train_size': 500}
    assert index['experiments'][0]['runs'][0]['seed'] == 0
    assert index['experiments'][0]['runs'][0]['tags'] == ['baseline']
    assert (study / 'shared' / 'data').is_dir()
    assert (study / 'docs').is_dir()
