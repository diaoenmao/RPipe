from pathlib import Path

import pytest

import json

from rpipe.flow.cli import main
from rpipe.structure.artifact import load_index, load_result

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.content,
    pytest.mark.p2,
    pytest.mark.flow_layer,
    pytest.mark.module_cli,
    pytest.mark.slow,
    pytest.mark.external,
]


def test_mnist_train_size_make_launch_processes_train_before_eval(tmp_path: Path):
    """A reduced real Study completes the public make/launch/process lifecycle."""
    import shutil

    repo = Path(__file__).resolve().parents[3]
    src = repo / 'studies' / 'mnist_train_size'
    study = tmp_path / 'mnist_train_size'
    shutil.copytree(
        src,
        study,
        ignore=shutil.ignore_patterns('runs', 'shared', 'index.json', '__pycache__'),
    )
    if (src / 'shared').is_dir():
        shutil.copytree(src / 'shared', study / 'shared')
    yaml_path = study / 'study.yaml'
    text = yaml_path.read_text(encoding='utf-8')
    text = text.replace(
        'data.config.train_size: [500, 2000, 8000]',
        'data.config.train_size: [500]',
    )
    text = text.replace('seeds: [0, 1, 2]', 'seeds: [0]')
    text = text.replace('num_epochs: 20', 'num_epochs: 2')
    yaml_path.write_text(text, encoding='utf-8')

    assert main(['make', str(study), '--num-gpus', '1']) == 0
    launch_plan = json.loads((study / 'scripts' / 'jobs.json').read_text(encoding='utf-8'))
    jobs = launch_plan['jobs']
    assert [job['mode'] for job in jobs] == ['train', 'eval']
    assert [job['device'] for job in jobs] == ['cuda', 'cuda']
    assert [job['gpu'] for job in jobs] == ['0', '0']

    assert main(['launch', str(study), '--num-gpus', '1', '--console', 'shared']) == 0
    results = [study / 'runs' / job['run_id'] / 'result.json' for job in jobs]
    train_result = load_result(results[0])
    eval_result = load_result(results[1])
    assert train_result['status'] == 'succeeded'
    assert eval_result['status'] == 'succeeded'
    assert train_result['control']['algorithm']['mode'] == 'train'
    assert eval_result['control']['algorithm']['mode'] == 'eval'
    assert 'accuracy' in train_result['metrics']
    assert 'train_loss' in train_result['metrics']
    assert 'eval_accuracy' in eval_result['metrics']
    log_path = results[0].parent / 'assets' / 'logs' / 'run.log'
    assert log_path.is_file()
    log_text = log_path.read_text(encoding='utf-8')
    assert 'Loss' in log_text
    assert 'elapsed=' in log_text
    tracker_state = results[0].parent / 'assets' / 'tracker' / 'tracker_state.json'
    assert tracker_state.is_file()
    ckpt = results[0].parent / 'assets' / 'checkpoints'
    assert (ckpt / 'latest.pt').is_file()
    assert (ckpt / 'best.pt').is_file()
    assert 'best_accuracy' in train_result['metrics']
    index = load_index(study)
    assert len(index['experiments']) == 2
    assert index['experiments'][0]['factors'] == {
        'data.config.train_size': 500,
        'algorithm.mode': 'train',
    }
    assert index['experiments'][1]['factors'] == {
        'data.config.train_size': 500,
        'algorithm.mode': 'eval',
    }
    assert index['experiments'][0]['runs'][0]['seed'] == 0
    assert index['experiments'][0]['runs'][0]['tags'] == ['baseline']
    assert index['experiments'][1]['runs'][0]['tags'] == []
    process = json.loads((study / 'process.json').read_text(encoding='utf-8'))
    assert process['complete'] is True
    assert len(process['experiments']) == 2
    assert process['figures']['learning_curves'] == 'docs/figures/learning_curves.png'
    assert (study / 'shared' / 'data').is_dir()
    assert (study / 'docs').is_dir()
