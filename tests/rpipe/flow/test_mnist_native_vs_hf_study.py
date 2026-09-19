from pathlib import Path

import pytest

import json

from rpipe.flow.cli import main
from rpipe.structure.artifact import load_index, load_result

pytest.importorskip('transformers')

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.content,
    pytest.mark.p2,
    pytest.mark.flow_layer,
    pytest.mark.module_cli,
    pytest.mark.slow,
    pytest.mark.external,
]


def test_mnist_native_vs_hf_cpu_make_launches_both_sources_without_gpu(tmp_path: Path):
    import shutil

    repo = Path(__file__).resolve().parents[3]
    src = repo / 'studies' / 'mnist_native_vs_hf'
    study = tmp_path / 'mnist_native_vs_hf'
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
    text = text.replace('num_epochs: 20', 'num_epochs: 1')
    yaml_path.write_text(text, encoding='utf-8')

    assert main(['make', str(study)]) == 0
    launch_plan = json.loads((study / 'scripts' / 'jobs.json').read_text(encoding='utf-8'))
    jobs = launch_plan['jobs']
    assert len(jobs) == 2
    assert all(job['device'] == 'cpu' for job in jobs)
    assert all('gpu' not in job for job in jobs)
    assert launch_plan['capacity']['gpus'] == []

    assert main(['launch', str(study), '--console', 'shared']) == 0
    results = [study / 'runs' / job['run_id'] / 'result.json' for job in jobs]
    sources = sorted(
        load_result(path)['control']['algorithm']['source'] for path in results
    )
    assert sources == ['custom_torch', 'transformers_trainer']
    for path in results:
        result = load_result(path)
        assert result['status'] == 'succeeded'
        assert 'accuracy' in result['metrics']
    index = load_index(study)
    assert len(index['experiments']) == 2
    process = json.loads((study / 'process.json').read_text(encoding='utf-8'))
    assert process['complete'] is True
    assert len(process['experiments']) == 2
