from pathlib import Path

import pytest

from rpipe.flow.cli import run_study
from rpipe.structure.artifact import load_index, load_result

pytest.importorskip('transformers')


def test_mnist_native_vs_hf_smoke(tmp_path: Path):
    import shutil

    repo = Path(__file__).resolve().parents[2]
    src = repo / 'studies' / 'mnist_native_vs_hf'
    study = tmp_path / 'mnist_native_vs_hf'
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
    text = text.replace('num_epochs: 20', 'num_epochs: 1')
    yaml_path.write_text(text, encoding='utf-8')

    out = run_study(study)
    assert len(out['configs']) == 2
    assert len(out['results']) == 2
    sources = sorted(
        load_result(path)['control']['algorithm']['source'] for path in out['results']
    )
    assert sources == ['custom_torch', 'transformers_trainer']
    for path in out['results']:
        result = load_result(path)
        assert result['status'] == 'succeeded'
        assert 'accuracy' in result['metrics']
    index = load_index(study)
    assert len(index['experiments']) == 2
