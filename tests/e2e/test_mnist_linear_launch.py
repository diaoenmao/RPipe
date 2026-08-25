from pathlib import Path

from rpipe.cli import run_study
from rpipe.structure.artifact import load_index, load_result


def test_mnist_seeds_study_runner_smoke(tmp_path: Path):
    """Copy minimal study into tmp and run one seed via study.yaml path.

    Uses the real studies/mnist_seeds recipe files but a disposable study_dir
    so we do not pollute the repo tree during CI.
    """
    import shutil

    repo = Path(__file__).resolve().parents[2]
    src = repo / 'studies' / 'mnist_seeds'
    study = tmp_path / 'mnist_seeds'
    shutil.copytree(
        src,
        study,
        ignore=shutil.ignore_patterns('runs', 'shared', 'index.json', '__pycache__'),
    )
    yaml_path = study / 'study.yaml'
    text = yaml_path.read_text(encoding='utf-8')
    text = text.replace('seeds: [0, 1]', 'seeds: [0]')
    yaml_path.write_text(text, encoding='utf-8')

    out = run_study(study)
    assert out['index'].is_file()
    assert len(out['configs']) == 1
    assert len(out['results']) == 1
    assert out['results'][0].is_file()
    result = load_result(out['results'][0])
    assert result['status'] == 'succeeded'
    index = load_index(study)
    assert len(index['experiments']) == 1
    assert index['experiments'][0]['factors'] == {}
    assert index['experiments'][0]['runs'][0]['seed'] == 0
    assert (study / 'shared' / 'data').is_dir()
    assert (study / 'docs').is_dir()
