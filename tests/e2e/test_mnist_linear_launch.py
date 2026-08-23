from pathlib import Path

from rpipe.study import run_study


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
    # only seed 0 for smoke speed
    yaml_path = study / 'study.yaml'
    text = yaml_path.read_text(encoding='utf-8')
    text = text.replace('seed: [0, 1]', 'seed: [0]')
    yaml_path.write_text(text, encoding='utf-8')

    out = run_study(study)
    assert out['index'].is_file()
    assert len(out['configs']) == 1
    assert len(out['results']) == 1
    assert out['results'][0].is_file()
    assert (study / 'shared' / 'data').is_dir()
    assert (study / 'docs').is_dir()
