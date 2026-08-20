from pathlib import Path

import runpy


def test_mnist_linear_grid_and_launch_smoke():
    repo = Path(__file__).resolve().parents[2]
    exp = repo / 'examples' / 'experiments' / 'mnist_linear'
    grid = runpy.run_path(str(exp / 'grid' / '__init__.py'))
    launch = runpy.run_path(str(exp / 'launch' / '__init__.py'))

    written = grid['expand']([0], exp_dir=exp)
    assert len(written) == 1
    assert written[0].is_file()

    run_dir = written[0].parent.name
    paths = launch['run_many'](exp_dir=exp, run_dirs=[run_dir])
    assert len(paths) == 1
    assert paths[0].is_file()
