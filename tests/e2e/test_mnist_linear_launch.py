from pathlib import Path

import runpy


def test_mnist_linear_launch_smoke():
    repo = Path(__file__).resolve().parents[2]
    launch = repo / 'examples' / 'experiments' / 'mnist_linear' / 'launch' / '__init__.py'
    runpy.run_path(str(launch), run_name='__not_main__')
    # import functions via run_path namespace
    ns = runpy.run_path(str(launch))
    exp = repo / 'examples' / 'experiments' / 'mnist_linear'
    paths = ns['run_many'](exp_dir=exp, slugs=['seed_0'])
    assert len(paths) == 1
    assert paths[0].is_file()
