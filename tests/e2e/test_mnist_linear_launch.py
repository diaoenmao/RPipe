from pathlib import Path

import runpy

from rpipe.artifact import build_index, load_config, load_result, load_index, write_index


def test_mnist_linear_grid_and_launch_smoke():
    repo = Path(__file__).resolve().parents[2]
    study = repo / 'examples' / 'studies' / 'mnist_seeds'
    exp = repo / 'examples' / 'experiments' / 'mnist_linear'
    grid = runpy.run_path(str(exp / 'grid' / '__init__.py'))
    launch = runpy.run_path(str(exp / 'launch' / '__init__.py'))

    written = grid['expand']([0], exp_dir=exp, tags_by_seed={0: ['baseline']})
    assert len(written) == 1
    assert written[0].is_file()
    cfg = load_config(written[0])
    assert cfg.get('description')
    assert cfg.get('tags') == ['baseline']

    base = load_config(exp / 'experiment_config.yaml')
    index = build_index(
        study='mnist_seeds',
        description='e2e smoke',
        experiments=[
            {
                'name': base.get('experiment') or 'mnist_linear',
                'description': base.get('description') or '',
                'path': str(exp),
                'runs': [
                    {
                        'id': cfg['id'],
                        'description': cfg.get('description'),
                        'tags': cfg.get('tags') or [],
                        'run_dir': written[0].parent.name,
                        'config': str(written[0]),
                    }
                ],
            }
        ],
    )
    index_path = write_index(study, index)
    assert index_path.is_file()
    assert load_index(study)['id'] == index['id']

    run_dir = written[0].parent.name
    paths = launch['run_many'](exp_dir=exp, run_dirs=[run_dir])
    assert len(paths) == 1
    assert paths[0].is_file()
    result = load_result(paths[0])
    assert result['status'] == 'succeeded'
