"""Study: expand seeds via mnist_linear grid, then launch Flow."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Study mnist_seeds')
    parser.add_argument('--seeds', default='0,1')
    parser.add_argument('--skip-grid', action='store_true')
    args = parser.parse_args(argv)

    study_dir = Path(__file__).resolve().parent
    repo = Path(__file__).resolve().parents[2]
    exp = repo / 'experiments' / 'mnist_linear'
    src = repo.parent / 'src'
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from rpipe.artifact import build_index, load_config, write_index

    grid = _load(exp / 'grid' / '__init__.py', 'mnist_linear_grid')
    launch = _load(exp / 'launch' / '__init__.py', 'mnist_linear_launch')

    configs: list[Path] = []
    if not args.skip_grid:
        seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
        configs = grid.expand(seeds, exp_dir=exp, tags_by_seed={0: ['baseline']})
        for path in configs:
            print(path)
    else:
        # still need configs for index: discover existing
        art = exp / 'artifact'
        if art.is_dir():
            configs = sorted(p for p in art.glob('*/config.yaml'))

    base = load_config(exp / 'experiment_config.yaml')
    runs = []
    for path in configs:
        cfg = load_config(path)
        runs.append(
            {
                'id': cfg.get('id'),
                'description': cfg.get('description'),
                'tags': cfg.get('tags') or [],
                'run_dir': path.parent.name,
                'config': str(path),
            }
        )
    index = build_index(
        study='mnist_seeds',
        description='MNIST linear seed sweep',
        experiments=[
            {
                'name': base.get('experiment') or 'mnist_linear',
                'description': base.get('description') or '',
                'path': str(exp),
                'runs': runs,
            }
        ],
    )
    print(write_index(study_dir, index))
    return launch.main([])


if __name__ == '__main__':
    raise SystemExit(main())
