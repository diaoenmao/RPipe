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

    repo = Path(__file__).resolve().parents[2]
    exp = repo / 'experiments' / 'mnist_linear'
    src = repo.parent / 'src'
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    grid = _load(exp / 'grid' / '__init__.py', 'mnist_linear_grid')
    launch = _load(exp / 'launch' / '__init__.py', 'mnist_linear_launch')

    if not args.skip_grid:
        seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
        for path in grid.expand(seeds, exp_dir=exp):
            print(path)
    return launch.main([])


if __name__ == '__main__':
    raise SystemExit(main())
