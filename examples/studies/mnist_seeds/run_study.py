"""Study: expand seeds via mnist_linear grid, then launch Flow."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def _run_module(path: Path, argv: list[str]) -> None:
    old = sys.argv
    try:
        sys.argv = [str(path)] + argv
        runpy.run_path(str(path), run_name='__main__')
    finally:
        sys.argv = old


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Study mnist_seeds')
    parser.add_argument('--seeds', default='0,1')
    parser.add_argument('--skip-grid', action='store_true')
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[2]
    exp = repo / 'experiments' / 'mnist_linear'
    grid = exp / 'grid' / '__init__.py'
    launch = exp / 'launch' / '__init__.py'

    if not args.skip_grid:
        _run_module(grid, ['--seeds', args.seeds])
    _run_module(launch, [])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
