"""Study mnist_seeds: grid then launch mnist_linear.

Usage (from repo root)::

    python examples/studies/mnist_seeds/run.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_EXP = _REPO / 'examples' / 'experiments' / 'mnist_linear'
_SRC = _REPO / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    grid = _load(_EXP / 'grid' / '__init__.py', 'mnist_linear_grid')
    launch = _load(_EXP / 'launch' / '__init__.py', 'mnist_linear_launch')
    for path in grid.expand([0, 1]):
        print(path)
    return launch.main([])


if __name__ == '__main__':
    raise SystemExit(main())
