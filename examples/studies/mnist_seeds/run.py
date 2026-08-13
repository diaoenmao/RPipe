"""Study mnist_seeds: grid then launch mnist_linear.

Usage (from repo root)::

    python examples/studies/mnist_seeds/run.py
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_EXP = _REPO / 'examples' / 'experiments' / 'mnist_linear'


def main() -> int:
    grid = _EXP / 'grid' / 'run.py'
    launch = _EXP / 'launch' / 'run.py'
    runpy.run_path(str(grid), run_name='__main__')
    # re-exec launch main without SystemExit from grid
    sys.argv = [str(launch)]
    runpy.run_path(str(launch), run_name='__not_main__')
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location('mnist_linear_launch', launch)
    mod = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.main([])


if __name__ == '__main__':
    raise SystemExit(main())
