"""Launch mnist_linear Flow runs.

Usage (from repo root)::

    python examples/experiments/mnist_linear/launch/run.py
    python examples/experiments/mnist_linear/launch/run.py --run-dirs <id>
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SRC = _REPO_ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_spec = importlib.util.spec_from_file_location(
    'mnist_linear_launch',
    Path(__file__).resolve().parent / '__init__.py',
)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

discover_run_dirs = _mod.discover_run_dirs
run_one = _mod.run_one
run_many = _mod.run_many
main = _mod.main

if __name__ == '__main__':
    raise SystemExit(main())
