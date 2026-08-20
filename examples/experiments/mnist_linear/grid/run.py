"""Write Config files into this Experiment's Artifact tree.

Usage::

    python examples/experiments/mnist_linear/grid/run.py
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
    'mnist_linear_grid',
    Path(__file__).resolve().parent / '__init__.py',
)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

expand = _mod.expand
main = _mod.main

if __name__ == '__main__':
    raise SystemExit(main())
