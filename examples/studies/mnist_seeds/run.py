"""Study mnist_seeds: grid then launch mnist_linear.

Usage (from repo root)::

    python examples/studies/mnist_seeds/run.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_STUDY = Path(__file__).resolve().parent
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


def _write_index(config_paths: list[Path]) -> Path:
    from rpipe.artifact import build_index, load_config, write_index

    base = load_config(_EXP / 'experiment_config.yaml')
    runs = []
    for path in config_paths:
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
        description='MNIST linear seed sweep (0, 1)',
        experiments=[
            {
                'name': base.get('experiment') or 'mnist_linear',
                'description': base.get('description') or '',
                'path': str(_EXP),
                'runs': runs,
            }
        ],
    )
    return write_index(_STUDY, index)


def main() -> int:
    grid = _load(_EXP / 'grid' / '__init__.py', 'mnist_linear_grid')
    launch = _load(_EXP / 'launch' / '__init__.py', 'mnist_linear_launch')
    configs = grid.expand([0, 1], tags_by_seed={0: ['baseline']})
    for path in configs:
        print(path)
    print(_write_index(configs))
    return launch.main([])


if __name__ == '__main__':
    raise SystemExit(main())
