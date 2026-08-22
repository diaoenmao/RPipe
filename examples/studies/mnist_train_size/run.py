"""Study mnist_train_size: vary MNIST train_size → test accuracy.

Plan: PLAN.md
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

TRAIN_SIZES = [500, 2000, 8000]
SEED = 0


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _patches() -> list[dict]:
    patches = []
    for size in TRAIN_SIZES:
        patch = {
            'seed': SEED,
            'description': f'mnist_linear train_size={size}',
            'data': {'config': {'train_size': size}},
            'algorithm': {'num_epochs': 2, 'lr': 0.1},
        }
        if size == TRAIN_SIZES[0]:
            patch['tags'] = ['baseline']
        patches.append(patch)
    return patches


def main() -> int:
    from rpipe.artifact import build_index, load_config, write_index

    grid = _load(_EXP / 'grid' / '__init__.py', 'mnist_linear_grid')
    launch = _load(_EXP / 'launch' / '__init__.py', 'mnist_linear_launch')

    configs = grid.expand(patches=_patches(), exp_dir=_EXP)
    for path in configs:
        print(path)

    base = load_config(_EXP / 'experiment_config.yaml')
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
        study='mnist_train_size',
        description='MNIST train_size sweep → test accuracy',
        experiments=[
            {
                'name': base.get('experiment') or 'mnist_linear',
                'description': base.get('description') or '',
                'path': str(_EXP),
                'runs': runs,
            }
        ],
    )
    print(write_index(_STUDY, index))

    run_dirs = [p.parent.name for p in configs]
    for path in launch.run_many(exp_dir=_EXP, run_dirs=run_dirs):
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
