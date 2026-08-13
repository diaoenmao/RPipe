"""Write Config files into this Experiment's Artifact tree.

Usage::

    python examples/experiments/mnist_linear/grid/run.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SRC = _REPO_ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rpipe.artifact import artifact_layout, write_config  # noqa: E402


def experiment_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def expand(seeds: list[int] | None = None) -> list[Path]:
    seeds = seeds or [0, 1]
    exp_dir = experiment_dir()
    written: list[Path] = []
    for seed in seeds:
        slug = f'seed_{seed}'
        layout = artifact_layout(exp_dir, slug)
        cfg = {
            'slug': slug,
            'seed': seed,
            'data': {'name': 'MNIST'},
            'model': {'name': 'linear'},
            'algorithm': {
                'semantics': ['train', 'eval'],
                'num_steps': 4,
            },
            'system': {'device': 'cpu'},
        }
        written.append(write_config(layout.config_path, cfg))
    return written


def cartesian(axes: dict[str, list]) -> list[dict]:
    keys = list(axes.keys())
    values = [axes[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def main() -> int:
    for path in expand():
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
