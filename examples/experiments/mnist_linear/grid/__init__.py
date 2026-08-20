"""Write Run Configs into this Experiment's Artifact tree."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[4] / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rpipe.artifact import artifact_layout, load_config, write_config
from rpipe.structure.control import experiment_config_from_mapping, run_config_from_merge


def experiment_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def experiment_config_path(exp_dir: Path | None = None) -> Path:
    return (exp_dir or experiment_dir()) / 'experiment_config.yaml'


def load_base(exp_dir: Path | None = None) -> Any:
    path = experiment_config_path(exp_dir)
    return experiment_config_from_mapping(load_config(path))


def expand(seeds: list[int], exp_dir: Path | None = None) -> list[Path]:
    """Merge experiment_config ⊕ {seed} → RunConfig (hash id) → write Artifact Config."""
    exp_dir = exp_dir or experiment_dir()
    base = load_base(exp_dir)
    written: list[Path] = []
    for seed in seeds:
        run = run_config_from_merge(base, {'seed': seed})
        cfg = run.to_mapping()
        layout = artifact_layout(exp_dir, run.id)
        written.append(write_config(layout.config_path, cfg))
    return written


def cartesian(axes: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(axes.keys())
    values = [axes[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Grid Configs for mnist_linear')
    parser.add_argument('--seeds', default='0,1', help='comma-separated seeds')
    args = parser.parse_args(argv)
    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    for path in expand(seeds):
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
