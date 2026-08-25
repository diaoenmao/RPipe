"""Write Run Configs into this Study's ``runs/<id>/`` tree."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[3] / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rpipe.structure.artifact import artifact_layout, load_config, write_config
from rpipe.structure.control import experiment_config_from_mapping, run_config_from_merge


def study_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def experiment_config_path(root: Path | None = None) -> Path:
    return (root or study_dir()) / 'experiment_config.yaml'


def load_base(root: Path | None = None) -> Any:
    return experiment_config_from_mapping(load_config(experiment_config_path(root)))


def expand(
    seeds: list[int] | None = None,
    exp_dir: Path | None = None,
    study_dir_path: Path | None = None,
    tags_by_seed: dict[int, list[str]] | None = None,
    patches: list[dict[str, Any]] | None = None,
) -> list[Path]:
    """Merge experiment_config ⊕ patch → RunConfig → write under ``runs/<id>/``."""
    root = study_dir_path or exp_dir or study_dir()
    base = load_base(root)
    tag_map = tags_by_seed or {}
    if patches is None:
        seed_list = seeds if seeds is not None else [0]
        patches = []
        for seed in seed_list:
            patch: dict[str, Any] = {
                'seed': seed,
                'description': f'{base.experiment or root.name} seed={seed}',
            }
            tags = tag_map.get(seed)
            if tags:
                patch['tags'] = list(tags)
            patches.append(patch)

    written: list[Path] = []
    for patch in patches:
        run = run_config_from_merge(base, patch)
        cfg = run.to_mapping()
        layout = artifact_layout(root, run.id)
        written.append(write_config(layout.config_path, cfg))
    return written


def cartesian(axes: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(axes.keys())
    values = [axes[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Grid Configs for this Study')
    parser.add_argument('--seeds', default='0,1', help='comma-separated seeds')
    args = parser.parse_args(argv)
    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    for path in expand(seeds=seeds):
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
