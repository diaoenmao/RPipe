"""Launch mnist_linear Flow runs.

Usage (from repo root)::

    python examples/experiments/mnist_linear/launch/run.py
    python examples/experiments/mnist_linear/launch/run.py --slugs seed_0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SRC = _REPO_ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rpipe.artifact import artifact_layout, load_config  # noqa: E402
from rpipe.flow import FlowContext, FlowRunner  # noqa: E402


def experiment_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def discover_slugs(exp_dir: Path) -> list[str]:
    root = exp_dir / 'artifact'
    if not root.is_dir():
        return []
    return [
        child.name
        for child in sorted(root.iterdir())
        if child.is_dir() and (child / 'config.yaml').is_file()
    ]


def run_one(exp_dir: Path, slug: str, phases: list[str] | None = None) -> Path:
    layout = artifact_layout(exp_dir, slug)
    if not layout.config_path.is_file():
        raise FileNotFoundError(f'missing Config: {layout.config_path}')
    cfg = load_config(layout.config_path)
    ctx = FlowContext(experiment_dir=exp_dir, layout=layout, config=cfg)
    return FlowRunner(phases=phases).run(ctx)


def run_many(exp_dir: Path | None = None, slugs: list[str] | None = None, phases=None) -> list[Path]:
    exp_dir = exp_dir or experiment_dir()
    selected = slugs or discover_slugs(exp_dir)
    if not selected:
        raise FileNotFoundError(f'no artifact/*/config.yaml under {exp_dir}')
    return [run_one(exp_dir, slug, phases=phases) for slug in selected]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Launch mnist_linear Flow runs')
    parser.add_argument('--slugs', default='', help='comma-separated control slugs')
    parser.add_argument('--phases', default='', help='comma-separated phases (default: all)')
    args = parser.parse_args(argv)
    slugs = [s.strip() for s in args.slugs.split(',') if s.strip()] or None
    phases = [s.strip() for s in args.phases.split(',') if s.strip()] or None
    for path in run_many(slugs=slugs, phases=phases):
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
