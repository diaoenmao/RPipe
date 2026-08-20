"""Launch one or more Artifact runs for mnist_linear."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[4] / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rpipe.artifact import artifact_layout, load_config
from rpipe.flow import FlowContext, FlowRunner


def experiment_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def discover_run_dirs(exp_dir: Path) -> list[str]:
    root = exp_dir / 'artifact'
    if not root.is_dir():
        return []
    return [
        child.name
        for child in sorted(root.iterdir())
        if child.is_dir() and (child / 'config.yaml').is_file()
    ]


def run_one(exp_dir: Path, run_dir: str, phases: list[str] | None = None) -> Path:
    layout = artifact_layout(exp_dir, run_dir)
    if not layout.config_path.is_file():
        raise FileNotFoundError(f'missing Config: {layout.config_path}')
    cfg = load_config(layout.config_path)
    ctx = FlowContext(experiment_dir=exp_dir, layout=layout, config=cfg)
    return FlowRunner(phases=phases).run(ctx)


def run_many(
    exp_dir: Path | None = None,
    run_dirs: list[str] | None = None,
    phases=None,
) -> list[Path]:
    exp_dir = exp_dir or experiment_dir()
    selected = run_dirs or discover_run_dirs(exp_dir)
    if not selected:
        raise FileNotFoundError(f'no artifact/*/config.yaml under {exp_dir}')
    return [run_one(exp_dir, run_dir, phases=phases) for run_dir in selected]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Launch mnist_linear Flow runs')
    parser.add_argument(
        '--run-dirs',
        default='',
        help='comma-separated Artifact directory names (default: discover all)',
    )
    parser.add_argument('--phases', default='', help='comma-separated phases (default: all)')
    args = parser.parse_args(argv)
    run_dirs = [s.strip() for s in args.run_dirs.split(',') if s.strip()] or None
    phases = [s.strip() for s in args.phases.split(',') if s.strip()] or None
    for path in run_many(run_dirs=run_dirs, phases=phases):
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
