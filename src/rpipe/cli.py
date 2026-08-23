"""CLI: ``python -m rpipe study run <study_dir>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='rpipe', description='RPipe study runner')
    sub = parser.add_subparsers(dest='cmd', required=True)

    study_p = sub.add_parser('study', help='Study orchestration')
    study_sub = study_p.add_subparsers(dest='study_cmd', required=True)

    run_p = study_sub.add_parser('run', help='Expand study.yaml, write index, launch Flow')
    run_p.add_argument('study_dir', type=Path, help='Path to studies/<name>/')
    run_p.add_argument(
        '--skip-launch',
        action='store_true',
        help='Only write Configs + index.json',
    )
    run_p.add_argument(
        '--phases',
        default='',
        help='comma-separated Flow phases (default: all)',
    )

    args = parser.parse_args(argv)
    if args.cmd == 'study' and args.study_cmd == 'run':
        from rpipe.study import run_study

        phases = [s.strip() for s in args.phases.split(',') if s.strip()] or None
        out = run_study(args.study_dir, skip_launch=args.skip_launch, phases=phases)
        for path in out['configs']:
            print(path)
        print(out['index'])
        for path in out['results']:
            print(path)
        return 0
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
