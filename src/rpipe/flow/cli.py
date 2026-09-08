"""flow cli: argv → make + Runner."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from rpipe.flow.context import FlowContext
from rpipe.flow.runner import FlowRunner
from rpipe.structure.artifact import artifact_layout, load_config
from rpipe.structure.make import (
    expand_study,
    launch_jobs,
    plan_jobs,
    write_launch_scripts,
)


def launch_one(
    study_dir: Path,
    run_id: str,
    phases: list[str] | None = None,
) -> Path:
    layout = artifact_layout(study_dir, run_id)
    cfg = load_config(layout.config_path)
    ctx = FlowContext(study_dir=study_dir, layout=layout, config=cfg)
    return FlowRunner(phases=phases).run(ctx)


def launch_runs(
    study_dir: Path,
    config_paths: list[Path],
    phases: list[str] | None = None,
) -> list[Path]:
    return [launch_one(study_dir, path.parent.name, phases=phases) for path in config_paths]


def run_study(
    study_dir: Path | str,
    *,
    skip_launch: bool = False,
    phases: list[str] | None = None,
) -> dict[str, Any]:
    out = expand_study(study_dir)
    result_paths: list[Path] = []
    if not skip_launch:
        result_paths = launch_runs(Path(out['study_dir']), list(out['configs']), phases=phases)
    return {
        'study_dir': out['study_dir'],
        'index': out['index'],
        'configs': out['configs'],
        'results': result_paths,
    }


def _parse_phases(raw: str) -> list[str] | None:
    return [s.strip() for s in raw.split(',') if s.strip()] or None


def _add_run_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('study_dir', type=Path, help='Path to studies/<name>/')
    parser.add_argument(
        '--skip-launch',
        action='store_true',
        help='Write configs and index only',
    )
    parser.add_argument(
        '--phases',
        default='',
        help='comma-separated phases; default is the full chain',
    )


def _add_make_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('study_dir', type=Path, help='Path to studies/<name>/')
    parser.add_argument('--init-gpu', default=0, type=int)
    parser.add_argument('--num-gpus', default=1, type=int)
    parser.add_argument(
        '--round',
        default=4,
        type=int,
        help='concurrent jobs before wait',
    )
    parser.add_argument('--split-round', default=65535, type=int)
    parser.add_argument(
        '--include-done',
        action='store_true',
        help='include Runs that already succeeded',
    )


def _make_from_args(args: argparse.Namespace) -> dict[str, Any]:
    out = expand_study(args.study_dir)
    study_dir = Path(out['study_dir'])
    jobs = plan_jobs(
        study_dir,
        list(out['configs']),
        init_gpu=args.init_gpu,
        num_gpus=args.num_gpus,
        include_done=bool(getattr(args, 'include_done', False)),
    )
    written = write_launch_scripts(
        study_dir,
        jobs,
        round_size=args.round,
        split_round=args.split_round,
        init_gpu=args.init_gpu,
        num_gpus=args.num_gpus,
    )
    return {'expand': out, 'job_list': jobs, **written}


def _execute_run(args: argparse.Namespace) -> int:
    phases = _parse_phases(args.phases)
    out = run_study(args.study_dir, skip_launch=args.skip_launch, phases=phases)
    for path in out['configs']:
        print(path)
    print(out['index'])
    for path in out['results']:
        print(path)
    return 0


def _print_make_paths(written: dict[str, Any]) -> None:
    print(written['expand']['index'])
    print(written['jobs_json'])
    print(written['ps1'])
    for path in written['bash']:
        print(path)
    print(f'{written["n_jobs"]} jobs')


def _execute_make(args: argparse.Namespace) -> int:
    _print_make_paths(_make_from_args(args))
    return 0


def _execute_launch(args: argparse.Namespace) -> int:
    written = _make_from_args(args)
    _print_make_paths(written)
    jobs = written['job_list']
    if not jobs:
        print('nothing to launch')
        return 0
    codes = launch_jobs(
        Path(written['expand']['study_dir']),
        jobs,
        round_size=args.round,
    )
    return 1 if any(code != 0 for code in codes) else 0


def _execute_run_one(args: argparse.Namespace) -> int:
    phases = _parse_phases(args.phases)
    path = launch_one(Path(args.study_dir).resolve(), args.run_id, phases=phases)
    print(path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='rpipe',
        description='Run a Study: make configs, then flow phases',
    )
    sub = parser.add_subparsers(dest='cmd', required=True)

    run_p = sub.add_parser('run', help='make configs and run each Flow in order')
    _add_run_flags(run_p)

    study_p = sub.add_parser('study', help='alias for run')
    study_sub = study_p.add_subparsers(dest='study_cmd', required=True)
    study_run = study_sub.add_parser('run', help='same as rpipe run')
    _add_run_flags(study_run)

    make_p = sub.add_parser('make', help='write configs, index, and launch scripts')
    _add_make_flags(make_p)

    launch_p = sub.add_parser('launch', help='make then run pending jobs by round and GPU')
    _add_make_flags(launch_p)

    one_p = sub.add_parser('run-one', help='run one Run id')
    one_p.add_argument('study_dir', type=Path)
    one_p.add_argument('run_id', type=str)
    one_p.add_argument('--phases', default='')

    args = parser.parse_args(argv)
    if args.cmd == 'run' or (args.cmd == 'study' and args.study_cmd == 'run'):
        return _execute_run(args)
    if args.cmd == 'make':
        return _execute_make(args)
    if args.cmd == 'launch':
        return _execute_launch(args)
    if args.cmd == 'run-one':
        return _execute_run_one(args)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
