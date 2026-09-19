"""flow cli: argv → make + Runner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from rpipe.flow.process import process_path, run_study as process_study
from rpipe.flow.context import FlowContext
from rpipe.flow.runner import FlowRunner
from rpipe.structure.artifact import artifact_layout, load_config
from rpipe.structure.make import (
    expand_study,
    launch_jobs,
    load_launch_plan,
    plan_jobs,
    run_succeeded,
    write_launch_scripts,
)
from rpipe.structure.api import data_api
from rpipe.structure.make.capacity import (
    attach_estimates,
    batch_summaries,
    capacity_report,
    estimate_wall_seconds,
    pack_jobs,
    probe_gpus,
    requires_gpu,
    summarize_capacity,
)


def _prepare_shared(study_dir: Path | str, config_paths: list[Path]) -> None:
    names = data_api.prepare_shared(study_dir, config_paths)
    if names:
        print('shared data: ' + ', '.join(names), flush=True)


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
        _prepare_shared(out['study_dir'], list(out['configs']))
        result_paths = launch_runs(Path(out['study_dir']), list(out['configs']), phases=phases)
        process_study(Path(out['study_dir']))
    return {
        'study_dir': out['study_dir'],
        'index': out['index'],
        'configs': out['configs'],
        'results': result_paths,
    }


def _parse_round(text: str) -> int:
    raw = str(text).strip().lower()
    if raw in ('auto', '0'):
        return 0
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('round must be an int or auto') from exc
    if value < 0:
        raise argparse.ArgumentTypeError('round must be >= 0 (0=auto)')
    return value


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
        default=0,
        type=_parse_round,
        metavar='N|auto',
        help='0/auto = same-type wait groups packed by VRAM; N = fixed chunk size',
    )
    parser.add_argument('--split-round', default=65535, type=int)
    parser.add_argument(
        '--include-done',
        action='store_true',
        help='include Runs that already succeeded',
    )
    parser.add_argument(
        '--console',
        default='auto',
        help='auto/new/shared: Windows auto=new window per run-one; shared=mix in this terminal',
    )


def _make_from_args(args: argparse.Namespace) -> dict[str, Any]:
    out = expand_study(args.study_dir)
    _prepare_shared(out['study_dir'], list(out['configs']))
    study_dir = Path(out['study_dir'])
    jobs = plan_jobs(
        study_dir,
        list(out['configs']),
        init_gpu=args.init_gpu,
        num_gpus=args.num_gpus,
        include_done=bool(getattr(args, 'include_done', False)),
    )
    attach_estimates(jobs)
    gpus = probe_gpus(int(args.init_gpu), int(args.num_gpus)) if requires_gpu(jobs) else []
    requested = int(args.round)
    batches: list | None
    if requested <= 0:
        batches = pack_jobs(jobs, gpus)
        jobs = [job for group in batches for job in group]
        round_size = max((len(group) for group in batches), default=1)
        round_source = 'pack'
    else:
        batches = None
        round_size = requested
        round_source = 'cli'
    report = capacity_report(
        jobs,
        gpus,
        round_size=round_size,
        round_source=round_source,
    )
    if batches is not None:
        report['batches'] = batch_summaries(batches)
        report['wall_seconds'] = estimate_wall_seconds(batches)
    written = write_launch_scripts(
        study_dir,
        jobs,
        round_size=round_size,
        split_round=args.split_round,
        init_gpu=args.init_gpu,
        num_gpus=args.num_gpus,
        extra={'capacity': report},
        batches=batches,
    )
    written['round'] = round_size
    written['capacity'] = report
    written['batches'] = batches
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
    cap = written.get('capacity')
    if isinstance(cap, dict):
        print(summarize_capacity(cap))


def _execute_make(args: argparse.Namespace) -> int:
    _print_make_paths(_make_from_args(args))
    return 0


def _execute_launch(args: argparse.Namespace) -> int:
    written = None
    if not bool(getattr(args, 'remake', False)):
        written = load_launch_plan(
            args.study_dir,
            init_gpu=int(args.init_gpu),
            num_gpus=int(args.num_gpus),
            round_size=int(args.round),
            include_done=bool(getattr(args, 'include_done', False)),
        )
    if written is None:
        written = _make_from_args(args)
        _print_make_paths(written)
    else:
        print(written['jobs_json'], flush=True)
        print(f'{written["n_jobs"]} jobs', flush=True)
    jobs = written['job_list']
    study_dir = Path(written.get('expand', {}).get('study_dir') or args.study_dir).resolve()
    if not jobs:
        print('nothing to launch')
        process_study(study_dir)
        print(process_path(study_dir))
        return 0
    launch_jobs(
        study_dir,
        jobs,
        round_size=int(written.get('round') or 1),
        batches=written.get('batches'),
        console=str(getattr(args, 'console', 'auto')),
    )
    still = [
        job['run_id']
        for job in jobs
        if not run_succeeded(study_dir, str(job['run_id']))
    ]
    if still:
        print('still failed: ' + ' '.join(still), flush=True)
    body = process_study(study_dir)
    print(process_path(study_dir), flush=True)
    if not body.get('complete'):
        print('process partial', flush=True)
    return 1 if still else 0


def _execute_process(args: argparse.Namespace) -> int:
    body = process_study(Path(args.study_dir).resolve())
    print(process_path(args.study_dir))
    print('complete' if body.get('complete') else 'partial')
    return 0


def _execute_run_one(args: argparse.Namespace) -> int:
    phases = _parse_phases(args.phases)
    path = launch_one(Path(args.study_dir).resolve(), args.run_id, phases=phases)
    print(path)
    return 0


def _configure_stdio() -> None:
    """Windows consoles default to GBK; pack labels use ``×`` and need UTF-8."""
    if sys.platform != 'win32':
        return
    try:
        import ctypes

        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except (AttributeError, OSError):
        pass
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, 'reconfigure', None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding='utf-8', errors='replace')
        except (OSError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
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

    launch_p = sub.add_parser('launch', help='run pending jobs from make; make only if jobs.json is missing')
    _add_make_flags(launch_p)
    launch_p.add_argument(
        '--remake',
        action='store_true',
        help='ignore scripts/jobs.json and run make again',
    )

    one_p = sub.add_parser('run-one', help='run one Run id')
    one_p.add_argument('study_dir', type=Path)
    one_p.add_argument('run_id', type=str)
    one_p.add_argument('--phases', default='')

    proc_p = sub.add_parser('process', help='Study-level process: mean/std/min/max history')
    proc_p.add_argument('study_dir', type=Path, help='Path to studies/<name>/')

    args = parser.parse_args(argv)
    if args.cmd == 'run' or (args.cmd == 'study' and args.study_cmd == 'run'):
        return _execute_run(args)
    if args.cmd == 'make':
        return _execute_make(args)
    if args.cmd == 'launch':
        return _execute_launch(args)
    if args.cmd == 'run-one':
        return _execute_run_one(args)
    if args.cmd == 'process':
        return _execute_process(args)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
