"""GPU round-robin and ``&`` / ``wait`` launch scripts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from rpipe.structure.api.algorithm_api import sibling_train_dependency
from rpipe.structure.artifact import artifact_layout, load_config
from rpipe.structure.artifact.index import load_index
from rpipe.structure.artifact.result import write_result
from rpipe.structure.artifact.result.format import STATUS_SUCCEEDED, decode_result

SCRIPTS_DIRNAME = 'scripts'
JOBS_NAME = 'jobs.json'


def repo_root_from(start: Path) -> Path:
    cur = start.resolve()
    for candidate in (cur, *cur.parents):
        pyproject = candidate / 'pyproject.toml'
        if pyproject.is_file():
            return candidate
    return Path.cwd().resolve()


def gpu_ids(init_gpu: int, num_gpus: int) -> list[str]:
    if num_gpus <= 0:
        return []
    return [str(init_gpu + i) for i in range(num_gpus)]


def bash_path(path: Path) -> str:
    text = str(path.resolve())
    if len(text) >= 2 and text[1] == ':':
        return '/' + text[0].lower() + text[2:].replace('\\', '/')
    return text.replace('\\', '/')


def _run_result(study_dir: Path, run_id: str) -> dict[str, Any]:
    result_path = Path(study_dir) / 'runs' / run_id / 'result.json'
    try:
        return decode_result(result_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def run_succeeded(study_dir: Path, run_id: str) -> bool:
    if _run_result(study_dir, run_id).get('status') != STATUS_SUCCEEDED:
        return False
    parent = sibling_train_dependency(study_dir, run_id)
    if parent is None:
        return True
    if _run_result(study_dir, parent).get('status') != STATUS_SUCCEEDED:
        return False
    # ponytail: conservative local freshness; copied/backdated artifacts need explicit re-evaluation.
    try:
        runs = Path(study_dir) / 'runs'
        return (runs / parent / 'result.json').stat().st_mtime_ns <= (runs / run_id / 'result.json').stat().st_mtime_ns
    except OSError:
        return False


def _fail_result(study_dir: Path, run_id: str, error: str) -> None:
    body = _run_result(study_dir, run_id)
    body.update(status='failed', error=error)
    write_result(artifact_layout(study_dir, run_id).result_path, body)


def _job_fields(config_path: Path) -> tuple[str, str]:
    try:
        cfg = load_config(config_path)
    except (OSError, ValueError):
        return 'train', 'cpu'
    algo = cfg.get('algorithm')
    system = cfg.get('system')
    mode = str(algo.get('mode') or 'train') if isinstance(algo, dict) else 'train'
    device = str(system.get('device') or 'cpu') if isinstance(system, dict) else 'cpu'
    return mode, device.lower()


def job_mode(job: dict[str, Any]) -> str:
    return str(job.get('mode') or 'train')


def filter_jobs_by_mode(
    jobs: list[dict[str, Any]],
    modes: list[str] | tuple[str, ...] | None,
) -> list[dict[str, Any]]:
    """Keep jobs whose ``mode`` is in ``modes``. Empty/None = no filter."""
    if not modes:
        return list(jobs)
    wanted = {str(mode).strip().lower() for mode in modes if str(mode).strip()}
    if not wanted:
        return list(jobs)
    return [job for job in jobs if job_mode(job) in wanted]


def filter_batches_by_mode(
    batches: list[list[dict[str, Any]]] | None,
    modes: list[str] | tuple[str, ...] | None,
) -> list[list[dict[str, Any]]] | None:
    if batches is None:
        return None
    if not modes:
        return [list(group) for group in batches if group]
    return [kept for kept in (filter_jobs_by_mode(list(group), modes) for group in batches) if kept]


def job_waves(jobs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Train jobs first, then eval. Eval waits until the train wave finishes."""
    trains = [job for job in jobs if job.get('mode') != 'eval']
    evals = [job for job in jobs if job.get('mode') == 'eval']
    if trains and evals:
        return [trains, evals]
    return [jobs] if jobs else []


def _assign_gpus(jobs: list[dict[str, Any]], init_gpu: int, num_gpus: int) -> None:
    gpus = gpu_ids(init_gpu, num_gpus)
    gpu_index = 0
    for job in jobs:
        if str(job.get('device') or 'cuda').lower() == 'cpu' or not gpus:
            job.pop('gpu', None)
            continue
        job['gpu'] = gpus[gpu_index % len(gpus)]
        gpu_index += 1


def plan_jobs(
    study_dir: Path,
    config_paths: list[Path],
    *,
    init_gpu: int = 0,
    num_gpus: int = 1,
    include_done: bool = False,
) -> list[dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    for path in config_paths:
        run_id = path.parent.name
        if not include_done and run_succeeded(study_dir, run_id):
            continue
        mode, device = _job_fields(path)
        pending.append(
            {
                'run_id': run_id,
                'config': str(path.as_posix()),
                'mode': mode,
                'device': device,
            }
        )
    ordered: list[dict[str, Any]] = []
    for wave in job_waves(pending):
        ordered.extend(wave)
    _assign_gpus(ordered, init_gpu, num_gpus)
    return ordered


def scripts_dir(study_dir: Path) -> Path:
    path = study_dir / SCRIPTS_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_launch_plan(
    study_dir: Path | str,
    *,
    init_gpu: int,
    num_gpus: int,
    round_size: int,
    include_done: bool = False,
) -> dict[str, Any] | None:
    """Reuse ``scripts/jobs.json`` from a prior make. None → caller should make.

    ``include_done`` keeps already-succeeded jobs that are still listed in the
    file. It does not invent Runs that were omitted from the original make;
    those need ``--remake --include-done``.
    """
    study_dir = Path(study_dir).resolve()
    path = study_dir / SCRIPTS_DIRNAME / JOBS_NAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get('jobs'), list):
        return None
    if int(payload.get('init_gpu', 0)) != int(init_gpu):
        return None
    if int(payload.get('num_gpus', 1)) != int(num_gpus):
        return None
    stored_round = int(payload.get('round') or 1)
    if int(round_size) > 0 and stored_round != int(round_size):
        return None

    def _keep(rows: list) -> list[dict[str, Any]]:
        kept = [job for job in rows if isinstance(job, dict)]
        if include_done:
            return kept
        return [job for job in kept if not run_succeeded(study_dir, str(job.get('run_id') or ''))]

    jobs = _keep(list(payload.get('jobs') or []))
    raw_batches = payload.get('batches')
    batches = None
    if isinstance(raw_batches, list) and raw_batches and isinstance(raw_batches[0], list):
        batches = [kept for kept in (_keep(list(group)) for group in raw_batches) if kept]
    return {
        'jobs_json': path,
        'job_list': jobs,
        'round': stored_round,
        'batches': batches,
        'n_jobs': len(jobs),
        'expand': {'study_dir': study_dir, 'index': study_dir / 'index.json', 'configs': []},
    }


def write_jobs_json(
    study_dir: Path,
    jobs: list[dict[str, Any]],
    *,
    round_size: int,
    init_gpu: int,
    num_gpus: int,
    extra: dict[str, Any] | None = None,
    batches: list[list[dict[str, Any]]] | None = None,
) -> Path:
    payload = {
        'study_dir': str(study_dir.resolve()),
        'round': int(round_size),
        'init_gpu': int(init_gpu),
        'num_gpus': int(num_gpus),
        'jobs': jobs,
    }
    if batches:
        payload['batches'] = batches
    if extra:
        payload.update(extra)
    dest = scripts_dir(study_dir) / JOBS_NAME
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return dest


def wait_groups(
    jobs: list[dict[str, Any]],
    round_size: int,
    batches: list[list[dict[str, Any]]] | None = None,
) -> list[list[dict[str, Any]]]:
    if batches:
        return [group for group in batches if group]
    groups: list[list[dict[str, Any]]] = []
    for wave in job_waves(jobs):
        resources: dict[str, list[dict[str, Any]]] = {}
        for job in wave:
            device = str(job.get('device') or 'cuda').lower()
            resources.setdefault(device, []).append(job)
        for rows in resources.values():
            for start in range(0, len(rows), round_size):
                groups.append(rows[start : start + round_size])
    return groups


def render_bash(
    study_dir: Path,
    jobs: list[dict[str, Any]],
    *,
    python_exe: str,
    round_size: int,
    split_round: int,
    batches: list[list[dict[str, Any]]] | None = None,
) -> list[str]:
    if round_size < 1:
        raise ValueError('round must be >= 1')
    if split_round < 1:
        raise ValueError('split_round must be >= 1')
    repo = repo_root_from(study_dir)
    py_b = bash_path(Path(python_exe))
    repo_b = bash_path(repo)
    chunks: list[str] = []
    groups = wait_groups(jobs, round_size, batches)
    partitions = [groups[start:start + split_round] for start in range(0, len(groups), split_round)] or [[]]
    for index, part in enumerate(partitions):
        body = [
            'from pathlib import Path',
            'from rpipe.structure.make.schedule import launch_jobs, run_succeeded',
            f'study = Path({str(study_dir.resolve())!r})',
            f'groups = {part!r}',
            'jobs = [job for group in groups for job in group]',
            f"launch_jobs(study, jobs, round_size={round_size}, batches=groups, console='shared')",
            "failed = any(not run_succeeded(study, str(job['run_id'])) for job in jobs)",
        ]
        if index == len(partitions) - 1:
            body.extend([
                'from rpipe.flow.process import run_study',
                'result = run_study(study)',
                "print(study / 'process.json', flush=True)",
                "failed = failed or not result.get('complete', False)",
            ])
        body.append('raise SystemExit(int(failed))')
        chunks.append('\n'.join([
            '#!/bin/bash', f'cd "{repo_b}"', f'"{py_b}" - <<\'RPIPE_LAUNCH\'',
            *body, 'RPIPE_LAUNCH', '',
        ]))
    return chunks


def write_launch_scripts(
    study_dir: Path,
    jobs: list[dict[str, Any]],
    *,
    round_size: int = 4,
    split_round: int = 65535,
    python_exe: str | None = None,
    init_gpu: int = 0,
    num_gpus: int = 1,
    extra: dict[str, Any] | None = None,
    batches: list[list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    py = python_exe or sys.executable
    dest = scripts_dir(study_dir)
    jobs_path = write_jobs_json(
        study_dir,
        jobs,
        round_size=round_size,
        init_gpu=init_gpu,
        num_gpus=num_gpus,
        extra=extra,
        batches=batches,
    )
    chunks = render_bash(
        study_dir,
        jobs,
        python_exe=py,
        round_size=round_size,
        split_round=split_round,
        batches=batches,
    )
    sh_paths: list[Path] = []
    stem = 'launch'
    for k, text in enumerate(chunks, start=1):
        name = f'{stem}_{k}.sh' if len(chunks) > 1 else f'{stem}.sh'
        path = dest / name
        path.write_text(text, encoding='utf-8', newline='\n')
        sh_paths.append(path)
    ps1 = dest / 'launch.ps1'
    study_win = str(study_dir.resolve())
    py_win = str(Path(py).resolve())
    ps1.write_text(
        '\n'.join(
            [
                '$ErrorActionPreference = "Stop"',
                f'Set-Location -LiteralPath {json.dumps(str(repo_root_from(study_dir)))}',
                f'& {json.dumps(py_win)} -m rpipe launch {json.dumps(study_win)}',
                '',
            ]
        ),
        encoding='utf-8',
        newline='\n',
    )
    return {
        'scripts_dir': dest,
        'jobs_json': jobs_path,
        'bash': sh_paths,
        'ps1': ps1,
        'n_jobs': len(jobs),
    }


def job_popen_kwargs(console: str) -> dict[str, Any]:
    """Windows ``new`` = one console window per run-one (same idea as Start-Process)."""
    if console == 'new' and os.name == 'nt':
        flag = getattr(subprocess, 'CREATE_NEW_CONSOLE', 0x00000010)
        return {'creationflags': flag}
    return {}


def resolve_console(mode: str = 'auto') -> str:
    raw = str(mode or 'auto').strip().lower()
    if raw in ('', 'auto'):
        return 'new' if os.name == 'nt' else 'shared'
    if raw in ('new', 'window', 'windows', 'separate'):
        return 'new'
    if raw in ('shared', 'same', 'mix'):
        return 'shared'
    raise ValueError('console must be auto, new, or shared')


def launch_job_env(job: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    gpu = job.get('gpu')
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    return env


def launch_jobs(
    study_dir: Path,
    jobs: list[dict[str, Any]],
    *,
    round_size: int = 4,
    python_exe: str | None = None,
    cwd: Path | None = None,
    batches: list[list[dict[str, Any]]] | None = None,
    retry_failed: bool = True,
    console: str = 'auto',
) -> list[int]:
    if round_size < 1:
        raise ValueError('round must be >= 1')
    py = python_exe or sys.executable
    root = cwd or repo_root_from(study_dir)
    codes: list[int] = []
    console_mode = resolve_console(console)
    popen_extra = job_popen_kwargs(console_mode)
    try:
        index = load_index(study_dir)
    except (OSError, TypeError, ValueError):
        index = {}
    dependencies = {
        run_id: parent
        for exp in index.get('experiments') or []
        for run in exp.get('runs') or []
        if (run_id := str(run.get('run_dir') or run.get('id') or ''))
        if (parent := sibling_train_dependency(study_dir, run_id)) is not None
    }
    for run_id, parent in dependencies.items():
        if _run_result(study_dir, run_id).get('status') == STATUS_SUCCEEDED and not run_succeeded(study_dir, run_id):
            _fail_result(study_dir, run_id, f'sibling train {parent} is unfinished or newer; re-evaluation required')
    groups = wait_groups(jobs, round_size, batches)
    # A supplied batch must not bypass the train/eval barrier either.
    waves = [
        [kept for group in groups if (kept := [job for job in group if (job_mode(job) == 'eval') == is_eval])]
        for is_eval in (False, True)
    ]
    total_waits = sum(len(wave) for wave in waves)
    wait_index = 0

    def _group_mode(group: list[dict[str, Any]]) -> str:
        modes: list[str] = []
        for job in group:
            mode = str(job.get('mode') or 'train')
            if mode not in modes:
                modes.append(mode)
        return '+'.join(modes) or 'train'

    def _run_groups(
        groups: list[list[dict[str, Any]]],
        *,
        kind: str,
    ) -> list[tuple[dict[str, Any], int]]:
        nonlocal wait_index
        pairs: list[tuple[dict[str, Any], int]] = []
        total = len(groups)
        for index, group in enumerate(groups, start=1):
            if kind == 'wait':
                wait_index += 1
            position = f'{wait_index}/{total_waits}' if kind == 'wait' else f'{index}/{total}'
            print(f'launch: {kind} {position} mode={_group_mode(group)}', flush=True)
            procs: list[tuple[dict[str, Any], subprocess.Popen[str]]] = []
            for job in group:
                run_id = str(job['run_id'])
                parent = dependencies.get(run_id)
                if parent is not None and not run_succeeded(study_dir, parent):
                    message = f'blocked: sibling train {parent} did not succeed'
                    _fail_result(study_dir, run_id, message)
                    print(f'error {run_id} {message}', flush=True)
                    pairs.append((job, 1))
                    continue
                for eval_id, train_id in dependencies.items():
                    if train_id == run_id and _run_result(study_dir, eval_id).get('status') == STATUS_SUCCEEDED:
                        _fail_result(study_dir, eval_id, f'sibling train {run_id} is being rerun; re-evaluation required')
                if _run_result(study_dir, run_id).get('status') == STATUS_SUCCEEDED:
                    _fail_result(study_dir, run_id, 'rerun started; no successful result from this execution yet')
                cmd = [py, '-m', 'rpipe', 'run-one', str(study_dir), run_id]
                where = 'window' if console_mode == 'new' and os.name == 'nt' else 'here'
                resource = f'gpu={job["gpu"]}' if job.get('gpu') is not None else 'cpu'
                print(f'+ {resource} {job["run_id"]} ({where})', flush=True)
                procs.append(
                    (
                        job,
                        subprocess.Popen(
                            cmd,
                            cwd=str(root),
                            env=launch_job_env(job),
                            **popen_extra,
                        ),
                    )
                )
            for job, proc in procs:
                code = int(proc.wait())
                pairs.append((job, code))
                if code != 0:
                    print(f'error {job["run_id"]} exit={code}', flush=True)
                    if _run_result(study_dir, str(job['run_id'])).get('status') == STATUS_SUCCEEDED:
                        _fail_result(study_dir, str(job['run_id']), f'run-one exited with code {code}')
        return pairs

    for wave in waves:
        pairs = _run_groups(wave, kind='wait')
        codes.extend(code for _, code in pairs)
        failed = [
            job for job, code in pairs
            if (code != 0 or not run_succeeded(study_dir, str(job['run_id'])))
            and (dependencies.get(str(job['run_id'])) is None
                 or run_succeeded(study_dir, dependencies[str(job['run_id'])]))
        ]
        if retry_failed and failed:
            print(f'retry {len(failed)} failed jobs (resume latest)', flush=True)
            retry_pairs = _run_groups([[job] for job in failed], kind='retry')
            codes.extend(code for _, code in retry_pairs)
    return codes
