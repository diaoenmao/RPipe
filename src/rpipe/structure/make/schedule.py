"""GPU round-robin and ``&`` / ``wait`` launch scripts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from rpipe.structure.artifact import artifact_layout, load_config
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


def run_succeeded(study_dir: Path, run_id: str) -> bool:
    result_path = artifact_layout(study_dir, run_id).result_path
    if not result_path.is_file():
        return False
    try:
        data = decode_result(result_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return data.get('status') == STATUS_SUCCEEDED


def job_mode(config_path: Path) -> str:
    try:
        cfg = load_config(config_path)
    except (OSError, ValueError):
        return 'train'
    algo = cfg.get('algorithm')
    if isinstance(algo, dict):
        return str(algo.get('mode') or 'train')
    return 'train'


def job_waves(jobs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Train jobs first, then eval. Eval waits until the train wave finishes."""
    trains = [job for job in jobs if job.get('mode') != 'eval']
    evals = [job for job in jobs if job.get('mode') == 'eval']
    if trains and evals:
        return [trains, evals]
    return [jobs] if jobs else []


def _assign_gpus(jobs: list[dict[str, Any]], init_gpu: int, num_gpus: int) -> None:
    gpus = gpu_ids(init_gpu, num_gpus)
    if not gpus:
        for job in jobs:
            job.pop('gpu', None)
        return
    for i, job in enumerate(jobs):
        job['gpu'] = gpus[i % len(gpus)]


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
        pending.append(
            {
                'run_id': run_id,
                'config': str(path.as_posix()),
                'mode': job_mode(path),
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


def write_jobs_json(
    study_dir: Path,
    jobs: list[dict[str, Any]],
    *,
    round_size: int,
    init_gpu: int,
    num_gpus: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    payload = {
        'study_dir': str(study_dir.resolve()),
        'round': int(round_size),
        'init_gpu': int(init_gpu),
        'num_gpus': int(num_gpus),
        'jobs': jobs,
    }
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
        for start in range(0, len(wave), round_size):
            groups.append(wave[start : start + round_size])
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
    study_b = bash_path(study_dir)
    py_b = bash_path(Path(python_exe))
    repo_b = bash_path(repo)
    chunks: list[str] = []
    buf = ['#!/bin/bash', f'cd "{repo_b}"', 'export KMP_DUPLICATE_LIB_OK=TRUE']
    wait_groups_n = 0
    groups = wait_groups(jobs, round_size, batches)

    def _flush() -> None:
        chunks.append('\n'.join(buf) + '\n')

    def _reset() -> list[str]:
        return ['#!/bin/bash', f'cd "{repo_b}"', 'export KMP_DUPLICATE_LIB_OK=TRUE']

    for gi, group in enumerate(groups):
        last_group = gi == len(groups) - 1
        for i, job in enumerate(group):
            run_id = job['run_id']
            gpu = job.get('gpu')
            cmd = f'"{py_b}" -m rpipe run-one "{study_b}" "{run_id}"'
            if gpu is not None:
                bg = f'CUDA_VISIBLE_DEVICES="{gpu}" {cmd} &'
                fg = f'CUDA_VISIBLE_DEVICES="{gpu}" {cmd}'
            else:
                bg = f'{cmd} &'
                fg = cmd
            last = i == len(group) - 1
            if last:
                buf.append(fg)
                buf.append('wait')
                wait_groups_n += 1
                if wait_groups_n % split_round == 0 or last_group:
                    _flush()
                    buf = _reset()
            else:
                buf.append(bg)
    return chunks or ['#!/bin/bash\nwait\n']


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
                '$env:KMP_DUPLICATE_LIB_OK = "TRUE"',
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


def launch_job_env(job: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
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
) -> list[int]:
    if round_size < 1:
        raise ValueError('round must be >= 1')
    py = python_exe or sys.executable
    root = cwd or repo_root_from(study_dir)
    codes: list[int] = []

    def _run_groups(groups: list[list[dict[str, Any]]]) -> list[tuple[dict[str, Any], int]]:
        pairs: list[tuple[dict[str, Any], int]] = []
        for group in groups:
            procs: list[tuple[dict[str, Any], subprocess.Popen[str]]] = []
            for job in group:
                cmd = [py, '-m', 'rpipe', 'run-one', str(study_dir), str(job['run_id'])]
                print(f'+ gpu={job.get("gpu", "-")} {job["run_id"]}', flush=True)
                procs.append(
                    (job, subprocess.Popen(cmd, cwd=str(root), env=launch_job_env(job)))
                )
            for job, proc in procs:
                code = int(proc.wait())
                pairs.append((job, code))
                if code != 0:
                    print(f'error {job["run_id"]} exit={code}', flush=True)
        return pairs

    pairs = _run_groups(wait_groups(jobs, round_size, batches))
    codes.extend(code for _, code in pairs)
    if not retry_failed:
        return codes
    failed = [
        job
        for job, code in pairs
        if code != 0 or not run_succeeded(study_dir, str(job['run_id']))
    ]
    if not failed:
        return codes
    print(f'retry {len(failed)} failed jobs (resume latest)', flush=True)
    retry_pairs = _run_groups([[job] for job in failed])
    codes.extend(code for _, code in retry_pairs)
    return codes
