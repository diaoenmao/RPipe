"""Estimate ``--round`` from Run configs and GPU memory.

make does not import data / model / algorithm. Complexity is a heuristic
from yaml fields; hardware comes from nvidia-smi (then torch). Seconds are
a conservative wall-clock guess for packing, not a measurement.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rpipe.structure.artifact import load_config
from rpipe.structure.make.schedule import job_waves

SAFETY = 0.5
TIME_SAFETY = 2.0
CPU_ROUND = 4
OVERHEAD_BYTES = 512 * 1024 * 1024

DATA_SHAPE: dict[str, tuple[int, int, int]] = {
    'MNIST': (1, 28, 28),
    'FashionMNIST': (1, 28, 28),
    'CIFAR10': (3, 32, 32),
    'CIFAR100': (3, 32, 32),
    'SVHN': (3, 32, 32),
}

# Activation volume relative to one input tensor (fp32 train).
MODEL_ACT: dict[str, int] = {
    'linear': 80,
    'mlp': 200,
    'cnn': 600,
    'resnet10': 1200,
    'resnet18': 1800,
    'wresnet': 2800,
}

# Params + grads + SGD momentum, rough.
MODEL_PARAM_BYTES: dict[str, int] = {
    'linear': 32 * 1024 * 1024,
    'mlp': 64 * 1024 * 1024,
    'cnn': 128 * 1024 * 1024,
    'resnet10': 256 * 1024 * 1024,
    'resnet18': 512 * 1024 * 1024,
    'wresnet': 1024 * 1024 * 1024,
}


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    total_bytes: int
    free_bytes: int


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def estimate_job_bytes(cfg: dict[str, Any] | None) -> int:
    """Peak VRAM guess for one ``run-one`` process."""
    if not isinstance(cfg, dict):
        return OVERHEAD_BYTES + MODEL_PARAM_BYTES['resnet18']
    data = cfg.get('data') if isinstance(cfg.get('data'), dict) else {}
    model = cfg.get('model') if isinstance(cfg.get('model'), dict) else {}
    algo = cfg.get('algorithm') if isinstance(cfg.get('algorithm'), dict) else {}
    dcfg = data.get('config') if isinstance(data.get('config'), dict) else {}
    name = str(data.get('name') or 'CIFAR10')
    model_name = str(model.get('name') or 'linear').lower()
    batch = max(1, _as_int(dcfg.get('batch_size'), 64))
    channels, height, width = DATA_SHAPE.get(name, (3, 224, 224))
    spatial = channels * height * width
    act = MODEL_ACT.get(model_name, 1800)
    params = MODEL_PARAM_BYTES.get(model_name, 512 * 1024 * 1024)
    activations = batch * spatial * 4 * act
    total = OVERHEAD_BYTES + params + activations
    if str(algo.get('mode') or 'train') == 'eval':
        total = int(total * 0.4)
    return max(OVERHEAD_BYTES, int(total))


# Conservative ms / optimizer step (slow card). Overestimate, not a benchmark.
MODEL_STEP_MS: dict[str, int] = {
    'linear': 15,
    'mlp': 25,
    'cnn': 40,
    'resnet10': 80,
    'resnet18': 120,
    'wresnet': 180,
}


def estimate_job_seconds(cfg: dict[str, Any] | None) -> int:
    """Conservative seconds for one ``run-one``. Wall clock uses max per wait group."""
    if not isinstance(cfg, dict):
        return int(math.ceil(600 * TIME_SAFETY))
    data = cfg.get('data') if isinstance(cfg.get('data'), dict) else {}
    model = cfg.get('model') if isinstance(cfg.get('model'), dict) else {}
    algo = cfg.get('algorithm') if isinstance(cfg.get('algorithm'), dict) else {}
    dcfg = data.get('config') if isinstance(data.get('config'), dict) else {}
    model_name = str(model.get('name') or 'linear').lower()
    batch = max(1, _as_int(dcfg.get('batch_size'), 64))
    train_size = max(1, _as_int(dcfg.get('train_size'), 50000))
    steps = _as_int(algo.get('num_steps'), 0)
    epochs = _as_int(algo.get('num_epochs'), 0)
    if steps <= 0 and epochs > 0:
        steps = epochs * max(1, int(math.ceil(train_size / batch)))
    if steps <= 0:
        steps = 1
    ms = MODEL_STEP_MS.get(model_name, 80)
    seconds = steps * ms / 1000.0
    if str(algo.get('mode') or 'train') == 'eval':
        seconds *= 0.25
    return max(1, int(math.ceil(seconds * TIME_SAFETY)))


def attach_estimates(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for job in jobs:
        need_cfg = (
            job.get('vram_bytes') is None
            or job.get('seconds') is None
            or not job.get('estimate_model')
        )
        path = job.get('config')
        cfg: dict[str, Any] | None = None
        if need_cfg and path:
            try:
                cfg = load_config(Path(str(path)))
            except (OSError, ValueError):
                cfg = None
        if job.get('vram_bytes') is None:
            job['vram_bytes'] = estimate_job_bytes(cfg)
        if job.get('seconds') is None:
            job['seconds'] = estimate_job_seconds(cfg)
        if cfg:
            data = cfg.get('data') if isinstance(cfg.get('data'), dict) else {}
            model = cfg.get('model') if isinstance(cfg.get('model'), dict) else {}
            system = cfg.get('system') if isinstance(cfg.get('system'), dict) else {}
            job['estimate_model'] = str(model.get('name') or '')
            job['estimate_data'] = str(data.get('name') or '')
            job['estimate_device'] = str(system.get('device') or 'cpu')
    return jobs


def probe_gpus(init_gpu: int, num_gpus: int) -> list[GpuInfo]:
    wanted = [init_gpu + i for i in range(max(0, num_gpus))]
    if not wanted:
        return []
    found = _probe_nvidia_smi(wanted)
    if found:
        return found
    return _probe_torch(wanted)


def _probe_torch(wanted: list[int]) -> list[GpuInfo]:
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    out: list[GpuInfo] = []
    count = int(torch.cuda.device_count())
    for index in wanted:
        if index < 0 or index >= count:
            continue
        free, total = torch.cuda.mem_get_info(index)
        prop = torch.cuda.get_device_properties(index)
        out.append(
            GpuInfo(
                index=index,
                name=str(prop.name),
                total_bytes=int(total),
                free_bytes=int(free),
            )
        )
    return out


def _probe_nvidia_smi(wanted: list[int]) -> list[GpuInfo]:
    exe = shutil.which('nvidia-smi')
    if not exe:
        return []
    try:
        raw = subprocess.check_output(
            [
                exe,
                '--query-gpu=index,name,memory.total,memory.free',
                '--format=csv,noheader,nounits',
            ],
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    by_index: dict[int, GpuInfo] = {}
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 4:
            continue
        try:
            index = int(parts[0])
            total_mb = float(parts[2])
            free_mb = float(parts[3])
        except ValueError:
            continue
        by_index[index] = GpuInfo(
            index=index,
            name=parts[1],
            total_bytes=int(total_mb * 1024 * 1024),
            free_bytes=int(free_mb * 1024 * 1024),
        )
    return [by_index[i] for i in wanted if i in by_index]


def usable_bytes(gpu: GpuInfo, *, safety: float = SAFETY) -> int:
    pool = min(int(gpu.free_bytes), int(gpu.total_bytes))
    return max(0, int(pool * safety))


def _job_vram(job: dict[str, Any]) -> int:
    return max(1, int(job.get('vram_bytes') or estimate_job_bytes(None)))


def round_fits(
    jobs: list[dict[str, Any]],
    round_size: int,
    gpu_usable: dict[str, int],
) -> bool:
    if round_size < 1:
        return False
    if not gpu_usable:
        return round_size == 1
    default_usable = min(gpu_usable.values())
    for wave in job_waves(jobs):
        for start in range(0, len(wave), round_size):
            load: dict[str, int] = {}
            for job in wave[start : start + round_size]:
                gpu = str(job.get('gpu', next(iter(gpu_usable))))
                load[gpu] = load.get(gpu, 0) + _job_vram(job)
            for gpu, total in load.items():
                if total > gpu_usable.get(gpu, default_usable):
                    return False
    return True


def suggest_round(
    jobs: list[dict[str, Any]],
    gpus: list[GpuInfo],
    *,
    safety: float = SAFETY,
    cap: int | None = None,
) -> int:
    n_jobs = len(jobs)
    if n_jobs == 0:
        return 1
    devices = {str(job.get('estimate_device') or 'cuda') for job in jobs}
    if devices and all(dev == 'cpu' for dev in devices):
        return min(n_jobs, CPU_ROUND if cap is None else cap)
    if not gpus:
        return 1
    gpu_usable = {str(g.index): usable_bytes(g, safety=safety) for g in gpus}
    heaviest = max(_job_vram(job) for job in jobs)
    min_usable = min(gpu_usable.values()) if gpu_usable else 0
    by_mem = 1 if heaviest <= 0 else max(1, min_usable // heaviest)
    upper = min(n_jobs, by_mem)
    if cap is not None:
        upper = min(upper, cap)
    chosen = 1
    for size in range(1, upper + 1):
        if round_fits(jobs, size, gpu_usable):
            chosen = size
    return chosen


def pack_jobs(
    jobs: list[dict[str, Any]],
    gpus: list[GpuInfo],
    *,
    safety: float = SAFETY,
) -> list[list[dict[str, Any]]]:
    """Same-type batches. Train then eval; never mix resnet with linear in one wait."""
    batches: list[list[dict[str, Any]]] = []
    for wave in job_waves(jobs):
        batches.extend(_pack_wave(wave, gpus, safety=safety))
    return batches


def _pack_class(job: dict[str, Any]) -> str:
    name = str(job.get('estimate_model') or '').strip().lower()
    return name or 'job'


def _pack_wave(
    wave: list[dict[str, Any]],
    gpus: list[GpuInfo],
    *,
    safety: float,
) -> list[list[dict[str, Any]]]:
    if not wave:
        return []
    clusters: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for job in wave:
        key = _pack_class(job)
        if key not in clusters:
            order.append(key)
            clusters[key] = []
        clusters[key].append(job)
    batches: list[list[dict[str, Any]]] = []
    for key in order:
        batches.extend(_pack_homogeneous(clusters[key], gpus, safety=safety))
    return batches


def _pack_homogeneous(
    jobs: list[dict[str, Any]],
    gpus: list[GpuInfo],
    *,
    safety: float,
) -> list[list[dict[str, Any]]]:
    if not jobs:
        return []
    devices = {str(job.get('estimate_device') or 'cuda') for job in jobs}
    cpu_only = bool(devices) and all(dev == 'cpu' for dev in devices)
    if cpu_only:
        return [jobs[i : i + CPU_ROUND] for i in range(0, len(jobs), CPU_ROUND)]
    if not gpus:
        return [[job] for job in jobs]
    gpu_names = [str(gpu.index) for gpu in gpus]
    usable = {str(gpu.index): usable_bytes(gpu, safety=safety) for gpu in gpus}
    remaining = list(jobs)
    batches: list[list[dict[str, Any]]] = []
    while remaining:
        load = {name: 0 for name in gpu_names}
        batch: list[dict[str, Any]] = []
        for job in remaining:
            vram = _job_vram(job)
            pick = None
            most_left = -1
            for name in gpu_names:
                left = usable[name] - load[name]
                if vram <= left and left > most_left:
                    pick = name
                    most_left = left
            if pick is None:
                continue
            job['gpu'] = pick
            load[pick] += vram
            batch.append(job)
        if not batch:
            job = remaining[0]
            job['gpu'] = gpu_names[0]
            batch = [job]
        taken = {id(job) for job in batch}
        remaining = [job for job in remaining if id(job) not in taken]
        batches.append(batch)
    return batches


def _job_seconds(job: dict[str, Any]) -> int:
    return max(0, int(job.get('seconds') or 0))


def estimate_wall_seconds(batches: list[list[dict[str, Any]]]) -> int:
    """Sum of per-wait max(job seconds). Group wall equals the slowest job."""
    total = 0
    for batch in batches:
        total += max((_job_seconds(job) for job in batch), default=0)
    return int(total)


def format_duration(seconds: int) -> str:
    value = max(0, int(seconds))
    hours, rem = divmod(value, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        if minutes:
            return f'{hours}h{minutes}m'
        return f'{hours}h'
    if minutes:
        if secs:
            return f'{minutes}m{secs}s'
        return f'{minutes}m'
    return f'{secs}s'


def pack_label(models: list[str]) -> str:
    """Collapse consecutive same names: linear,linear,linear → linear×3."""
    if not models:
        return ''
    parts: list[str] = []
    i = 0
    while i < len(models):
        name = models[i]
        j = i + 1
        while j < len(models) and models[j] == name:
            j += 1
        count = j - i
        parts.append(name if count == 1 else f'{name}×{count}')
        i = j
    return '+'.join(parts)


def batch_summaries(batches: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch in batches:
        models = [str(job.get('estimate_model') or job.get('mode') or 'job') for job in batch]
        seconds = max((_job_seconds(job) for job in batch), default=0)
        rows.append(
            {
                'size': len(batch),
                'vram_bytes': sum(_job_vram(job) for job in batch),
                'seconds': seconds,
                'models': models,
                'run_ids': [str(job.get('run_id')) for job in batch],
                'label': pack_label(models),
            }
        )
    return rows


def format_bytes(n: int) -> str:
    value = float(n)
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if value < 1024 or unit == 'TiB':
            if unit == 'B':
                return f'{int(value)}{unit}'
            return f'{value:.1f}{unit}'
        value /= 1024
    return f'{n}B'


def capacity_report(
    jobs: list[dict[str, Any]],
    gpus: list[GpuInfo],
    *,
    round_size: int,
    round_source: str,
    safety: float = SAFETY,
) -> dict[str, Any]:
    heaviest = max((_job_vram(j) for j in jobs), default=0)
    min_usable = 0
    if gpus:
        min_usable = min(usable_bytes(g, safety=safety) for g in gpus)
    slots = 1 if heaviest <= 0 else max(1, min_usable // heaviest) if min_usable else 0
    return {
        'round': int(round_size),
        'round_source': round_source,
        'safety': safety,
        'n_jobs': len(jobs),
        'heaviest_bytes': heaviest,
        'slots_from_mem': int(slots),
        'batches': [],
        'gpus': [
            {
                'index': g.index,
                'name': g.name,
                'total_bytes': g.total_bytes,
                'free_bytes': g.free_bytes,
                'usable_bytes': usable_bytes(g, safety=safety),
            }
            for g in gpus
        ],
    }


def summarize_capacity(report: dict[str, Any]) -> str:
    gpus = report.get('gpus') or []
    gpu_txt = 'no-gpu'
    usable_txt = '0B'
    if gpus:
        parts = []
        for gpu in gpus:
            parts.append(
                'GPU{0} {1} free {2} usable {3}'.format(
                    gpu['index'],
                    gpu['name'],
                    format_bytes(int(gpu['free_bytes'])),
                    format_bytes(int(gpu['usable_bytes'])),
                )
            )
        gpu_txt = '; '.join(parts)
        usable_txt = format_bytes(int(gpus[0]['usable_bytes']))
    batches = report.get('batches') or []
    if batches:
        packed = ', '.join(
            '{0}[{1}]'.format(row.get('size'), row.get('label')) for row in batches
        )
        wall = int(report.get('wall_seconds') or 0)
        if wall <= 0:
            wall = sum(int(row.get('seconds') or 0) for row in batches)
        wall_txt = ' est wall {0}'.format(format_duration(wall)) if wall else ''
        return 'pack {0} waits: {1}{2} | {3}'.format(len(batches), packed, wall_txt, gpu_txt)
    return (
        'round={round} ({source}) = min({n} pending, {usable} // {heavy} = {slots} slots) | {gpu}'
        .format(
            round=report.get('round'),
            source=report.get('round_source'),
            n=report.get('n_jobs'),
            usable=usable_txt,
            heavy=format_bytes(int(report.get('heaviest_bytes') or 0)),
            slots=report.get('slots_from_mem'),
            gpu=gpu_txt,
        )
    )
