"""Estimate ``--round`` from Run configs and GPU memory.

make does not import data / model / algorithm. Complexity is a heuristic
from yaml fields; hardware comes from nvidia-smi (then torch). Seconds are process start plus train steps plus each scheduled test pass.
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
CPU_ROUND = 4
OVERHEAD_BYTES = 512 * 1024 * 1024

DATA_SHAPE: dict[str, tuple[int, int, int]] = {
    'MNIST': (1, 28, 28),
    'FashionMNIST': (1, 28, 28),
    'CIFAR10': (3, 32, 32),
    'CIFAR100': (3, 32, 32),
    'SVHN': (3, 32, 32),
}

# ms/step is at batch 64. linear comes from mnist_train_size (2026-09-30):
# a run is about 3s to start, 6 ms per linear step, and about 1s per full MNIST test.
# mlp / cnn / resnet18 are the same shape fitted to cifar_grid train times.
# resnet10 / wresnet are not measured; they sit on that scale.
REF_BATCH = 64
STARTUP_SECONDS = 3
_TEST_ROWS = {
    'MNIST': 10000,
    'FashionMNIST': 10000,
    'CIFAR10': 10000,
    'CIFAR100': 10000,
    'SVHN': 26032,
}
# act multiplier, param MiB, ms/step at REF_BATCH
_MODEL = {
    'linear': (80, 32, 6),
    'mlp': (200, 64, 6),
    'cnn': (600, 128, 10),
    'resnet10': (1200, 256, 14),
    'resnet18': (1800, 512, 18),
    'wresnet': (2800, 1024, 28),
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
        return OVERHEAD_BYTES + _MODEL['resnet18'][1] * 1024 * 1024
    data = cfg.get('data') if isinstance(cfg.get('data'), dict) else {}
    model = cfg.get('model') if isinstance(cfg.get('model'), dict) else {}
    algo = cfg.get('algorithm') if isinstance(cfg.get('algorithm'), dict) else {}
    dcfg = data.get('config') if isinstance(data.get('config'), dict) else {}
    name = str(data.get('name') or 'CIFAR10')
    model_name = str(model.get('name') or 'linear').lower()
    batch = max(1, _as_int(dcfg.get('batch_size'), 64))
    channels, height, width = DATA_SHAPE.get(name, (3, 224, 224))
    spatial = channels * height * width
    act, param_mib, _ = _MODEL.get(model_name, _MODEL['resnet18'])
    params = param_mib * 1024 * 1024
    activations = batch * spatial * 4 * act
    total = OVERHEAD_BYTES + params + activations
    if str(algo.get('mode') or 'train') == 'eval':
        total = int(total * 0.4)
    return max(OVERHEAD_BYTES, int(total))


def _ms_per_step(model_name: str, batch: int) -> float:
    ms = _MODEL.get(model_name, _MODEL['resnet18'])[2]
    return float(ms) * (max(int(batch), 1) / REF_BATCH)


def _scheduled_test_passes(algo: dict[str, Any], units: int) -> int:
    """How many full test passes this train run makes. ``eval_period <= 0`` is once at the end."""
    raw = algo.get('eval_period')
    period = 1 if raw is None else _as_int(raw, 1)
    if period <= 0:
        return 1
    if units <= 0:
        return 0
    return units // period


def estimate_job_seconds(cfg: dict[str, Any] | None) -> int:
    """Seconds for one run: process start + train steps + each scheduled test pass.

    Wall clock is still the sum of the slowest job in each wait group.
    There is no safety multiplier on top of this sum.
    """
    if not isinstance(cfg, dict):
        return STARTUP_SECONDS
    data = cfg.get('data') if isinstance(cfg.get('data'), dict) else {}
    model = cfg.get('model') if isinstance(cfg.get('model'), dict) else {}
    algo = cfg.get('algorithm') if isinstance(cfg.get('algorithm'), dict) else {}
    dcfg = data.get('config') if isinstance(data.get('config'), dict) else {}
    model_name = str(model.get('name') or 'linear').lower()
    batch = max(1, _as_int(dcfg.get('batch_size'), REF_BATCH))
    ratio = float(dcfg.get('test_batch_ratio') or 1)
    test_batch = max(1, int(batch * ratio))
    data_name = str(data.get('name') or '')
    epochs = _as_int(algo.get('num_epochs'), 0)
    steps = _as_int(algo.get('num_steps'), 0)
    if epochs > 0:
        rows = _as_int(dcfg.get('train_size'), 0)
        if rows <= 0:
            rows = _TEST_ROWS.get(data_name, 50000)
        steps = epochs * max(1, int(math.ceil(rows / batch)))
    elif steps <= 0:
        steps = 1
    unit = str(algo.get('progress_unit') or 'step').lower()
    units = epochs if unit == 'epoch' and epochs > 0 else steps
    mode = str(algo.get('mode') or 'train')
    passes = 1 if mode == 'eval' else _scheduled_test_passes(algo, units)
    limit = algo.get('eval_num_steps')
    if limit is not None and _as_int(limit, -1) >= 0:
        test_steps = _as_int(limit, 0)
    else:
        test_rows = _TEST_ROWS.get(data_name, 0)
        test_steps = int(math.ceil(test_rows / test_batch)) if test_rows else 0
    seconds = float(STARTUP_SECONDS)
    if mode != 'eval':
        seconds += steps * _ms_per_step(model_name, batch) / 1000.0
    seconds += passes * test_steps * _ms_per_step(model_name, test_batch) / 1000.0
    return max(1, int(math.ceil(seconds)))


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
            job['device'] = str(system.get('device') or 'cpu').lower()
        job['estimate_device'] = str(job.get('device') or 'cuda').lower()
    return jobs


def _job_device(job: dict[str, Any]) -> str:
    return str(job.get('device') or job.get('estimate_device') or 'cuda').lower()


def requires_gpu(jobs: list[dict[str, Any]]) -> bool:
    return any(_job_device(job) != 'cpu' for job in jobs)


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
    devices = {_job_device(job) for job in jobs}
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
    return f'{_job_device(job)}:{name or "job"}'


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
    devices = {_job_device(job) for job in jobs}
    cpu_only = bool(devices) and all(dev == 'cpu' for dev in devices)
    if cpu_only:
        for job in jobs:
            job.pop('gpu', None)
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
        'devices': sorted({_job_device(job) for job in jobs}),
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
    devices = set(report.get('devices') or [])
    cpu_only = bool(devices) and devices == {'cpu'}
    gpu_txt = 'CPU' if cpu_only else 'no-gpu'
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
    if cpu_only:
        return 'round={0} ({1}) for {2} CPU jobs | CPU'.format(
            report.get('round'), report.get('round_source'), report.get('n_jobs')
        )
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
