"""Config-only job estimates; no runtime hardware discovery."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from rpipe.structure.api.artifact_api import load_config
from rpipe.structure.control.mapping import get_setting

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
_TRAIN_ROWS = {'MNIST': 60000, 'FashionMNIST': 60000, 'CIFAR10': 50000, 'CIFAR100': 50000, 'SVHN': 73257}
# act multiplier, param MiB, ms/step at REF_BATCH
_MODEL = {
    'linear': (80, 32, 6),
    'mlp': (200, 64, 6),
    'cnn': (600, 128, 10),
    'resnet10': (1200, 256, 14),
    'resnet18': (1800, 512, 18),
    'wresnet': (2800, 1024, 28),
}


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def estimate_job_bytes(cfg: dict[str, Any] | None) -> int:
    """Peak VRAM guess for one ``run-one`` process."""
    if not isinstance(cfg, dict):
        return OVERHEAD_BYTES + _MODEL['resnet18'][1] * 1024 * 1024
    system = cfg.get('system') if isinstance(cfg.get('system'), dict) else {}
    measured = get_setting(system, 'vram_bytes')
    if measured is not None:
        if type(measured) is not int or measured < OVERHEAD_BYTES:
            raise ValueError(f'system.vram_bytes must be an integer >= {OVERHEAD_BYTES}')
        return measured
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
    test_batch = max(int(batch * float(dcfg.get('test_batch_ratio', 1) or 1)), 1)
    test_activations = test_batch * spatial * 4 * act
    evaluation = OVERHEAD_BYTES + int((params + test_activations) * 0.4)
    if str(algo.get('mode') or 'train') == 'eval':
        return evaluation
    return max(OVERHEAD_BYTES + params + activations, evaluation)


def _ms_per_step(model_name: str, batch: int) -> float:
    ms = _MODEL.get(model_name, _MODEL['resnet18'])[2]
    return float(ms) * (max(int(batch), 1) / REF_BATCH)


def _scheduled_test_passes(algo: dict[str, Any], units: int) -> int:
    """How many full test passes this train run makes. ``eval_period <= 0`` is once at the end."""
    raw = get_setting(algo, 'eval_period')
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
    epochs = _as_int(get_setting(algo, 'num_epochs'), 0)
    steps = _as_int(get_setting(algo, 'num_steps'), 0)
    period = max(1, _as_int(get_setting(algo, 'step_period'), 1))
    if epochs > 0:
        rows = _as_int(dcfg.get('train_size'), 0)
        if rows <= 0:
            rows = _TRAIN_ROWS.get(data_name, 50000)
        batches_per_epoch = max(1, int(math.ceil(rows / batch)))
        steps = epochs * int(math.ceil(batches_per_epoch / period))
        train_batches = epochs * batches_per_epoch
    else:
        steps = max(steps, 1)
        train_batches = steps * period
    unit = str(get_setting(algo, 'progress_unit') or 'step').lower()
    if unit != 'epoch':
        train_batches = steps * period
    units = epochs if unit == 'epoch' and epochs > 0 else steps
    mode = str(algo.get('mode') or 'train')
    passes = 1 if mode == 'eval' else _scheduled_test_passes(algo, units)
    limit = get_setting(algo, 'eval_num_steps')
    if limit is not None and _as_int(limit, -1) >= 0:
        test_steps = _as_int(limit, 0)
    else:
        test_rows = _TEST_ROWS.get(data_name, 0)
        test_steps = int(math.ceil(test_rows / test_batch)) if test_rows else 0
    seconds = float(STARTUP_SECONDS)
    if mode != 'eval':
        seconds += train_batches * _ms_per_step(model_name, batch) / 1000.0
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
            job['device'] = str(get_setting(system, 'device') or 'cpu').lower()
        job['estimate_device'] = str(job.get('device') or 'cuda').lower()
    return jobs
