"""GPU discovery and hardware descriptions; no packing or reporting."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    total_bytes: int
    free_bytes: int


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
