"""Shared pytest fixtures, markers, and per-test result persistence.

Implements the project test spec (研讨纪要 测试规范):
- tags: unit|integration|e2e × location|content|physical × p1|p2|p3 × architecture layers
- persist each result to disk (JSONL) for later Markdown reporting
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch

# Avoid OpenMP duplicate-runtime abort on Windows Anaconda+torch.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / 'output' / 'test_results'
RESULTS_JSONL = RESULTS_DIR / 'results.jsonl'


def pytest_configure(config):
    markers = [
        ('unit', 'Unit test'),
        ('integration', 'Integration test (call-subgraph)'),
        ('e2e', 'End-to-end flow'),
        ('location', 'Location / structure test'),
        ('content', 'Content / functional test'),
        ('physical', 'Physical resource test (time/memory)'),
        ('p1', 'Priority 1 (highest)'),
        ('p2', 'Priority 2'),
        ('p3', 'Priority 3'),
        ('data_layer', 'data architecture layer'),
        ('model_layer', 'model architecture layer'),
        ('algorithm_layer', 'algorithm / metrics layer'),
        ('system_layer', 'system / trainer backend layer'),
        ('config_layer', 'config layer'),
        ('schema_layer', 'schema layer'),
        ('plugins_layer', 'plugins API layer'),
        ('application_layer', 'experiments / CLI application layer'),
        ('runtime', 'runtime timing physical trait'),
        ('memory', 'memory physical trait'),
        ('slow', 'slow test'),
        ('external', 'needs optional third-party package or network'),
        ('gpu', 'prefers or requires CUDA when available'),
        ('module_registry', 'module: registry'),
        ('module_provider', 'module: provider'),
        ('module_trainer', 'module: trainer'),
        ('feature_smoke', 'feature: smoke train/eval'),
        ('feature_providers', 'feature: provider registry'),
    ]
    for name, desc in markers:
        config.addinivalue_line('markers', f'{name}: {desc}')

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # fresh run file
    if RESULTS_JSONL.exists():
        RESULTS_JSONL.unlink()
    RESULTS_JSONL.touch()
    meta = {
        'schema': 'rpipe.test_run.v1',
        'started_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'cuda_available': torch.cuda.is_available(),
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'torch_version': torch.__version__,
    }
    (RESULTS_DIR / 'run_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')


@pytest.fixture(scope='session')
def device() -> torch.device:
    """Prefer CUDA when available (spec allows GPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


@pytest.fixture(scope='session')
def cuda_available() -> bool:
    return torch.cuda.is_available()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != 'call':
        return
    markers = sorted({m.name for m in item.iter_markers()})
    record = {
        'nodeid': item.nodeid,
        'name': item.name,
        'path': str(item.fspath),
        'outcome': report.outcome,
        'duration_s': getattr(report, 'duration', None),
        'markers': markers,
        'level': _first(markers, ('unit', 'integration', 'e2e')),
        'type': _first(markers, ('location', 'content', 'physical')),
        'priority': _first(markers, ('p1', 'p2', 'p3')),
        'layers': [m for m in markers if m.endswith('_layer')],
        'modules': [m for m in markers if m.startswith('module_')],
        'features': [m for m in markers if m.startswith('feature_')],
        'longrepr': str(report.longrepr) if report.failed else None,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_JSONL.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def _first(markers: set[str] | list[str], candidates: tuple[str, ...]) -> str | None:
    s = set(markers)
    for c in candidates:
        if c in s:
            return c
    return None
