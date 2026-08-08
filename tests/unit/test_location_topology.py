"""Location tests: layer/provider folder topology."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / 'src' / 'rpipe'

REQUIRED = [
    'data/native',
    'data/datasets',
    'model/native',
    'model/gguf',
    'algorithm/train/native',
    'algorithm/train/accelerate',
    'algorithm/metric/native',
    'algorithm/metric/lm_eval',
    'algorithm/generate/llama_cpp',
    'algorithm/generate/diffusers',
    'system/pytorch',
    'system/ggml',
    'provider',
]


@pytest.mark.unit
@pytest.mark.location
@pytest.mark.p1
@pytest.mark.plugins_layer
def test_layer_provider_folders_exist():
    assert not (SRC / 'plugins').exists(), 'plugins/ dump should be removed'
    for rel in REQUIRED:
        path = SRC / rel
        assert path.is_dir(), f'missing {rel}'
        assert (path / '__init__.py').is_file(), f'missing {rel}/__init__.py'


@pytest.mark.unit
@pytest.mark.location
@pytest.mark.p1
@pytest.mark.config_layer
def test_tests_mirror_unit_dirs_exist():
    unit = REPO / 'tests' / 'unit'
    for name in ('config', 'schema', 'provider', 'data', 'model', 'algorithm', 'system'):
        assert (unit / name).is_dir()
