"""Unit tests for rpipe.provider registry + layer folders."""

from __future__ import annotations

import pytest

from rpipe.provider import get_algorithm, get_provider, list_algorithms, list_providers, load_builtin_providers
from rpipe.provider.api import ALGORITHM_PROVIDERS, PROVIDERS


@pytest.mark.unit
@pytest.mark.content
@pytest.mark.p1
@pytest.mark.plugins_layer
@pytest.mark.feature_providers
@pytest.mark.module_provider
def test_layer_folders_registered():
    providers = list_providers()
    assert set(providers['data']) == {'native', 'datasets'}
    assert 'gguf' in providers['model']
    assert set(providers['system']) == {'pytorch', 'ggml'}
    assert set(providers['algorithm']['train']) == {'accelerate', 'native'}
    assert 'lm_eval' in providers['algorithm']['metric']
    assert set(providers['algorithm']['generate']) >= {'llama_cpp', 'diffusers'}


@pytest.mark.unit
@pytest.mark.content
@pytest.mark.p1
@pytest.mark.plugins_layer
def test_algorithm_types_and_bindings_metadata():
    load_builtin_providers()
    assert get_algorithm('train', 'accelerate', require_available=False).requires_system == 'pytorch'
    assert get_algorithm('generate', 'llama_cpp', require_available=False).requires_system == 'ggml'
    assert get_algorithm('generate', 'llama_cpp', require_available=False).requires_model == 'gguf'
    assert get_provider('model', 'gguf').binds_generate == 'llama_cpp'
    assert get_algorithm('metric', 'lm_eval', require_available=False).mode == 'benchmark'


@pytest.mark.unit
@pytest.mark.content
@pytest.mark.p1
@pytest.mark.system_layer
def test_system_tensor_libs():
    assert get_provider('system', 'pytorch').tensor_lib == 'pytorch'
    assert get_provider('system', 'ggml', require_available=False).tensor_lib == 'ggml'


@pytest.mark.unit
@pytest.mark.content
@pytest.mark.p1
@pytest.mark.plugins_layer
def test_native_available():
    assert get_provider('data', 'native').available()
    assert get_provider('model', 'native').available()
    assert get_algorithm('train', 'native').available()
    assert get_algorithm('metric', 'native').available()
    assert get_provider('system', 'pytorch').available()
