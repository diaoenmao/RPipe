import hashlib
import os
from pathlib import Path

import pytest

from rpipe.structure.origin import (
    apply_model_origin,
    bind_endpoint,
    discard_bad_archive,
    domestic_endpoint,
    endpoint_for,
    foreign_endpoint,
    model_hub,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.content,
    pytest.mark.p1,
    pytest.mark.structure_layer,
    pytest.mark.module_data,
]


def test_study_origin_selects_model_hub(monkeypatch):
    monkeypatch.setenv('HF_ENDPOINT', 'https://example.invalid')
    assert model_hub('foreign') == 'https://huggingface.co'
    assert model_hub('domestic') == 'https://hf-mirror.com'
    assert apply_model_origin('domestic') == 'https://hf-mirror.com'
    assert os.environ['HF_ENDPOINT'] == 'https://hf-mirror.com'


def test_two_endpoints_use_different_cifar_hosts():
    foreign = foreign_endpoint('CIFAR10')
    domestic = domestic_endpoint('CIFAR10')
    assert 'toronto.edu' in (foreign.url or '')
    assert 'bcebos.com' in (domestic.url or '')
    assert foreign.archive_md5 == domestic.archive_md5
    assert endpoint_for('MNIST', None).origin == 'foreign'
    assert endpoint_for('MNIST', 'domestic').mirrors[0].endswith('/mnist/')


def test_domestic_rejects_datasets_without_a_mirror():
    with pytest.raises(ValueError, match='domestic'):
        domestic_endpoint('SVHN')
    with pytest.raises(ValueError, match='origin'):
        endpoint_for('CIFAR10', 'vpn')


def test_bind_endpoint_restores_class_url():
    class _Set:
        url = 'https://foreign.example/cifar.tar.gz'
        mirrors = ['https://foreign.example/mnist/']

    domestic = domestic_endpoint('CIFAR10')
    with bind_endpoint(_Set, domestic):
        assert _Set.url == domestic.url
    assert _Set.url == 'https://foreign.example/cifar.tar.gz'

    mnist = domestic_endpoint('MNIST')
    with bind_endpoint(_Set, mnist):
        assert _Set.mirrors == [mnist.mirrors[0]]
    assert _Set.mirrors == ['https://foreign.example/mnist/']


def test_discard_bad_archive_removes_partial_and_keeps_matching_md5(tmp_path: Path):
    endpoint = domestic_endpoint('CIFAR10')
    partial = tmp_path / endpoint.archive
    partial.write_bytes(b'partial')
    discard_bad_archive(tmp_path, endpoint)
    assert not partial.is_file()

    good = tmp_path / endpoint.archive
    payload = b'cifar-bytes'
    good.write_bytes(payload)
    endpoint = endpoint.__class__(
        endpoint.name,
        endpoint.origin,
        url=endpoint.url,
        archive=endpoint.archive,
        archive_md5=hashlib.md5(payload).hexdigest(),
    )
    discard_bad_archive(tmp_path, endpoint)
    assert good.read_bytes() == payload
