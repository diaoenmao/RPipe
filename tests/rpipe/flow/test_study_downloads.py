"""Manual Study raw preparation obeys domestic routing without network access."""

import hashlib
import importlib.util
import io
import json
import socket
import urllib.request
import urllib.response
from email.message import Message
from pathlib import Path

import pytest

from rpipe.structure.origin import domestic_endpoint

pytestmark = [pytest.mark.unit, pytest.mark.content, pytest.mark.p1,
              pytest.mark.flow_layer, pytest.mark.module_data,
              pytest.mark.cost(cost_class='c1'),
              pytest.mark.result_type('categorical', detail='summary')]


@pytest.fixture
def raw_prepare(monkeypatch):
    source = Path(__file__).resolve().parents[3] / 'studies/main_exp/flow/prepare_data.py'
    spec = importlib.util.spec_from_file_location('_raw_prepare_test', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(urllib.request, '_opener', None)
    def no_socket(*args, **kwargs):
        raise AssertionError('test must not make a real network connection')
    monkeypatch.setattr(socket.socket, 'connect', no_socket)
    return module


def _response(request, body=b'raw fixture', *, redirect=None, length=None):
    headers = Message()
    if redirect is not None:
        headers['Location'] = redirect
    else:
        headers['Content-Length'] = str(len(body) if length is None else length)
    response = urllib.response.addinfourl(io.BytesIO(body), headers, request.full_url,
                                         302 if redirect is not None else 200)
    response.msg = 'Found' if redirect is not None else 'OK'
    return response


def _transport(monkeypatch, receive):
    # Keep urllib's actual redirect machinery; replace only network transport.
    monkeypatch.setattr(urllib.request.HTTPSHandler, 'https_open', receive)
    monkeypatch.setattr(urllib.request.HTTPHandler, 'http_open', receive)
    monkeypatch.setattr(urllib.request.FTPHandler, 'ftp_open', receive)


@pytest.mark.parametrize('dataset', ['MNIST', 'CIFAR10'])
def test_manual_raw_preparation_uses_registered_domestic_source(tmp_path, monkeypatch, raw_prepare, dataset):
    seen = []
    refusal = OSError('transport deliberately stopped')
    def receive(self, request):
        seen.append(request.full_url)
        raise refusal
    _transport(monkeypatch, receive)
    with pytest.raises(OSError) as caught:
        if dataset == 'MNIST':
            raw_prepare.prepare_mnist(tmp_path / 'data', {})
        else:
            raw_prepare.prepare_cifar(tmp_path / 'data', {'CIFAR10/raw/cifar-10-batches-py/data_batch_1': '0' * 64})
    assert caught.value is refusal
    endpoint = domestic_endpoint(dataset)
    expected = endpoint.mirrors[0] + raw_prepare.MNIST_RESOURCES[0][0] if dataset == 'MNIST' else endpoint.url
    assert seen == [expected]


@pytest.mark.parametrize('url', [
    'https://huggingface.co/raw.gz',
    'http://dataset.bj.bcebos.com/raw.gz',
    'https://dataset.bj.bcebos.com.evil.invalid/raw.gz',
    'https://dataset.bj.bcebos.com@outside.invalid/raw.gz',
    'https://outside.invalid@dataset.bj.bcebos.com/raw.gz',
    'https://dataset.bj.bcebos.com:444/raw.gz',
])
def test_manual_download_refuses_foreign_authority_before_transport(tmp_path, monkeypatch, raw_prepare, url):
    seen = []
    def receive(self, request):
        seen.append(request.full_url)
        return _response(request)
    _transport(monkeypatch, receive)
    target = tmp_path / 'raw.gz'
    partial = target.with_name(target.name + '.part')
    partial.write_bytes(b'previous failed download')
    with pytest.raises(ValueError, match='domestic'):
        raw_prepare.download(url, target, hashlib.md5(b'raw fixture').hexdigest())
    assert not seen and not target.exists()
    assert partial.read_bytes() == b'previous failed download'


@pytest.mark.parametrize('redirect', ['https://huggingface.co/raw.gz', 'http://outside.invalid/raw.gz', 'ftp://outside.invalid/raw.gz'])
def test_manual_download_refuses_foreign_redirect_before_transport(tmp_path, monkeypatch, raw_prepare, redirect):
    initial = domestic_endpoint('MNIST').mirrors[0] + 'raw.gz'
    seen = []
    def receive(self, request):
        seen.append(request.full_url)
        return _response(request, redirect=redirect) if len(seen) == 1 else _response(request)
    _transport(monkeypatch, receive)
    target = tmp_path / 'raw.gz'
    partial = target.with_name(target.name + '.part')
    partial.write_bytes(b'old partial evidence')
    with pytest.raises(ValueError, match='domestic'):
        raw_prepare.download(initial, target, hashlib.md5(b'raw fixture').hexdigest())
    assert seen == [initial]
    assert not target.exists() and partial.read_bytes() == b'old partial evidence'


@pytest.mark.parametrize('authority', ['dataset.bj.bcebos.com', 'DATASET.BJ.BCEBOS.COM', 'dataset.bj.bcebos.com:443'])
def test_manual_domestic_download_preserves_hash_and_manifest(tmp_path, monkeypatch, raw_prepare, authority):
    url = f'https://{authority}/raw.gz'
    seen = []
    def receive(self, request):
        seen.append(request.full_url)
        return _response(request)
    _transport(monkeypatch, receive)
    target = tmp_path / 'raw.gz'
    md5 = hashlib.md5(b'raw fixture').hexdigest()
    row = raw_prepare.download(url, target, md5)
    assert row == {'url': url, 'actual_url': url, 'source': 'download', 'md5': md5}
    assert seen == [url] and target.read_bytes() == b'raw fixture'
    assert not target.with_name(target.name + '.part').exists()


def test_manual_download_allows_same_domestic_redirect(tmp_path, monkeypatch, raw_prepare):
    initial = domestic_endpoint('MNIST').mirrors[0] + 'raw.gz'
    final = domestic_endpoint('MNIST').mirrors[0] + 'final.gz'
    seen = []
    def receive(self, request):
        seen.append(request.full_url)
        return _response(request, redirect=final) if len(seen) == 1 else _response(request)
    _transport(monkeypatch, receive)
    target = tmp_path / 'raw.gz'
    row = raw_prepare.download(initial, target, hashlib.md5(b'raw fixture').hexdigest())
    assert seen == [initial, final] and row['url'] == initial and row['actual_url'] == final
    assert target.read_bytes() == b'raw fixture'


@pytest.mark.parametrize('failure', ['length', 'checksum'])
def test_manual_domestic_download_retains_bad_partial_evidence(tmp_path, monkeypatch, raw_prepare, failure):
    def receive(self, request):
        return _response(request, length=100 if failure == 'length' else None)
    _transport(monkeypatch, receive)
    target = tmp_path / 'raw.gz'
    peer = tmp_path / 'peer.bin'
    peer.write_bytes(b'peer')
    md5 = '0' * 32 if failure == 'checksum' else hashlib.md5(b'raw fixture').hexdigest()
    with pytest.raises(ValueError, match='incomplete download|checksum mismatch'):
        raw_prepare.download(domestic_endpoint('MNIST').mirrors[0] + 'raw.gz', target, md5)
    assert not target.exists() and target.with_name(target.name + '.part').read_bytes() == b'raw fixture'
    assert peer.read_bytes() == b'peer'


def test_manual_cached_archive_with_historical_url_needs_no_network(tmp_path, monkeypatch, raw_prepare):
    target = tmp_path / 'raw.gz'
    target.write_bytes(b'raw fixture')
    def receive(self, request):
        raise AssertionError('cached bytes must not download')
    _transport(monkeypatch, receive)
    md5 = hashlib.md5(b'raw fixture').hexdigest()
    row = raw_prepare.download('https://old-foreign.invalid/raw.gz', target, md5)
    assert row == {'url': 'https://old-foreign.invalid/raw.gz', 'source': 'existing archive', 'md5': md5}
    assert target.read_bytes() == b'raw fixture'


def test_manual_existing_bad_archive_is_preserved(tmp_path, monkeypatch, raw_prepare):
    target = tmp_path / 'raw.gz'
    target.write_bytes(b'old invalid archive')
    def receive(self, request):
        raise AssertionError('existing bad archive must not be overwritten')
    _transport(monkeypatch, receive)
    with pytest.raises(ValueError, match='existing archive checksum mismatch'):
        raw_prepare.download(domestic_endpoint('MNIST').mirrors[0] + 'raw.gz', target, '0' * 32)
    assert target.read_bytes() == b'old invalid archive'


def test_manual_verify_only_retains_original_download_provenance(tmp_path, monkeypatch, raw_prepare):
    raw = tmp_path / 'data/MNIST/raw/fixture'
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b'verified bytes')
    expected = tmp_path / 'expected.json'
    expected.write_text(json.dumps({'data/MNIST/raw/fixture': hashlib.sha256(raw.read_bytes()).hexdigest()}), encoding='utf-8')
    manifest = tmp_path / 'manifest.json'
    downloads = [{'url': 'https://old-foreign.invalid/raw.gz', 'source': 'download'}]
    manifest.write_text(json.dumps({'downloads': downloads, 'reused_caches': {'old': 'cache'}}), encoding='utf-8')
    def forbidden(*args, **kwargs):
        raise AssertionError('verify-only must not prepare or download')
    monkeypatch.setattr(raw_prepare, 'prepare_mnist', forbidden)
    monkeypatch.setattr(raw_prepare, 'prepare_cifar', forbidden)
    monkeypatch.setattr(raw_prepare, 'split_counts', lambda root: {'fixture': 1})
    before = raw.read_bytes()
    assert raw_prepare.main(['--data-root', str(raw.parents[2]), '--manifest', str(manifest),
                             '--expected-manifest', str(expected), '--verify-only']) == 0
    report = json.loads(manifest.read_text(encoding='utf-8'))
    assert report['downloads'] == downloads and report['reused_caches'] == {'old': 'cache'}
    assert report['historical_sha256_matches'] == 1 and raw.read_bytes() == before


@pytest.mark.parametrize('kind,cleanup', [
    ('download', 'destination'), ('download', 'source'),
    ('mnist', 'destination'), ('mnist', 'source'),
    ('cifar', 'destination'), ('cifar', 'source'), ('cifar', 'archive'),
])
@pytest.mark.parametrize('outcome', ['error', 'cancel', 'normal-close', 'programming-close', 'ok'])
def test_manual_raw_copy_preserves_primary_across_owned_close(tmp_path, monkeypatch, stream_close_fault,
                                                             raw_prepare, kind, cleanup, outcome):
    import gzip
    import tarfile

    payload = b'synthetic raw evidence'
    primary = (KeyboardInterrupt('original cancellation') if outcome == 'cancel'
               else OSError('original partial write failure') if outcome in ('error', 'programming-close') else None)
    secondary = (RuntimeError('unexpected close implementation bug') if outcome == 'programming-close'
                 else OSError('secondary close failure') if outcome != 'ok' else None)
    sources = []
    close_calls = []
    class Source:
        def __init__(self, raw, owner):
            self.raw, self.owner = raw, owner
            sources.append(raw)
        def __getattr__(self, name):
            return getattr(self.raw, name)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()
        def close(self):
            close_calls.append(self.owner)
            self.raw.close()
            if cleanup == self.owner and secondary is not None:
                raise secondary

    peer = tmp_path / 'peer.bin'
    peer.write_bytes(b'peer evidence')
    if kind == 'download':
        target = tmp_path / 'raw.gz'
        def receive(self, request):
            return Source(_response(request, payload), 'source')
        _transport(monkeypatch, receive)
        invoke = lambda: raw_prepare.download(domestic_endpoint('MNIST').mirrors[0] + 'raw.gz', target,
                                               hashlib.md5(payload).hexdigest())
    elif kind == 'mnist':
        archive = tmp_path / 'MNIST/raw/fixture.gz'
        archive.parent.mkdir(parents=True)
        archive.write_bytes(gzip.compress(payload))
        monkeypatch.setattr(raw_prepare, 'MNIST_RESOURCES', (('fixture.gz', hashlib.md5(archive.read_bytes()).hexdigest()),))
        target = archive.with_suffix('')
        native = gzip.open
        monkeypatch.setattr(gzip, 'open', lambda *args, **kwargs: Source(native(*args, **kwargs), 'source'))
        invoke = lambda: raw_prepare.prepare_mnist(tmp_path, {'MNIST/raw/fixture': hashlib.sha256(payload).hexdigest()})
    else:
        archive = tmp_path / 'CIFAR10/raw/cifar-10-python.tar.gz'
        archive.parent.mkdir(parents=True)
        with tarfile.open(archive, 'w:gz') as handle:
            member = tarfile.TarInfo('cifar-10-batches-py/data_batch_1')
            member.size = len(payload)
            handle.addfile(member, io.BytesIO(payload))
        monkeypatch.setattr(raw_prepare, 'CIFAR_MD5', hashlib.md5(archive.read_bytes()).hexdigest())
        target = archive.parent / 'cifar-10-batches-py/data_batch_1'
        native_open, native_extract = tarfile.open, tarfile.TarFile.extractfile
        monkeypatch.setattr(tarfile, 'open', lambda *args, **kwargs: Source(native_open(*args, **kwargs), 'archive'))
        monkeypatch.setattr(tarfile.TarFile, 'extractfile', lambda obj, *args, **kwargs: Source(native_extract(obj, *args, **kwargs), 'source'))
        invoke = lambda: raw_prepare.prepare_cifar(tmp_path, {'CIFAR10/raw/cifar-10-batches-py/data_batch_1': hashlib.sha256(payload).hexdigest()})
    before = {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    handles, calls = stream_close_fault(lambda path, mode: path.name.endswith('.part') and mode == 'wb',
                                        primary, secondary if cleanup == 'destination' else None)
    expected = primary if outcome in ('error', 'cancel') else secondary
    if expected is None:
        invoke()
    else:
        with pytest.raises(type(expected)) as caught:
            invoke()
        assert caught.value is expected
    assert len(handles) == 1 and all(handle.closed for handle in handles)
    assert calls.count('close') == 1
    assert sources and all(source.closed for source in sources)
    assert close_calls.count('source') == 1
    if kind == 'cifar':
        assert close_calls.count('archive') == 1
    partial = target.with_name(target.name + '.part')
    if outcome == 'ok' or (cleanup == 'archive' and outcome == 'normal-close'):
        assert target.read_bytes() == payload and not partial.exists()
    else:
        assert not target.exists()
        assert partial.read_bytes() == (payload[:3] if primary is not None else payload)
    assert {path: path.read_bytes() for path in before} == before
