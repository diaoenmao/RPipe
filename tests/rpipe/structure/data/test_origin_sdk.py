"""Actual SDK transfer policy; repository metadata and transport are synthetic."""
import hashlib
import socket
import sys
from types import SimpleNamespace

import pytest

from rpipe.structure.origin import download_model

pytestmark = [
    pytest.mark.integration, pytest.mark.content, pytest.mark.p1,
    pytest.mark.structure_layer, pytest.mark.module_data,
    pytest.mark.cost(cost_class='c1'),
    pytest.mark.result_type('categorical', detail='summary'),
]


@pytest.fixture
def sdk_download(monkeypatch, tmp_path):
    httpx = pytest.importorskip('httpx')
    pytest.importorskip('huggingface_hub')
    from huggingface_hub import constants
    from huggingface_hub.hf_api import HfApi, RepoFile
    from huggingface_hub.utils import _http

    payload = b'{"synthetic":true}\n'
    commit = 'a' * 40
    oid = hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).hexdigest()
    calls = []
    target = ['https://hf-mirror.com/synthetic/config.json']
    failure = [None]
    extra_headers = {}

    def transport(request):
        calls.append(str(request.url))
        if failure[0] is not None:
            raise failure[0]
        headers = {'x-repo-commit': commit, 'etag': oid,
                   'content-length': str(len(payload)), **extra_headers}
        if '/resolve/' in request.url.path:
            return httpx.Response(302, headers={**headers, 'location': target[0],
                                  'x-linked-size': str(len(payload))}, request=request)
        return httpx.Response(200, headers=headers, content=payload, request=request)

    def unavailable(**kwargs):
        raise OSError('synthetic ModelScope unavailable')

    def no_network(*args, **kwargs):
        raise AssertionError('real network prohibited')

    monkeypatch.setattr(socket.socket, 'connect', no_network)
    monkeypatch.setattr(constants, 'HF_HUB_DISABLE_XET', True)
    monkeypatch.setattr(constants, 'HF_HUB_DISABLE_IMPLICIT_TOKEN', True)
    monkeypatch.setitem(sys.modules, 'modelscope', SimpleNamespace(snapshot_download=unavailable))
    monkeypatch.setattr(HfApi, 'list_repo_tree', lambda *a, **k: [
        RepoFile(path='config.json', size=len(payload), oid=oid)])
    with httpx.Client(transport=httpx.MockTransport(transport), follow_redirects=True) as client:
        monkeypatch.setattr(_http, '_GLOBAL_CLIENT', client)

        def run():
            return download_model('rpipe-audit/synthetic-model', tmp_path, revision=commit)

        yield SimpleNamespace(run=run, target=target, calls=calls, client=client,
                              payload=payload, httpx=httpx, failure=failure, extra_headers=extra_headers)


@pytest.mark.parametrize('target', [
    'https://huggingface.co/synthetic/config.json',
    'https://cdn-lfs.huggingface.co/synthetic/config.json',
    'http://hf-mirror.com/synthetic/config.json',
    'https://hf-mirror.com.evil.invalid/synthetic/config.json',
])
def test_model_sdk_rejects_redirect_before_unapproved_transport(sdk_download, target):
    sdk_download.target[0] = target
    with pytest.raises(RuntimeError, match='domestic|overseas'):
        sdk_download.run()
    assert sdk_download.calls
    assert all(sdk_download.httpx.URL(url).host == 'hf-mirror.com'
               and sdk_download.httpx.URL(url).scheme == 'https'
               for url in sdk_download.calls)
    assert not sdk_download.client.is_closed


def test_model_sdk_domestic_redirect_and_cached_snapshot(sdk_download):
    from pathlib import Path
    result = Path(sdk_download.run())
    assert (result / 'config.json').read_bytes() == sdk_download.payload
    sdk_download.calls.clear()
    assert Path(sdk_download.run()) == result
    assert sdk_download.calls == []
    assert not sdk_download.client.is_closed


def test_model_sdk_preserves_shared_client_for_foreign_consumer(sdk_download):
    sdk_download.run()
    response = sdk_download.client.get('https://example.org/unrelated.json')
    assert response.content == sdk_download.payload
    assert not sdk_download.client.is_closed


def test_model_sdk_cancellation_preserves_client_and_original_exception(sdk_download):
    error = KeyboardInterrupt('synthetic transport cancellation')
    hooks = {name: list(values) for name, values in sdk_download.client.event_hooks.items()}
    sdk_download.failure[0] = error
    with pytest.raises(KeyboardInterrupt) as caught:
        sdk_download.run()
    assert caught.value is error
    assert sdk_download.client.event_hooks == hooks
    assert not sdk_download.client.is_closed
    sdk_download.failure[0] = None
    assert sdk_download.client.get('https://example.org/unrelated.json').status_code == 200


def test_model_sdk_xet_metadata_uses_guarded_http_without_native_transfer(sdk_download, monkeypatch):
    from pathlib import Path
    from huggingface_hub import file_download
    native = []
    monkeypatch.setattr(file_download, 'is_xet_available', lambda: True)

    def unguarded_native(**kwargs):
        native.append(kwargs)
        raise AssertionError('native transfer has no domestic request guard')

    monkeypatch.setattr(file_download, 'xet_get', unguarded_native)
    sdk_download.extra_headers.update({
        'x-xet-hash': 'b' * 64,
        'x-xet-refresh-route': 'https://hf-mirror.com/api/xet/read-token',
    })
    result = Path(sdk_download.run())
    assert (result / 'config.json').read_bytes() == sdk_download.payload
    assert native == []
    assert file_download.is_xet_available() is True
