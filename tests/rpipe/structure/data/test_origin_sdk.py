"""Actual SDK transfer policy; repository metadata and transport are synthetic."""
import hashlib
import io
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
    on_request = [None]

    def transport(request):
        calls.append(str(request.url))
        if on_request[0] is not None:
            on_request[0](request)
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
    if hasattr(_http, '_GLOBAL_CLIENT'):
        client = httpx.Client(transport=httpx.MockTransport(transport), follow_redirects=True)
        monkeypatch.setattr(_http, '_GLOBAL_CLIENT', client)
    else:
        import requests

        class Adapter(requests.adapters.BaseAdapter):
            def send(self, request, **kwargs):
                fake = transport(httpx.Request(request.method, request.url))
                response = requests.Response()
                response.status_code = fake.status_code
                response.headers.update(dict(fake.headers))
                response.raw = io.BytesIO(fake.content)
                response._content = fake.content
                response.url = request.url
                response.request = request
                return response

            def close(self):
                pass

        class Session(requests.Session):
            is_closed = False

            def close(self):
                self.is_closed = True
                super().close()

        client = Session()
        client.mount('https://', Adapter())
        client.mount('http://', Adapter())
        client.event_hooks = client.hooks
        monkeypatch.setattr(_http, '_get_session_from_cache', lambda **kwargs: client)
        monkeypatch.setattr(HfApi, 'repo_info', lambda *a, **k: SimpleNamespace(
            sha=commit, siblings=[SimpleNamespace(rfilename='config.json')]))
    with client:

        def run():
            return download_model('rpipe-audit/synthetic-model', tmp_path, revision=commit)

        yield SimpleNamespace(run=run, target=target, calls=calls, client=client,
                              payload=payload, httpx=httpx, failure=failure, extra_headers=extra_headers, on_request=on_request)


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


def test_model_sdk_foreign_thread_keeps_client_and_native_configuration(sdk_download, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from huggingface_hub import file_download
    entered, release = Event(), Event()
    monkeypatch.setattr(file_download, 'is_xet_available', lambda: True)
    original_constants = file_download.constants
    hooks = {name: list(values) for name, values in sdk_download.client.event_hooks.items()}

    def pause(request):
        if request.url.host == 'hf-mirror.com' and request.method == 'HEAD':
            entered.set()
            assert release.wait(5), 'test failed to release the synthetic model transfer'

    sdk_download.on_request[0] = pause
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(sdk_download.run)
        try:
            assert entered.wait(5), 'model never entered synthetic transport'
            assert file_download.is_xet_available() is True
            response = sdk_download.client.get('https://example.org/concurrent.json')
            assert response.content == sdk_download.payload
            assert sdk_download.client.event_hooks == hooks
        finally:
            release.set()
        assert future.result(timeout=5)
    assert file_download.constants is original_constants
    assert file_download.is_xet_available() is True
    assert sdk_download.client.event_hooks == hooks
    assert not sdk_download.client.is_closed


@pytest.mark.parametrize('target,allowed', [
    ('https://modelscope.cn/api/synthetic', True),
    ('https://cdn.modelscope.cn/synthetic', True),
    ('https://modelhub.oss-cn-hangzhou.aliyuncs.com/synthetic', True),
    ('https://modelscope.ai/synthetic', False),
    ('https://modelhub.oss-us-west-1.aliyuncs.com/synthetic', False),
    ('https://modelscope.cn.evil.invalid/synthetic', False),
    ('https://user:password@modelscope.cn/synthetic', False),
    ('http://modelscope.cn/synthetic', False),
])
def test_modelscope_requests_policy_precedes_transfer(monkeypatch, tmp_path, target, allowed):
    requests = pytest.importorskip('requests')
    calls = []

    def transport(self, request, **kwargs):
        calls.append(request.url)
        response = requests.Response()
        response.status_code = 200
        response._content = b'synthetic'
        response.url = request.url
        response.request = request
        return response

    def modelscope(**kwargs):
        requests.get(target).raise_for_status()
        return str(tmp_path / 'modelscope-snapshot')

    def no_mirror(**kwargs):
        raise OSError('synthetic mirror unavailable')

    monkeypatch.setattr(requests.adapters.HTTPAdapter, 'send', transport)
    monkeypatch.setattr(socket.socket, 'connect', lambda *a, **k: pytest.fail('real network'))
    monkeypatch.setitem(sys.modules, 'modelscope', SimpleNamespace(snapshot_download=modelscope))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=no_mirror))
    if allowed:
        assert download_model('synthetic/model', tmp_path) == str(tmp_path / 'modelscope-snapshot')
        assert calls == [target]
    else:
        with pytest.raises(RuntimeError, match='domestic'):
            download_model('synthetic/model', tmp_path)
        assert calls == []


@pytest.mark.parametrize('streaming', [False, True])
def test_dataset_helper_guards_eager_and_deferred_requests(monkeypatch, tmp_path, streaming):
    from rpipe.structure.origin import load_hf_dataset
    requests = pytest.importorskip('requests')
    calls = []

    def transport(self, request, **kwargs):
        calls.append(request.url)
        response = requests.Response()
        response.status_code = 200
        response._content = b'synthetic'
        response.url = request.url
        return response

    def rows():
        requests.get('https://huggingface.co/synthetic/data.json')
        yield {'value': 1}

    class Examples:
        generate_examples_fn = staticmethod(rows)

        def __iter__(self):
            return self.generate_examples_fn()

    class DeferredDataset:
        _ex_iterable = Examples()

        def __iter__(self):
            return iter(self._ex_iterable)

    def load_dataset(*args, **kwargs):
        return DeferredDataset() if streaming else list(rows())

    monkeypatch.setattr(requests.adapters.HTTPAdapter, 'send', transport)
    monkeypatch.setattr(socket.socket, 'connect', lambda *a, **k: pytest.fail('real network'))
    monkeypatch.setitem(sys.modules, 'datasets', SimpleNamespace(load_dataset=load_dataset))
    for name in ('datasets.config', 'huggingface_hub.constants'):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(HF_ENDPOINT='https://hf-mirror.com',
                                                             ENDPOINT='https://hf-mirror.com'))
    with pytest.raises(RuntimeError, match='domestic'):
        result = load_hf_dataset('synthetic/data', streaming=streaming)
        list(result)
    assert calls == []


def test_actual_dataset_streaming_preserves_views_state_and_pickle(monkeypatch, tmp_path):
    import pickle
    datasets = pytest.importorskip('datasets')
    from huggingface_hub import constants
    from rpipe.structure import origin
    monkeypatch.setattr(constants, 'ENDPOINT', 'https://hf-mirror.com')
    monkeypatch.setattr(datasets.config, 'HF_ENDPOINT', 'https://hf-mirror.com')
    monkeypatch.setattr(socket.socket, 'connect', lambda *a, **k: pytest.fail('real network'))
    paths = []
    for index in range(2):
        path = tmp_path / f'part{index}.jsonl'
        path.write_text(''.join('{"value":%d}\n' % value
                                for value in range(index * 2, index * 2 + 2)), encoding='utf-8')
        paths.append(str(path))
    result = origin.load_hf_dataset('json', data_files=paths, split='train', streaming=True)
    assert isinstance(result, datasets.IterableDataset)
    assert list(result) == [{'value': n} for n in range(4)]
    assert origin._download_origin.get() is None
    assert list(result.iter(batch_size=3)) == [{'value': [0, 1, 2]}, {'value': [3]}]
    assert sorted(row['value'] for row in result.shuffle(seed=7, buffer_size=2)) == list(range(4))
    assert list(result.shard(num_shards=2, index=1)) == [{'value': 2}, {'value': 3}]
    assert list(result.map(lambda row: {'next': row['value'] + 1}))[-1] == {'value': 3, 'next': 4}
    assert int(next(iter(result.with_format('torch')))['value']) == 0
    restored = pickle.loads(pickle.dumps(result))
    iterator = iter(restored)
    assert next(iterator) == {'value': 0}
    state = restored.state_dict()
    restored.load_state_dict(state)
    assert list(restored) == [{'value': n} for n in range(1, 4)]
    assert origin._download_origin.get() is None


def test_actual_dataset_deferred_generator_refuses_transfer(monkeypatch, tmp_path):
    datasets = pytest.importorskip('datasets')
    requests = pytest.importorskip('requests')
    from huggingface_hub import constants
    from rpipe.structure.origin import load_hf_dataset
    monkeypatch.setattr(constants, 'ENDPOINT', 'https://hf-mirror.com')
    monkeypatch.setattr(datasets.config, 'HF_ENDPOINT', 'https://hf-mirror.com')
    calls = []

    def transport(self, request, **kwargs):
        calls.append(request.url)
        raise AssertionError('foreign transport entered')

    def rows():
        requests.get('https://huggingface.co/synthetic.json')
        yield {'value': 1}

    result = datasets.IterableDataset.from_generator(rows)
    monkeypatch.setattr(datasets, 'load_dataset', lambda *a, **k: result)
    monkeypatch.setattr(requests.adapters.HTTPAdapter, 'send', transport)
    monkeypatch.setattr(socket.socket, 'connect', lambda *a, **k: pytest.fail('real network'))
    loaded = load_hf_dataset('synthetic/data', streaming=True)
    assert loaded is result
    with pytest.raises(RuntimeError, match='domestic'):
        list(loaded)
    assert calls == []


@pytest.mark.parametrize('target,allowed', [
    ('https://hf-mirror.com/blob', True),
    ('https://huggingface.co/blob', False),
    ('http://hf-mirror.com/blob', False),
    ('https://user:password@hf-mirror.com/blob', False),
])
def test_aiohttp_actual_redirect_loop_checks_before_next_connection(monkeypatch, target, allowed):
    import asyncio
    import inspect
    aiohttp = pytest.importorskip('aiohttp')
    from multidict import CIMultiDict, CIMultiDictProxy
    from rpipe.structure import origin
    calls = []

    async def connect(self, request, **kwargs):
        calls.append(str(request.url))
        return SimpleNamespace(protocol=SimpleNamespace(set_response_params=lambda **k: None),
                               close=lambda: None)

    async def send(request, connection):
        kwargs = dict(writer=None, continue100=None, timer=None, request_info=request.request_info,
                      traces=[], loop=asyncio.get_running_loop(), session=request.session)
        if 'stream_writer' in inspect.signature(aiohttp.ClientResponse).parameters:
            kwargs['stream_writer'] = SimpleNamespace(output_size=0)
        return aiohttp.ClientResponse(request.method, request.url, **kwargs)

    async def start(response, connection):
        redirected = response.url.path == '/entry'
        response.status = 302 if redirected else 200
        response.reason = 'synthetic'
        response._headers = CIMultiDictProxy(CIMultiDict({'Location': target} if redirected else {}))
        response._raw_headers = tuple((key.encode(), value.encode())
                                      for key, value in response._headers.items())
        response._body = b'synthetic'
        response._released = False
        response._closed = False
        return response

    monkeypatch.setattr(aiohttp.TCPConnector, 'connect', connect)
    monkeypatch.setattr(aiohttp.ClientRequest, 'send', send)
    monkeypatch.setattr(aiohttp.ClientResponse, 'start', start)

    async def run():
        # Windows creates its event-loop wakeup socket before this coroutine starts.
        monkeypatch.setattr(socket.socket, 'connect', lambda *a, **k: pytest.fail('real network'))
        async with aiohttp.ClientSession(trust_env=False) as client:
            with origin._domestic_transfers('mirror'):
                if allowed:
                    async with client.get('https://hf-mirror.com/entry') as response:
                        assert await response.read() == b'synthetic'
                else:
                    with pytest.raises(RuntimeError, match='domestic'):
                        await client.get('https://hf-mirror.com/entry')
            assert not client.closed
            assert calls == (['https://hf-mirror.com/entry', target] if allowed
                             else ['https://hf-mirror.com/entry'])
            async with client.get('https://example.org/unrelated') as response:
                assert response.status == 200
        assert client.closed

    asyncio.run(run())


def _dataset_worker_policy_probe(batch):
    import os
    import requests
    import threading
    from rpipe.structure import origin
    entered = []

    class Adapter(requests.adapters.BaseAdapter):
        def send(self, request, **kwargs):
            entered.append(request.url)
            response = requests.Response()
            response.status_code = 200
            response._content = b'synthetic'
            return response

        def close(self):
            pass

    with requests.Session() as client:
        client.mount('https://', Adapter())
        try:
            client.get('https://huggingface.co/synthetic-worker')
        except RuntimeError:
            pass
    return [dict(policy=origin._download_origin.get(), pid=os.getpid(),
                 thread=threading.get_ident(), transport=bool(entered))
            for _ in batch]


@pytest.mark.parametrize('backend', ['spawn', 'threads'])
def test_dataset_download_workers_keep_domestic_policy(monkeypatch, backend):
    import importlib
    import multiprocessing
    import os
    import threading
    from multiprocessing.pool import ThreadPool
    datasets = pytest.importorskip('datasets')
    from datasets.download import download_manager
    from huggingface_hub import constants
    from rpipe.structure.origin import load_hf_dataset
    parallel = importlib.import_module('datasets.parallel.parallel')
    pool = multiprocessing.get_context('spawn').Pool if backend == 'spawn' else ThreadPool
    monkeypatch.setattr(parallel, 'Pool', pool)
    monkeypatch.setattr(constants, 'ENDPOINT', 'https://hf-mirror.com')
    monkeypatch.setattr(datasets.config, 'HF_ENDPOINT', 'https://hf-mirror.com')

    def load_dataset(*args, **kwargs):
        return download_manager.map_nested(_dataset_worker_policy_probe, list(range(4)),
                                          num_proc=2, batched=True, batch_size=2,
                                          disable_tqdm=True)

    monkeypatch.setattr(datasets, 'load_dataset', load_dataset)
    result = load_hf_dataset('synthetic/data')
    assert len(result) == 4
    if backend == 'spawn':
        assert all(row['pid'] != os.getpid() for row in result), 'probe did not reach child processes'
    else:
        assert all(row['thread'] != threading.get_ident() for row in result), 'probe ran on caller thread'
    assert all(row['policy'] == 'mirror' and not row['transport'] for row in result)


@pytest.mark.parametrize('primary,close_error', [
    (OSError('read failed'), OSError('close failed')),
    (KeyboardInterrupt('cancel'), OSError('close failed')),
    (ValueError('decode failed'), OSError('close failed')),
    (None, OSError('close failed')),
    (OSError('read failed'), ValueError('programming close error')),
    (None, None),
])
def test_dataset_generator_close_keeps_primary_failure_and_context(primary, close_error):
    from rpipe.structure import origin
    closed = []

    class Iterator:
        def __iter__(self):
            return self

        def __next__(self):
            assert origin._download_origin.get() == 'mirror'
            if primary is not None:
                raise primary
            raise StopIteration

        def close(self):
            closed.append(origin._download_origin.get())
            if close_error is not None:
                raise close_error

    iterator = origin._DomesticDatasetGenerator(Iterator)()
    expected = close_error if primary is None or isinstance(close_error, ValueError) else primary
    if expected is None:
        assert list(iterator) == []
    else:
        with pytest.raises(type(expected)) as caught:
            next(iterator)
        assert caught.value is expected
    assert closed == ['mirror']
    assert origin._download_origin.get() is None
    assert origin._download_transports_installed is False


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('host,allowed', [('hf-mirror.com', True), ('huggingface.co', False)])
def test_actual_dataset_media_reads_keep_domestic_policy(monkeypatch, streaming, host, allowed):
    datasets = pytest.importorskip('datasets')
    requests = pytest.importorskip('requests')
    from PIL import Image
    from huggingface_hub import constants
    import datasets.features.image as image_module
    from rpipe.structure.origin import load_hf_dataset
    monkeypatch.setattr(constants, 'ENDPOINT', 'https://hf-mirror.com')
    monkeypatch.setattr(datasets.config, 'HF_ENDPOINT', 'https://hf-mirror.com')
    content = io.BytesIO()
    Image.new('RGB', (1, 1), (1, 2, 3)).save(content, format='PNG')
    calls = []
    url = f'https://{host}/synthetic/image.png'

    def transport(self, request, **kwargs):
        calls.append(request.url)
        response = requests.Response()
        response.status_code = 200
        response._content = content.getvalue()
        return response

    def xopen(path, *args, **kwargs):
        return io.BytesIO(requests.get(path).content)

    def rows():
        yield {'image': url}

    features = datasets.Features({'image': datasets.Image()})
    result = (datasets.IterableDataset.from_generator(rows, features=features) if streaming
              else datasets.Dataset.from_dict({'image': [url]}, features=features))
    monkeypatch.setattr(datasets, 'load_dataset', lambda *a, **k: result)
    monkeypatch.setattr(image_module, 'xopen', xopen)
    monkeypatch.setattr(requests.adapters.HTTPAdapter, 'send', transport)
    monkeypatch.setattr(socket.socket, 'connect', lambda *a, **k: pytest.fail('real network'))
    loaded = load_hf_dataset('synthetic/images', streaming=streaming)

    def consume():
        return next(iter(loaded)) if streaming else loaded[0]

    if allowed:
        assert consume()['image'].getpixel((0, 0)) == (1, 2, 3)
        assert calls == [url]
    else:
        with pytest.raises(RuntimeError, match='domestic'):
            consume()
        assert calls == []


def test_domestic_sdk_worker_keeps_guard_after_parent_scope_returns(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    requests = pytest.importorskip('requests')
    from rpipe.structure import origin
    started, release = Event(), Event()
    calls = []

    def transport(self, request, **kwargs):
        calls.append(request.url)
        response = requests.Response()
        response.status_code = 200
        response._content = b'synthetic'
        return response

    def worker():
        started.set()
        assert release.wait(5), 'test did not release worker'
        try:
            requests.get('https://huggingface.co/synthetic-prefetch')
        except RuntimeError:
            return True
        return False

    monkeypatch.setattr(requests.adapters.HTTPAdapter, 'send', transport)
    monkeypatch.setattr(socket.socket, 'connect', lambda *a, **k: pytest.fail('real network'))
    with ThreadPoolExecutor(max_workers=1) as executor:
        with origin._domestic_transfers('mirror'):
            future = executor.submit(worker)
            assert started.wait(5), 'worker did not start'
        release.set()
        assert future.result(timeout=5), 'worker lost guard when its parent returned'
    assert calls == []
    assert origin._download_origin.get() is None
    assert origin._download_transports_installed is False
