"""Study ``origin``: one switch for data hosts and model hubs.

``foreign`` uses the upstream hosts. ``domestic`` uses mirrors reachable from
mainland China. A Study sets this once; data and model downloads both read it.
Missing origin means ``foreign`` and does not change ``HF_ENDPOINT``.
"""

from __future__ import annotations

import hashlib
import importlib
import re
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar, copy_context
from functools import wraps
from inspect import iscoroutinefunction
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Iterator
from urllib.parse import urljoin, urlsplit

FOREIGN = 'foreign'
DOMESTIC = 'domestic'
FOREIGN_MODEL_HUB = 'https://huggingface.co'
DOMESTIC_MODEL_HUB = 'https://hf-mirror.com'

_CIFAR10_MD5 = 'c58f30108f718f92721af3b95e74349a'
_CIFAR100_MD5 = 'eb9058c3a382ffc7106e4002c42a8d85'
_BCEBOS = 'https://dataset.bj.bcebos.com'


@dataclass(frozen=True)
class DownloadEndpoint:
    name: str
    origin: str
    url: str | None = None
    mirrors: tuple[str, ...] = ()
    archive: str | None = None
    archive_md5: str | None = None

    @property
    def location(self) -> str:
        if self.url:
            return self.url
        if self.mirrors:
            return self.mirrors[0]
        return self.origin


def normalize_origin(value: Any) -> str:
    if value in (None, ''):
        return FOREIGN
    text = str(value).strip().lower()
    if text not in (FOREIGN, DOMESTIC):
        raise ValueError("origin must be 'foreign' or 'domestic'")
    return text


def model_hub(origin: Any) -> str:
    if normalize_origin(origin) == DOMESTIC:
        return DOMESTIC_MODEL_HUB
    return FOREIGN_MODEL_HUB


def apply_model_origin(origin: Any) -> str:
    """Point this process at the Study's model hub. Returns the hub URL."""
    hub = model_hub(origin)
    os.environ['HF_ENDPOINT'] = hub
    return hub


def foreign_endpoint(name: str) -> DownloadEndpoint:
    """Official torchvision hosts."""
    if name == 'CIFAR10':
        return DownloadEndpoint(
            name,
            FOREIGN,
            url='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            archive='cifar-10-python.tar.gz',
            archive_md5=_CIFAR10_MD5,
        )
    if name == 'CIFAR100':
        return DownloadEndpoint(
            name,
            FOREIGN,
            url='https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
            archive='cifar-100-python.tar.gz',
            archive_md5=_CIFAR100_MD5,
        )
    if name == 'MNIST':
        return DownloadEndpoint(
            name,
            FOREIGN,
            mirrors=(
                'https://ossci-datasets.s3.amazonaws.com/mnist/',
                'http://yann.lecun.com/exdb/mnist/',
            ),
        )
    if name in ('FashionMNIST', 'SVHN'):
        return DownloadEndpoint(name, FOREIGN)
    raise ValueError(f'no foreign download endpoint for {name}')


def domestic_endpoint(name: str) -> DownloadEndpoint:
    """Mainland data mirrors. Same archives and md5 as the foreign files."""
    if name == 'CIFAR10':
        return DownloadEndpoint(
            name,
            DOMESTIC,
            url=f'{_BCEBOS}/cifar/cifar-10-python.tar.gz',
            archive='cifar-10-python.tar.gz',
            archive_md5=_CIFAR10_MD5,
        )
    if name == 'CIFAR100':
        return DownloadEndpoint(
            name,
            DOMESTIC,
            url=f'{_BCEBOS}/cifar/cifar-100-python.tar.gz',
            archive='cifar-100-python.tar.gz',
            archive_md5=_CIFAR100_MD5,
        )
    if name == 'MNIST':
        return DownloadEndpoint(
            name,
            DOMESTIC,
            mirrors=(f'{_BCEBOS}/mnist/',),
        )
    raise ValueError(f'no domestic download endpoint for {name}')


def endpoint_for(name: str, origin: Any) -> DownloadEndpoint:
    chosen = normalize_origin(origin)
    if chosen == DOMESTIC:
        return domestic_endpoint(name)
    return foreign_endpoint(name)


def discard_bad_archive(root: Path, endpoint: DownloadEndpoint) -> None:
    """Drop a partial archive so the next download does not keep a bad file."""
    if not endpoint.archive or not endpoint.archive_md5:
        return
    path = Path(root) / endpoint.archive
    if not path.is_file():
        return
    if _md5(path) == endpoint.archive_md5:
        return
    path.unlink()


_endpoint_lock = RLock()


@contextmanager
def bind_endpoint(ctor: Any, endpoint: DownloadEndpoint) -> Iterator[DownloadEndpoint]:
    """Point one torchvision dataset class at this endpoint, then restore it."""
    with _endpoint_lock:
        saved_url = getattr(ctor, 'url', None)
        saved_mirrors = getattr(ctor, 'mirrors', None)
        if endpoint.url is not None and saved_url is not None:
            ctor.url = endpoint.url
        if endpoint.mirrors and saved_mirrors is not None:
            ctor.mirrors = list(endpoint.mirrors)
        try:
            yield endpoint
        finally:
            if endpoint.url is not None and saved_url is not None:
                ctor.url = saved_url
            if endpoint.mirrors and saved_mirrors is not None:
                ctor.mirrors = saved_mirrors


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open('rb') as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


_download_lock = RLock()
_download_origin: ContextVar[str | None] = ContextVar('rpipe_download_origin', default=None)


def _check_transfer_url(value: Any) -> None:
    policy = _download_origin.get()
    if policy is None:
        return
    url = urlsplit(str(value))
    host = url.hostname or ''
    allowed = host == 'hf-mirror.com' if policy == 'mirror' else (
        host == 'modelscope.cn' or host.endswith('.modelscope.cn')
        or re.fullmatch(r'[a-z0-9.-]+\.oss-cn-[a-z0-9-]+\.aliyuncs\.com', host) is not None
    )
    if (url.scheme != 'https' or url.username is not None or url.password is not None
            or url.port not in (None, 443) or not allowed):
        raise RuntimeError(f'domestic {policy} transfer refused unapproved host {host!r}')


class _DownloadConstants:
    def __init__(self, original: Any):
        self.original = original

    def __getattr__(self, name: str) -> Any:
        if _download_origin.get() is not None:
            if name == 'HF_HUB_ENABLE_HF_TRANSFER':
                return False
            if name == 'HF_HUB_DISABLE_XET':
                return True
        return getattr(self.original, name)


@contextmanager
def _domestic_transfers(policy: str) -> Iterator[None]:
    """Guard SDK transports without reconfiguring their shared clients."""
    with _download_lock, ExitStack() as stack:
        token = _download_origin.set(policy)
        stack.callback(_download_origin.reset, token)

        def replace(owner: Any, name: str, replacement: Any) -> None:
            original = getattr(owner, name)
            setattr(owner, name, replacement)

            def restore() -> None:
                if getattr(owner, name) is replacement:
                    setattr(owner, name, original)

            stack.callback(restore)

        def guarded_send(original: Any) -> Any:
            if iscoroutinefunction(original):
                @wraps(original)
                async def async_send(client: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
                    _check_transfer_url(request.url)
                    return await original(client, request, *args, **kwargs)
                return async_send

            @wraps(original)
            def send(client: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
                _check_transfer_url(request.url)
                return original(client, request, *args, **kwargs)
            return send

        for name in ('requests', 'httpx', 'aiohttp'):
            try:
                module = importlib.import_module(name)
            except ModuleNotFoundError as error:
                if error.name != name:
                    raise
                continue
            if name == 'requests':
                replace(module.Session, 'send', guarded_send(module.Session.send))
            elif name == 'httpx':
                for cls in (module.Client, module.AsyncClient):
                    replace(cls, '_send_single_request', guarded_send(cls._send_single_request))
            else:
                original_init = module.ClientRequest.__init__
                original_start = module.ClientResponse.start

                @wraps(original_init)
                def request_init(request: Any, method: Any, url: Any, *args: Any, **kwargs: Any) -> None:
                    _check_transfer_url(url)
                    original_init(request, method, url, *args, **kwargs)

                @wraps(original_start)
                async def response_start(response: Any, *args: Any, **kwargs: Any) -> Any:
                    result = await original_start(response, *args, **kwargs)
                    if 300 <= response.status < 400 and response.headers.get('Location'):
                        _check_transfer_url(urljoin(str(response.url), response.headers['Location']))
                    return result

                replace(module.ClientRequest, '__init__', request_init)
                replace(module.ClientResponse, 'start', response_start)

        original_submit = ThreadPoolExecutor.submit

        @wraps(original_submit)
        def submit(executor: Any, function: Any, /, *args: Any, **kwargs: Any) -> Any:
            if _download_origin.get() is None:
                return original_submit(executor, function, *args, **kwargs)
            return original_submit(executor, copy_context().run, function, *args, **kwargs)

        replace(ThreadPoolExecutor, 'submit', submit)
        file_download = sys.modules.get('huggingface_hub.file_download')
        if file_download is not None:
            original_available = file_download.is_xet_available

            @wraps(original_available)
            def xet_available() -> bool:
                return _download_origin.get() is None and original_available()

            replace(file_download, 'is_xet_available', xet_available)
            replace(file_download, 'constants', _DownloadConstants(file_download.constants))
        yield


def download_model(repo_id: str, cache_dir: Path | str, *, revision: str | None = None) -> str:
    """Resolve a local input or download via ModelScope, then hf-mirror only."""
    if Path(repo_id).exists():
        return str(repo_id)
    try:
        with _domestic_transfers('modelscope'):
            from modelscope import snapshot_download

            return snapshot_download(model_id=repo_id, revision=revision, cache_dir=str(cache_dir))
    except Exception as modelscope_error:
        # SDK failures permit a domestic fallback; cancellation still propagates.
        try:
            from huggingface_hub import snapshot_download

            with _domestic_transfers('mirror'):
                return snapshot_download(repo_id=repo_id, revision=revision,
                                         cache_dir=str(cache_dir), endpoint=DOMESTIC_MODEL_HUB)
        except Exception as mirror_error:
            raise RuntimeError(
                f'ModelScope failed ({modelscope_error}); hf-mirror failed ({mirror_error}); '
                'no overseas fallback is permitted'
            ) from mirror_error


def load_hf_dataset(*args: Any, **kwargs: Any) -> Any:
    """Load HF datasets after selecting the domestic endpoint before import."""
    _check_dataset_urls((args, kwargs.get('path'), kwargs.get('data_files'), kwargs.get('data_dir')))
    apply_model_origin(DOMESTIC)
    _check_hf_endpoint()
    from datasets import load_dataset

    _check_hf_endpoint()
    with _domestic_transfers('mirror'):
        result = load_dataset(*args, **kwargs)
    return _bind_dataset_transfers(result) if kwargs.get('streaming') else result


@dataclass
class _DomesticDatasetGenerator:
    function: Any

    def __call__(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        with _domestic_transfers('mirror'):
            iterator = iter(self.function(*args, **kwargs))
        failed = False
        end = object()
        try:
            while True:
                with _domestic_transfers('mirror'):
                    row = next(iterator, end)
                if row is end:
                    return
                yield row
        except BaseException:
            failed = True
            raise
        finally:
            close = getattr(iterator, 'close', None)
            if close is not None:
                try:
                    with _domestic_transfers('mirror'):
                        close()
                except OSError:
                    if not failed:
                        raise


def _bind_dataset_transfers(result: Any) -> Any:
    if isinstance(result, dict):
        for value in result.values():
            _bind_dataset_transfers(value)
        return result
    examples = getattr(result, '_ex_iterable', None)
    if examples is None:
        raise RuntimeError('domestic streaming requires a supported datasets examples iterable')
    bound = False
    for name in ('generate_examples_fn', 'generate_tables_fn', 'generate_more_kwargs_fn'):
        function = getattr(examples, name, None)
        if function is not None:
            if not isinstance(function, _DomesticDatasetGenerator):
                setattr(examples, name, _DomesticDatasetGenerator(function))
            bound = True
    if not bound:
        raise RuntimeError('domestic streaming requires a supported datasets generator')
    return result


def _check_hf_endpoint() -> None:
    for name, attr in (('huggingface_hub.constants', 'ENDPOINT'),
                       ('datasets.config', 'HF_ENDPOINT')):
        module = sys.modules.get(name)
        if module is not None and getattr(module, attr, DOMESTIC_MODEL_HUB) != DOMESTIC_MODEL_HUB:
            raise RuntimeError(f'select domestic origin before importing {name}; start a fresh worker')


def _check_dataset_urls(value: Any) -> None:
    if isinstance(value, dict):
        value = tuple(value.values())
    if isinstance(value, (list, tuple)):
        for item in value:
            _check_dataset_urls(item)
    elif isinstance(value, str) and '://' in value:
        url = urlsplit(value)
        if url.scheme != 'hf' and (url.scheme != 'https' or url.netloc != 'hf-mirror.com'):
            raise ValueError('HF dataset URLs must use domestic hf-mirror; prepare other inputs locally')
