"""Study ``origin``: one switch for data hosts and model hubs.

``foreign`` uses the upstream hosts. ``domestic`` uses mirrors reachable from
mainland China. A Study sets this once; data and model downloads both read it.
Missing origin means ``foreign`` and does not change ``HF_ENDPOINT``.
"""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

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


@contextmanager
def bind_endpoint(ctor: Any, endpoint: DownloadEndpoint) -> Iterator[DownloadEndpoint]:
    """Point one torchvision dataset class at this endpoint, then restore it."""
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
