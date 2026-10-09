"""Prepare the unchanged MNIST/CIFAR10 raw files used by historical main.

Run with the same Python executable as the study, for example::

    python -B studies/main_exp/prepare_data.py
    python -B studies/main_exp/prepare_data.py --verify-only

Only upstream dataset URLs from torchvision and pinned main 4ccb28d are used.
The legacy datasets create their own pickle-format ``processed`` caches later.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import struct
import tarfile
import urllib.request


STUDY = Path(__file__).resolve().parent
HISTORICAL_REF = '4ccb28d0496110253e9f8e3f3df658853f07996b'
MNIST_URL = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
MNIST_RESOURCES = (
    ('train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
    ('train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
    ('t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
    ('t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c'),
)
CIFAR_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR_MD5 = 'c58f30108f718f92721af3b95e74349a'


def digest(path: Path, algorithm: str = 'sha256') -> str:
    hasher = hashlib.new(algorithm)
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def expected_files(manifest: Path) -> dict[str, str]:
    source = json.loads(manifest.read_text(encoding='utf-8'))
    result = {}
    for relative, expected in source.items():
        relative = relative.replace('\\', '/')
        if not relative.startswith('data/'):
            raise ValueError(f'unexpected historical manifest path: {relative}')
        result[relative.removeprefix('data/')] = expected
    return result


def verify_file(path: Path, expected: str, algorithm: str = 'sha256') -> bool:
    return path.is_file() and digest(path, algorithm) == expected


def download(url: str, destination: Path, md5: str) -> dict[str, str]:
    if verify_file(destination, md5, 'md5'):
        print(f'cached archive: {destination}', flush=True)
        return {'url': url, 'source': 'existing archive', 'md5': md5}
    if destination.exists():
        raise ValueError(f'existing archive checksum mismatch: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.part')
    print(f'download: {url}', flush=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'RPipe historical data preparation'})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open('wb') as handle:
        actual_url = response.geturl()
        expected_size = response.headers.get('Content-Length')
        total = 0
        next_report = 32 * 1024 * 1024
        while block := response.read(1024 * 1024):
            handle.write(block)
            total += len(block)
            if total >= next_report:
                print(f'downloaded {total / 1024 / 1024:.0f} MiB: {destination.name}', flush=True)
                next_report += 32 * 1024 * 1024
    if expected_size is not None and total != int(expected_size):
        raise ValueError(f'incomplete download: {destination}, got {total} of {expected_size} bytes')
    if digest(partial, 'md5') != md5:
        raise ValueError(f'download checksum mismatch; retained partial file: {partial}')
    partial.replace(destination)
    return {'url': url, 'actual_url': actual_url, 'source': 'download', 'md5': md5}


def reuse_caches(data_root: Path, expected: dict[str, str], cache_roots: list[Path]) -> dict[str, str]:
    copied = {}
    for relative, expected_hash in expected.items():
        destination = data_root / relative
        if destination.is_file():
            if not verify_file(destination, expected_hash):
                raise ValueError(f'existing raw file checksum mismatch: {destination}')
            continue
        for cache_root in cache_roots:
            match = next((path for path in cache_root.rglob(destination.name)
                          if path.is_file() and path.resolve() != destination.resolve()
                          and verify_file(path, expected_hash)), None)
            if match is not None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(match, destination)
                copied[relative] = str(match.resolve())
                print(f'reused raw cache: {match}', flush=True)
                break
    return copied


def prepare_mnist(data_root: Path, expected: dict[str, str]) -> list[dict[str, str]]:
    folder = data_root / 'MNIST' / 'raw'
    folder.mkdir(parents=True, exist_ok=True)
    downloads = []
    for name, md5 in MNIST_RESOURCES:
        archive = folder / name
        downloads.append(download(MNIST_URL + name, archive, md5))
        extracted = archive.with_suffix('')
        relative = extracted.relative_to(data_root).as_posix()
        if verify_file(extracted, expected[relative]):
            continue
        if extracted.exists():
            raise ValueError(f'existing IDX checksum mismatch: {extracted}')
        partial = extracted.with_name(extracted.name + '.part')
        with gzip.open(archive, 'rb') as source, partial.open('wb') as destination:
            shutil.copyfileobj(source, destination)
        if not verify_file(partial, expected[relative]):
            raise ValueError(f'extracted IDX checksum mismatch: {partial}')
        partial.replace(extracted)
    return downloads


def prepare_cifar(data_root: Path, expected: dict[str, str]) -> list[dict[str, str]]:
    required = {relative: value for relative, value in expected.items() if relative.startswith('CIFAR10/')}
    if all(verify_file(data_root / relative, value) for relative, value in required.items()):
        return [{'url': CIFAR_URL, 'source': 'verified raw cache', 'md5': CIFAR_MD5}]
    folder = data_root / 'CIFAR10' / 'raw'
    archive = folder / 'cifar-10-python.tar.gz'
    record = download(CIFAR_URL, archive, CIFAR_MD5)
    with tarfile.open(archive, 'r:gz') as handle:
        for relative, expected_hash in required.items():
            destination = data_root / relative
            if verify_file(destination, expected_hash):
                continue
            if destination.exists():
                raise ValueError(f'existing CIFAR raw checksum mismatch: {destination}')
            member_name = destination.relative_to(folder).as_posix()
            member = handle.getmember(member_name)
            if not member.isfile():
                raise ValueError(f'expected a regular dataset member: {member_name}')
            source = handle.extractfile(member)
            if source is None:
                raise ValueError(f'cannot read dataset member: {member_name}')
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_name(destination.name + '.part')
            with source, partial.open('wb') as target:
                shutil.copyfileobj(source, target)
            if not verify_file(partial, expected_hash):
                raise ValueError(f'extracted CIFAR checksum mismatch: {partial}')
            partial.replace(destination)
    return [record]


def split_counts(data_root: Path) -> dict[str, dict[str, int]]:
    counts = {}
    mnist = data_root / 'MNIST' / 'raw'
    for split, prefix, expected in [('train', 'train', 60000), ('test', 't10k', 10000)]:
        images = mnist / f'{prefix}-images-idx3-ubyte'
        labels = mnist / f'{prefix}-labels-idx1-ubyte'
        with images.open('rb') as handle:
            magic, n, rows, cols = struct.unpack('>IIII', handle.read(16))
        with labels.open('rb') as handle:
            label_magic, label_n = struct.unpack('>II', handle.read(8))
        if (magic, n, rows, cols, label_magic, label_n) != (2051, expected, 28, 28, 2049, expected):
            raise ValueError(f'invalid MNIST {split} IDX header')
        if images.stat().st_size != 16 + n * rows * cols or labels.stat().st_size != 8 + label_n:
            raise ValueError(f'invalid MNIST {split} IDX length')
    counts['MNIST'] = {'train': 60000, 'test': 10000, 'classes': 10}
    cifar = data_root / 'CIFAR10' / 'raw' / 'cifar-10-batches-py'
    for name in [*(f'data_batch_{i}' for i in range(1, 6)), 'test_batch']:
        with (cifar / name).open('rb') as handle:
            batch = pickle.load(handle, encoding='latin1')
        if batch['data'].shape != (10000, 3072) or len(batch['labels']) != 10000:
            raise ValueError(f'invalid CIFAR10 batch dimensions: {name}')
        if min(batch['labels']) < 0 or max(batch['labels']) > 9:
            raise ValueError(f'invalid CIFAR10 labels: {name}')
    with (cifar / 'batches.meta').open('rb') as handle:
        if len(pickle.load(handle, encoding='latin1')['label_names']) != 10:
            raise ValueError('invalid CIFAR10 metadata')
    counts['CIFAR10'] = {'train': 50000, 'test': 10000, 'classes': 10}
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=STUDY / 'shared' / 'data')
    parser.add_argument('--manifest', type=Path, default=STUDY / 'docs' / 'DATA_MANIFEST.json')
    parser.add_argument('--expected-manifest', type=Path,
                        default=STUDY / 'docs' / 'EXPECTED_DATA.json')
    parser.add_argument('--cache-root', type=Path, action='append', default=[])
    parser.add_argument('--verify-only', action='store_true', help='validate complete raw files without downloads')
    args = parser.parse_args(argv)
    data_root = args.data_root.resolve()
    expected = expected_files(args.expected_manifest)
    reused, downloads = {}, []
    if not args.verify_only:
        reused = reuse_caches(data_root, expected, args.cache_root)
        downloads = prepare_mnist(data_root, expected) + prepare_cifar(data_root, expected)
    elif args.manifest.is_file():
        previous = json.loads(args.manifest.read_text(encoding='utf-8'))
        downloads = previous.get('downloads', [])
        reused = previous.get('reused_caches', {})
    files = []
    for relative, expected_hash in sorted(expected.items()):
        path = data_root / relative
        if not verify_file(path, expected_hash):
            raise ValueError(f'raw dataset differs from historical manifest: {path}')
        files.append({'path': relative, 'bytes': path.stat().st_size,
                      'sha256': expected_hash, 'historical_sha256_match': True})
    try:
        expected_path = args.expected_manifest.resolve().relative_to(STUDY.parent.parent).as_posix()
    except ValueError:
        expected_path = str(args.expected_manifest.resolve())
    manifest = {
        'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
        'historical_ref': HISTORICAL_REF,
        'layout': 'shared/data/{MNIST,CIFAR10}/raw; legacy datasets create processed caches',
        'upstream_urls': [MNIST_URL, CIFAR_URL],
        'expected_manifest': expected_path,
        'expected_manifest_sha256': digest(args.expected_manifest),
        'preparation_script_sha256': digest(Path(__file__)),
        'raw_files': len(files),
        'historical_sha256_matches': len(files),
        'split_counts': split_counts(data_root),
        'downloads': downloads,
        'reused_caches': reused,
        'files': files,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'verified {len(files)}/{len(expected)} historical raw files; manifest: {args.manifest}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
