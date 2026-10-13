"""Pinned historical archive and observation support for this Study."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable


STUDY = Path(__file__).resolve().parents[2]
REPO = STUDY.parents[1]
COMMIT = '4ccb28d0496110253e9f8e3f3df658853f07996b'
SOURCE = 'historical_4ccb28d'
HORIZON = 80000
DATA_NAMES = ('MNIST', 'CIFAR10')
MODEL_NAMES = ('linear', 'mlp', 'cnn', 'resnet18')
ARCHIVE_ROOT = REPO / '.tmp' / 'historical-source-4ccb28d'
_ARCHIVE: Any = None


def bootstrap() -> None:
    """Load the repository and temporary dependencies without shell env setup."""
    for path in (REPO / 'src', REPO / '.tmp' / 'runtime'):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    # This import order is required by this Windows host's native runtimes.
    import numpy  # noqa: F401


def export_source() -> dict[str, Any]:
    """Export tracked blobs, refusing to overwrite a different existing file."""
    tree = subprocess.check_output(
        ['git', 'ls-tree', '--full-tree', '-r', '-z', COMMIT], cwd=REPO
    )
    blobs = []
    for entry in tree.split(b'\0'):
        if not entry:
            continue
        attributes, raw_name = entry.split(b'\t', 1)
        _, kind, oid = attributes.decode('ascii').split()
        if kind != 'blob':
            continue
        blobs.append((raw_name.decode('utf-8'), oid))
    # Windows process creation dominated archive preparation when every blob
    # used its own cat-file process. Git's batch protocol preserves raw bytes.
    request = ''.join(oid + '\n' for _, oid in blobs).encode('ascii')
    stream = subprocess.check_output(['git', 'cat-file', '--batch'], input=request, cwd=REPO)
    offset = 0
    rows = []
    for name, oid in blobs:
        end = stream.find(b'\n', offset)
        if end < 0:
            raise ValueError('truncated Git batch header')
        header = stream[offset:end].decode('ascii').split()
        if len(header) != 3 or header[:2] != [oid, 'blob']:
            raise ValueError(f'unexpected Git batch header for {name}: {header}')
        size = int(header[2])
        if size < 0:
            raise ValueError(f'invalid Git blob size for {name}')
        offset = end + 1
        payload = stream[offset:offset + size]
        offset += size
        if len(payload) != size or stream[offset:offset + 1] != b'\n':
            raise ValueError(f'truncated Git batch blob for {name}')
        offset += 1
        target = ARCHIVE_ROOT / name
        if target.exists():
            if target.read_bytes() != payload:
                raise ValueError(f'archived source differs from Git: {target}')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        rows.append({'path': name, 'git_blob': oid,
                     'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)})
    if offset != len(stream):
        raise ValueError('unexpected trailing Git batch output')
    manifest = {'commit': COMMIT, 'root': str(ARCHIVE_ROOT), 'files': rows}
    (ARCHIVE_ROOT / 'SOURCE_MANIFEST.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    return manifest


@contextlib.contextmanager
def _working_directory(directory: Path):
    previous = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


class Archive:
    """Keep old top-level imports out of the process's normal module namespace."""

    def __init__(self) -> None:
        bootstrap()
        import torch

        benchmark = torch.backends.cudnn.benchmark
        self.manifest = export_source()
        roots = ('config', 'module', 'dataset', 'model', 'metric', 'train_model')
        is_old = lambda name: any(name == root or name.startswith(root + '.') for root in roots)
        saved = {name: value for name, value in sys.modules.items() if is_old(name)}
        for name in saved:
            del sys.modules[name]
        old_argv, old_path = sys.argv, list(sys.path)
        try:
            sys.path.insert(0, str(ARCHIVE_ROOT / 'src'))
            sys.argv = ['train_model.py']
            with _working_directory(ARCHIVE_ROOT / 'src'):
                self.config = importlib.import_module('config')
                self.module = importlib.import_module('module')
                self.dataset = importlib.import_module('dataset')
                self.model = importlib.import_module('model')
                self.metric = importlib.import_module('metric')
                self.train_model = importlib.import_module('train_model')
            self.modules = {name: value for name, value in sys.modules.items() if is_old(name)}
            self.defaults = copy.deepcopy(self.config.cfg)
            for name, value in self.modules.items():
                sys.modules['_rpipe_archive_4ccb28d.' + name] = value
        finally:
            for name in list(sys.modules):
                if is_old(name):
                    del sys.modules[name]
            sys.modules.update(saved)
            sys.path[:] = old_path
            sys.argv = old_argv
            # Archived train_model sets benchmark=True at import. Importing
            # historical code must not override the current Run's controls.
            torch.backends.cudnn.benchmark = benchmark

    def configure(self, data_name: str, model_name: str, seed: int, device: str = 'cpu') -> dict[str, Any]:
        if data_name not in DATA_NAMES or model_name not in MODEL_NAMES:
            raise ValueError(f'unsupported historical combination: {data_name}/{model_name}')
        cfg = self.config.cfg
        cfg.clear()
        cfg.update(copy.deepcopy(self.defaults))
        cfg['control'] = {'data_name': data_name, 'model_name': model_name}
        cfg['control_name'] = f'{data_name}_{model_name}'
        cfg['model_tag'] = f'{seed}_{data_name}_{model_name}'
        self.module.process_control()
        cfg.update(seed=int(seed), iteration=0, device=device, num_samples=250 * HORIZON)
        return cfg

    def datasets(self, data_root: Path, data_name: str) -> dict[str, Any]:
        # make_dataset has a literal data/<name> root. Construction eagerly
        # loads arrays; afterwards all sample reads use those in-memory arrays.
        if data_root.name != 'data':
            raise ValueError('historical data_root must end in "data"')
        from rpipe.structure.artifact.shared import cache_lock
        with cache_lock(data_root), _working_directory(data_root.parent):
            return self.dataset.make_dataset(data_name, verbose=False)


def load_archive() -> Archive:
    global _ARCHIVE
    if _ARCHIVE is None:
        _ARCHIVE = Archive()
    return _ARCHIVE


class Observation:
    """Hash actual sampled and transformed inputs; used only in preflight."""

    def __init__(self) -> None:
        self.hashes = {key: hashlib.sha256() for key in ('indices', 'images', 'targets')}
        self.samples = 0
        self.batches = 0

    def add(self, batch: dict[str, Any]) -> None:
        for key, field in (('indices', 'id'), ('images', 'data'), ('targets', 'target')):
            tensor = batch[field].detach().cpu().contiguous()
            self.hashes[key].update(tensor.numpy().tobytes())
        self.samples += len(batch['target'])
        self.batches += 1

    def result(self) -> dict[str, Any]:
        return {'samples': self.samples, 'batches': self.batches,
                **{key + '_sha256': value.hexdigest() for key, value in self.hashes.items()}}


def _observed_dicts(loader: Iterable[Any], observation: Observation):
    for batch in loader:
        observation.add(batch)
        yield batch
