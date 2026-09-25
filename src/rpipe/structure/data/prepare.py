"""Materialize Study-level shared datasets once (not per Run)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rpipe.structure.artifact.activity import announce
from rpipe.structure.artifact.config import load_config
from rpipe.structure.artifact.layout import ensure_study_layout
from rpipe.structure.data.config import DataConfig
from rpipe.structure.data.factory import DataFactory, vision_root
from rpipe.structure.origin import endpoint_for, normalize_origin

READY_NAME = '.ready'


def prepare_shared_data(study_dir: Path | str, config_paths: list[Path]) -> list[str]:
    """Download / build each unique ``data.name`` + ``source`` into ``shared/data``.

    ``train_size`` is ignored here so subset Runs reuse the same files.
    A dataset is cached only after a successful build writes ``.ready``.
    Returns names materialized on this call (cached names omitted).
    """
    study = ensure_study_layout(study_dir)
    shared = study / 'shared' / 'data'
    seen: set[tuple[str, str, str]] = set()
    names: list[str] = []
    previous = os.environ.get('TQDM_DISABLE')
    os.environ['TQDM_DISABLE'] = '1'
    try:
        for path in config_paths:
            loaded = load_config(path)
            mapping = _shared_data_mapping(loaded)
            if mapping is None:
                continue
            origin = normalize_origin(loaded.get('origin'))
            key = (str(mapping.get('name') or ''), str(mapping.get('source') or ''), origin)
            if key in seen:
                continue
            seen.add(key)
            root_name = vision_root(key[0]) or key[0]
            folder = shared / root_name
            marker = folder / READY_NAME
            if marker.is_file() and marker.read_text(encoding='utf-8').strip() == origin:
                announce(study, 'make', f'shared {key[0]} cached')
                continue
            endpoint = endpoint_for(key[0], origin)
            detail = f'shared {key[0]} download {origin} {endpoint.location}'
            announce(study, 'make', detail)
            DataFactory.build(DataConfig.from_mapping(mapping), shared, origin=origin)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(f'{origin}\n', encoding='utf-8')
            announce(study, 'make', f'shared {key[0]} ready')
            names.append(key[0])
    finally:
        if previous is None:
            os.environ.pop('TQDM_DISABLE', None)
        else:
            os.environ['TQDM_DISABLE'] = previous
    return names


def _shared_data_mapping(cfg: dict[str, Any]) -> dict[str, Any] | None:
    data = cfg.get('data')
    if not isinstance(data, dict) or not data.get('name'):
        return None
    if data.get('source') in (None, '', 'stub'):
        return None
    mapping = dict(data)
    inner = dict(mapping.get('config') or {})
    inner.pop('train_size', None)
    mapping['config'] = inner
    return mapping
