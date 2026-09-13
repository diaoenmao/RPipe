"""Materialize Study-level shared datasets once (not per Run)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rpipe.structure.artifact.config import load_config
from rpipe.structure.artifact.layout import ensure_study_layout
from rpipe.structure.data.config import DataConfig
from rpipe.structure.data.factory import DataFactory


def prepare_shared_data(study_dir: Path | str, config_paths: list[Path]) -> list[str]:
    """Download / build each unique ``data.name`` + ``source`` into ``shared/data``.

    ``train_size`` is ignored here so subset Runs reuse the same files.
    Returns names materialized on this call (cached names omitted).
    """
    study = ensure_study_layout(study_dir)
    shared = study / 'shared' / 'data'
    seen: set[tuple[str, str]] = set()
    names: list[str] = []
    previous = os.environ.get('TQDM_DISABLE')
    os.environ['TQDM_DISABLE'] = '1'
    try:
        for path in config_paths:
            mapping = _shared_data_mapping(load_config(path))
            if mapping is None:
                continue
            key = (str(mapping.get('name') or ''), str(mapping.get('source') or ''))
            if key in seen:
                continue
            seen.add(key)
            folder = shared / key[0]
            if folder.is_dir() and any(folder.rglob('*')):
                continue
            DataFactory.build(DataConfig.from_mapping(mapping), shared)
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
