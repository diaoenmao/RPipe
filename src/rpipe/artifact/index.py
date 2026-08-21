"""Unified Study ``index.json`` (orchestration plan; before Flow).

Module: ``rpipe.artifact.index`` — not the Flow ``index`` phase (see CONCEPT §6.4).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rpipe.structure.control.hashing import compute_index_id

INDEX_NAME = 'index.json'


def index_path(study_dir: Path | str) -> Path:
    return Path(study_dir) / INDEX_NAME


def build_index(
    *,
    study: str,
    description: str,
    experiments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble Study index body and assign content-hash ``id``."""
    body: dict[str, Any] = {
        'description': description,
        'study': study,
        'experiments': experiments,
    }
    body['id'] = compute_index_id(body)
    return body


def write_index(study_dir: Path | str, mapping: dict[str, Any]) -> Path:
    target = index_path(study_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(mapping)
    if not payload.get('id'):
        payload['id'] = compute_index_id(payload)
    with target.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return target


def load_index(study_dir: Path | str) -> dict[str, Any]:
    with index_path(study_dir).open(encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError('Study index must be an object')
    return data
