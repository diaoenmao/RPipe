"""Study ``index.json`` — orchestration plan written before Flow.

This is **not** a Flow phase. Flow serializes a Run via ``flow.write``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rpipe.structure.control.hashing import compute_index_id

INDEX_NAME = 'index.json'


def index_path(study_dir: Path | str) -> Path:
    return Path(study_dir) / INDEX_NAME


def _get_dotted(mapping: dict[str, Any], dotted: str) -> Any:
    cur: Any = mapping
    for key in dotted.split('.'):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _config_ref(path: Path, study_dir: Path | None) -> str:
    if study_dir is None:
        return str(path)
    try:
        return path.resolve().relative_to(Path(study_dir).resolve()).as_posix()
    except ValueError:
        return str(path)


def experiment_entries(
    *,
    configs: list[tuple[Path, dict[str, Any]]],
    axis_keys: list[str],
    study_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Group Run configs by Experiment factors (axes, excluding seed)."""
    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for path, cfg in configs:
        factors = {key: _get_dotted(cfg, key) for key in axis_keys}
        token = json.dumps(factors, sort_keys=True, default=str)
        if token not in groups:
            groups[token] = {'factors': factors, 'runs': []}
            order.append(token)
        groups[token]['runs'].append(
            {
                'id': cfg.get('id'),
                'seed': cfg.get('seed'),
                'description': cfg.get('description'),
                'tags': cfg.get('tags') or [],
                'run_dir': path.parent.name,
                'config': _config_ref(Path(path), study_dir),
            }
        )
    return [groups[key] for key in order]


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
