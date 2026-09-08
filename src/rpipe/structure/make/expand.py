"""Expand a Study declaration into Run patches."""

from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any

import yaml


def set_dotted(mapping: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split('.')
    cur: dict[str, Any] = mapping
    for key in keys[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[keys[-1]] = value


def cartesian(axes: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not axes:
        return [{}]
    keys = list(axes.keys())
    values = [axes[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def format_description(template: str, patch: dict[str, Any], experiment: str) -> str:
    class _Safe(dict):
        def __missing__(self, key: str) -> str:
            return '{' + key + '}'

    flat = {'experiment': experiment}
    for dotted, value in patch.items():
        if dotted in ('description', 'tags'):
            continue
        flat[dotted] = value
        flat[str(dotted).split('.')[-1]] = value
        flat[str(dotted).replace('.', '_')] = value
    try:
        return template.format_map(_Safe(flat))
    except Exception:
        return template


def match_when(axis_flat: dict[str, Any], when: dict[str, Any]) -> bool:
    for key, expected in when.items():
        if axis_flat.get(key) != expected:
            return False
    return True


def axis_keys_from_study(study: dict[str, Any]) -> list[str]:
    raw = dict(study.get('axes') or {})
    raw.pop('seed', None)
    return list(raw.keys())


def expand_patches(study: dict[str, Any]) -> list[dict[str, Any]]:
    """``axes`` → Experiment cells; ``seeds`` → Runs under each cell."""
    fixed = dict(study.get('fixed') or {})
    raw_axes = dict(study.get('axes') or {})
    axis_seed_values = raw_axes.pop('seed', None)
    axes = raw_axes
    tag_rules = list(study.get('tags') or [])
    exp_name = ''
    exp = study.get('experiment')
    if isinstance(exp, dict):
        exp_name = str(exp.get('name') or '')
    elif isinstance(exp, str):
        exp_name = exp
    desc_t = str(study.get('run_description') or '{experiment} seed={seed}')

    if study.get('seeds') is not None:
        seeds = list(study['seeds'])
    elif axis_seed_values is not None:
        seeds = list(axis_seed_values)
    elif 'seed' in fixed:
        seeds = [fixed['seed']]
    else:
        seeds = [0]

    fixed_no_seed = {k: v for k, v in fixed.items() if k != 'seed'}

    patches: list[dict[str, Any]] = []
    for combo in cartesian(axes):
        for seed in seeds:
            patch: dict[str, Any] = copy.deepcopy(fixed_no_seed)
            axis_flat: dict[str, Any] = {}
            for dotted, value in combo.items():
                set_dotted(patch, dotted, value)
                axis_flat[dotted] = value
            patch['seed'] = seed
            axis_flat['seed'] = seed
            tags: list[str] = []
            for rule in tag_rules:
                if not isinstance(rule, dict):
                    continue
                when = dict(rule.get('when') or {})
                when_no_seed = {k: v for k, v in when.items() if k != 'seed'}
                if match_when(axis_flat, when_no_seed) and (
                    'seed' not in when or axis_flat.get('seed') == when.get('seed')
                ):
                    tags.extend(list(rule.get('tags') or []))
            if tags:
                patch['tags'] = tags
            patch['description'] = format_description(
                desc_t, {**axis_flat, **patch}, exp_name
            )
            patches.append(patch)
    return patches


def load_study_yaml(study_dir: Path) -> dict[str, Any]:
    path = study_dir / 'study.yaml'
    if not path.is_file():
        raise FileNotFoundError(f'missing study.yaml: {path}')
    data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    if not isinstance(data, dict):
        raise ValueError(f'study.yaml must be a mapping: {path}')
    return data
