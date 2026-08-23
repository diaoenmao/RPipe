"""Expand study.yaml axes into Run Config patches."""

from __future__ import annotations

import itertools
from typing import Any


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


def get_dotted(mapping: dict[str, Any], dotted: str) -> Any:
    cur: Any = mapping
    for key in dotted.split('.'):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def cartesian(axes: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not axes:
        return [{}]
    keys = list(axes.keys())
    values = [axes[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def format_description(template: str, patch: dict[str, Any], experiment: str) -> str:
    """Best-effort format; unknown braces left as-is via SafeDict."""

    class _Safe(dict):
        def __missing__(self, key: str) -> str:
            return '{' + key + '}'

    flat = {'experiment': experiment}
    for dotted, value in patch.items():
        if dotted in ('description', 'tags'):
            continue
        flat[dotted] = value
        # also expose last segment for short templates
        flat[dotted.split('.')[-1]] = value
    try:
        return template.format_map(_Safe(flat))
    except Exception:
        return template


def match_when(axis_flat: dict[str, Any], when: dict[str, Any]) -> bool:
    for key, expected in when.items():
        if axis_flat.get(key) != expected:
            return False
    return True


def expand_patches(study: dict[str, Any]) -> list[dict[str, Any]]:
    """Build merge patches from study.yaml ``fixed`` + ``axes`` + ``tags``."""
    fixed = dict(study.get('fixed') or {})
    axes = dict(study.get('axes') or {})
    tag_rules = list(study.get('tags') or [])
    exp_name = ''
    exp = study.get('experiment')
    if isinstance(exp, dict):
        exp_name = str(exp.get('name') or '')
    elif isinstance(exp, str):
        exp_name = exp
    desc_t = str(study.get('run_description') or f'{exp_name} run')

    patches: list[dict[str, Any]] = []
    for combo in cartesian(axes):
        patch: dict[str, Any] = {}
        # fixed first as nested mapping mergeable with experiment_config
        for key, value in fixed.items():
            if isinstance(value, dict) and key in ('data', 'model', 'algorithm', 'system'):
                patch[key] = value
            else:
                patch[key] = value
        axis_flat: dict[str, Any] = {}
        for dotted, value in combo.items():
            set_dotted(patch, dotted, value)
            axis_flat[dotted] = value
        tags: list[str] = []
        for rule in tag_rules:
            if not isinstance(rule, dict):
                continue
            when = dict(rule.get('when') or {})
            if match_when(axis_flat, when):
                tags.extend(list(rule.get('tags') or []))
        if tags:
            patch['tags'] = tags
        patch['description'] = format_description(desc_t, {**axis_flat, **patch}, exp_name)
        patches.append(patch)
    return patches
