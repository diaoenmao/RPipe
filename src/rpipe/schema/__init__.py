"""Formal schemas for pipeline artifacts (stdlib validation, no extra deps)."""

from __future__ import annotations

from typing import Any

RESULT_BLOB_SCHEMA = {
    '$id': 'rpipe.result_blob.v1',
    'type': 'object',
    'required': ['schema', 'cfg', 'logger'],
    'properties': {
        'schema': {'const': 'rpipe.result_blob.v1'},
        'cfg': {'type': 'object', 'required': ['tag', 'control_name', 'data_name', 'model_name', 'seed']},
        'logger': {
            'type': 'object',
            'required': ['train', 'test'],
            'properties': {
                'train': {'type': 'object'},
                'test': {'type': 'object'},
            },
        },
    },
}

RUN_MANIFEST_SCHEMA = {
    '$id': 'rpipe.run_manifest.v1',
    'type': 'object',
    'required': ['schema', 'generated_at', 'suite', 'run', 'artifacts'],
    'properties': {
        'schema': {'const': 'rpipe.run_manifest.v1'},
        'generated_at': {'type': 'string'},
        'suite': {
            'type': 'object',
            'required': ['name', 'data_names', 'model_names'],
            'properties': {
                'name': {'type': 'string'},
                'description': {'type': 'string'},
                'data_names': {'type': 'array', 'items': {'type': 'string'}},
                'model_names': {'type': 'array', 'items': {'type': 'string'}},
                'num_experiments': {'type': ['integer', 'null']},
                'init_seed': {'type': ['integer', 'null']},
                'hyper': {'type': 'object'},
            },
        },
        'run': {
            'type': 'object',
            'required': ['stages', 'device'],
            'properties': {
                'stages': {'type': 'array', 'items': {'type': 'string'}},
                'device': {'type': 'string'},
                'cwd': {'type': 'string'},
            },
        },
        'artifacts': {
            'type': 'object',
            'required': ['result_paths', 'plot_paths'],
            'properties': {
                'result_paths': {'type': 'array', 'items': {'type': 'string'}},
                'plot_paths': {'type': 'array', 'items': {'type': 'string'}},
                'excel': {'type': 'array', 'items': {'type': 'string'}},
                'processed_result': {'type': 'string'},
                'stats_dir': {'type': 'string'},
                'exp_dir': {'type': 'string'},
            },
        },
        'notes': {'type': 'array', 'items': {'type': 'string'}},
        'report_hint': {'type': 'string'},
    },
}


def _type_name(value: Any) -> str:
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'boolean'
    if isinstance(value, int) and not isinstance(value, bool):
        return 'integer'
    if isinstance(value, float):
        return 'number'
    if isinstance(value, str):
        return 'string'
    if isinstance(value, list):
        return 'array'
    if isinstance(value, dict):
        return 'object'
    return type(value).__name__


def _matches_type(value: Any, expected: str | list[str]) -> bool:
    kinds = expected if isinstance(expected, list) else [expected]
    actual = _type_name(value)
    if actual in kinds:
        return True
    # JSON number accepts int
    if 'number' in kinds and actual == 'integer':
        return True
    return False


def validate_against_schema(obj: Any, schema: dict[str, Any], path: str = '$') -> list[str]:
    """Minimal JSON-Schema subset validator (type/required/const/properties/items)."""
    errors: list[str] = []
    if 'const' in schema and obj != schema['const']:
        errors.append(f'{path}: expected const {schema["const"]!r}, got {obj!r}')
        return errors
    if 'type' in schema and not _matches_type(obj, schema['type']):
        errors.append(f'{path}: expected type {schema["type"]!r}, got {_type_name(obj)}')
        return errors
    if schema.get('type') == 'object' and isinstance(obj, dict):
        for key in schema.get('required', []):
            if key not in obj:
                errors.append(f'{path}: missing required property {key!r}')
        props = schema.get('properties', {})
        for key, sub in props.items():
            if key in obj:
                errors.extend(validate_against_schema(obj[key], sub, f'{path}.{key}'))
    if schema.get('type') == 'array' and isinstance(obj, list):
        item_schema = schema.get('items')
        if item_schema:
            for i, item in enumerate(obj):
                errors.extend(validate_against_schema(item, item_schema, f'{path}[{i}]'))
    return errors


def validate_result_blob(obj: dict[str, Any]) -> list[str]:
    return validate_against_schema(obj, RESULT_BLOB_SCHEMA)


def validate_run_manifest(obj: dict[str, Any]) -> list[str]:
    return validate_against_schema(obj, RUN_MANIFEST_SCHEMA)


def assert_valid_result_blob(obj: dict[str, Any]) -> None:
    errors = validate_result_blob(obj)
    if errors:
        raise ValueError('Invalid result blob:\n- ' + '\n- '.join(errors))


def assert_valid_run_manifest(obj: dict[str, Any]) -> None:
    errors = validate_run_manifest(obj)
    if errors:
        raise ValueError('Invalid run manifest:\n- ' + '\n- '.join(errors))
