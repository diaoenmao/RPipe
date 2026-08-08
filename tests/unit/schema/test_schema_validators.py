"""Unit tests for rpipe.schema (mirrors src/rpipe/schema)."""

from __future__ import annotations

import copy

import pytest

from rpipe.schema import (
    RESULT_BLOB_SCHEMA,
    RUN_MANIFEST_SCHEMA,
    assert_valid_result_blob,
    assert_valid_run_manifest,
    validate_result_blob,
    validate_run_manifest,
)


def _valid_manifest():
    return {
        'schema': 'rpipe.run_manifest.v1',
        'generated_at': '2026-08-08T21:00:00',
        'suite': {
            'name': 'smoke',
            'description': 'x',
            'data_names': ['MNIST'],
            'model_names': ['linear'],
            'num_experiments': 1,
            'init_seed': 0,
            'hyper': {},
        },
        'run': {'stages': ['prepare'], 'device': 'cpu', 'cwd': '/tmp'},
        'artifacts': {
            'result_paths': [],
            'plot_paths': [],
            'excel': [],
            'processed_result': 'output/result/processed_result',
            'stats_dir': 'output/stats',
            'exp_dir': 'output/exp',
        },
        'notes': [],
        'report_hint': 'hint',
    }


def _valid_result_blob():
    return {
        'schema': 'rpipe.result_blob.v1',
        'cfg': {
            'tag': '0_MNIST_linear',
            'control_name': 'MNIST_linear',
            'data_name': 'MNIST',
            'model_name': 'linear',
            'seed': 0,
        },
        'logger': {
            'train': {'mean': {}, 'history': {}},
            'test': {'mean': {'test/Accuracy': 70.0}, 'history': {}},
        },
    }


@pytest.mark.unit
@pytest.mark.content
@pytest.mark.p1
@pytest.mark.schema_layer
def test_result_blob_schema_accepts_valid():
    assert validate_result_blob(_valid_result_blob()) == []
    assert_valid_result_blob(_valid_result_blob())


@pytest.mark.unit
@pytest.mark.content
@pytest.mark.p1
@pytest.mark.schema_layer
def test_result_blob_schema_rejects_wrong_schema_id():
    bad = copy.deepcopy(_valid_result_blob())
    bad['schema'] = 'nope'
    assert validate_result_blob(bad)
    with pytest.raises(ValueError):
        assert_valid_result_blob(bad)


@pytest.mark.unit
@pytest.mark.content
@pytest.mark.p1
@pytest.mark.schema_layer
def test_manifest_schema_accepts_and_rejects():
    assert validate_run_manifest(_valid_manifest()) == []
    assert_valid_run_manifest(_valid_manifest())
    bad = copy.deepcopy(_valid_manifest())
    del bad['suite']['name']
    assert any('name' in e for e in validate_run_manifest(bad))


@pytest.mark.unit
@pytest.mark.location
@pytest.mark.p2
@pytest.mark.schema_layer
def test_schema_ids_stable():
    assert RESULT_BLOB_SCHEMA['$id'] == 'rpipe.result_blob.v1'
    assert RUN_MANIFEST_SCHEMA['$id'] == 'rpipe.run_manifest.v1'
