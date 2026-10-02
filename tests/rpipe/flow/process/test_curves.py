import json
from pathlib import Path

import pytest

from rpipe.flow.process.curves import _history

pytestmark = [
    pytest.mark.unit,
    pytest.mark.content,
    pytest.mark.p1,
    pytest.mark.flow_layer,
    pytest.mark.module_process,
    pytest.mark.cost(cost_class='c1'),
    pytest.mark.result_type('categorical', detail='summary'),
]


def test_history_prefers_all_report_observations_over_sparse_history(tmp_path: Path):
    tracker = tmp_path / 'runs' / 'run' / 'assets' / 'tracker'
    tracker.mkdir(parents=True)
    (tracker / 'tracker_state.json').write_text(
        json.dumps({'splits': {'train': {'Accuracy': {'history': [99.0]}}}}), encoding='utf-8'
    )
    train = [0.5] + [float(i * 5) for i in range(1, 12)] + [55.0]
    test = [float(i + 20) for i in range(12)]
    rows = []
    for i, value in enumerate(train):
        rows.append({'step': i, 'split': 'train', 'name': 'Accuracy', 'mean': value})
        rows.append({'step': i, 'split': 'train', 'name': 'Loss', 'mean': 2.0})
        if i < len(test):
            rows.append({'step': i, 'split': 'test', 'name': 'Accuracy', 'mean': test[i]})
    (tracker / 'scalars.jsonl').write_text(
        '\n'.join(json.dumps(row) for row in rows) + '\n', encoding='utf-8'
    )

    # Keep the real 0.5% and repeated terminal observation, rather than inventing units or steps.
    assert _history(tmp_path, 'run', 'train', 'Accuracy') == train
    assert _history(tmp_path, 'run', 'test', 'Accuracy') == test


def test_history_skips_invalid_jsonl_rows_and_keeps_valid_observations(tmp_path: Path):
    tracker = tmp_path / 'runs' / 'run' / 'assets' / 'tracker'
    tracker.mkdir(parents=True)
    rows = [
        b'{"split":"train","name":"Accuracy","mean":0.5}',
        b'{',
        b'[]',
        b'null',
        b'\xff',
        b'{"split":"train","name":"Accuracy","mean":true}',
        b'{"split":"train","name":"Accuracy","mean":"50"}',
        b'{"split":"train","name":"Accuracy","mean":NaN}',
        b'{"split":"train","name":"Accuracy","mean":Infinity}',
        b'{"split":"train","name":"Accuracy","mean":75.0}',
        b'{"split":"train","name":"Accuracy","mean":',
    ]
    (tracker / 'scalars.jsonl').write_bytes(b'\n'.join(rows))

    assert _history(tmp_path, 'run', 'train', 'Accuracy') == [0.5, 75.0]


@pytest.mark.parametrize(
    'jsonl_bytes',
    [None, b'', b'{', b'{"split":"test","name":"Loss","mean":2.0}\n'],
    ids=['missing', 'empty', 'invalid', 'other-metric'],
)
def test_history_falls_back_when_jsonl_has_no_matching_values(tmp_path: Path, jsonl_bytes):
    tracker = tmp_path / 'runs' / 'run' / 'assets' / 'tracker'
    tracker.mkdir(parents=True)
    (tracker / 'tracker_state.json').write_text(
        json.dumps({'splits': {'train': {'Accuracy': {'history': [0.5, 80.0]}}}}), encoding='utf-8'
    )
    if jsonl_bytes is not None:
        (tracker / 'scalars.jsonl').write_bytes(jsonl_bytes)

    assert _history(tmp_path, 'run', 'train', 'Accuracy') == [0.5, 80.0]
