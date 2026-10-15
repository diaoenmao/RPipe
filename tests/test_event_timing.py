"""Recorded event time identifies the actual reported phase start."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from conftest import pytest_runtest_makereport

pytestmark = [pytest.mark.unit, pytest.mark.content, pytest.mark.p1,
              pytest.mark.cost(cost_class='c1'),
              pytest.mark.result_type('categorical', detail='summary')]


@pytest.mark.parametrize('when', ['call', 'setup'])
def test_event_started_at_is_report_start_not_serialization_time(tmp_path, request, when):
    config = SimpleNamespace(_rpipe_results_dir=tmp_path, _rpipe_run_id='event-clock')
    item = SimpleNamespace(config=config, nodeid='synthetic::phase', location=('synthetic.py', 0, 'phase'),
                           iter_markers=request.node.iter_markers)
    report = SimpleNamespace(when=when, failed=when == 'setup', passed=when == 'call',
                             skipped=False, duration=1.25, start=946684800.5,
                             longrepr='fixture setup failure')
    hook = pytest_runtest_makereport(item, None)
    next(hook)
    with pytest.raises(StopIteration):
        hook.send(SimpleNamespace(get_result=lambda: report))
    event = json.loads((tmp_path / 'events.jsonl').read_text())
    assert event['started_at'] == datetime.fromtimestamp(report.start, timezone.utc).isoformat()
    assert event['duration_ms'] == event['cost']['actual_wall_time_ms'] == 1250
    assert event['status'] == ('failed' if when == 'setup' else 'passed')



def _diagnostic(kind):
    return '正常诊断🙂 ' + {'valid': 'normal', 'high': chr(0xD800),
                          'low': chr(0xDCFF), 'mixed': chr(0xD800) + '/' + chr(0xDCFF)}[kind]


@pytest.mark.parametrize('kind', ['valid', 'high', 'low', 'mixed'])
@pytest.mark.parametrize('when', ['call', 'setup'])
def test_failure_event_preserves_unicode_and_verdict(tmp_path, request, kind, when):
    text = _diagnostic(kind)
    config = SimpleNamespace(_rpipe_results_dir=tmp_path, _rpipe_run_id='unicode-event')
    item = SimpleNamespace(config=config, nodeid='synthetic::failure',
                           location=('synthetic.py', 0, 'failure'),
                           iter_markers=request.node.iter_markers)
    report = SimpleNamespace(when=when, failed=True, passed=False, skipped=False,
                             duration=1.25, start=946684800.5, longrepr=text)
    hook = pytest_runtest_makereport(item, None)
    next(hook)
    with pytest.raises(StopIteration):
        hook.send(SimpleNamespace(get_result=lambda: report))
    raw = (tmp_path / 'events.jsonl').read_bytes()
    event = json.loads(raw)
    assert event['failure'] == {'longrepr': text, 'when': when}
    assert (event['status'], event['verdict'], event['category']) == ('failed', 'fail', 'fail')
    assert event['started_at'] == datetime.fromtimestamp(report.start, timezone.utc).isoformat()
    assert event['duration_ms'] == event['cost']['actual_wall_time_ms'] == 1250
    assert '正常诊断🙂 '.encode('utf-8') in raw


@pytest.mark.parametrize('kind', ['valid', 'high', 'low', 'mixed'])
def test_manifest_preserves_unicode_arguments(tmp_path, monkeypatch, kind):
    import conftest as records
    text = _diagnostic(kind)
    monkeypatch.setattr(records, 'RESULTS_ROOT', tmp_path)
    monkeypatch.setenv('GITHUB_SHA', 'a' * 40)
    config = SimpleNamespace(invocation_params=SimpleNamespace(args=['pytest', text]),
                             option=SimpleNamespace(markexpr=text, cost_class=[]))
    records.pytest_configure(config)
    raw = (config._rpipe_results_dir / 'manifest.json').read_bytes()
    manifest = json.loads(raw)
    assert manifest['command'] == ['pytest', text]
    assert manifest['markexpr'] == text
    assert manifest['git_sha'] == 'a' * 40
    assert '正常诊断🙂 '.encode('utf-8') in raw
