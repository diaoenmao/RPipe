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
