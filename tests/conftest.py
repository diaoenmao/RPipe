"""TESTING.md: require level, objective, priority, cost, and result type."""

from __future__ import annotations

import json
import os
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

LEVELS = frozenset({'unit', 'integration', 'e2e'})
TYPES = frozenset({'location', 'content', 'physical'})
PRIORITIES = frozenset({'p1', 'p2', 'p3'})
COST_CLASSES = frozenset({'c1', 'c2', 'c3', 'c4'})
RESULT_TYPES = frozenset({'categorical', 'numeric'})
RESULT_DETAILS = frozenset({'summary', 'metrics', 'samples'})

REPO = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO / '.tmp' / 'test-results'


def _marker_names(item: pytest.Item) -> set[str]:
    return {marker.name for marker in item.iter_markers()}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        '--cost-class',
        action='append',
        default=[],
        choices=sorted(COST_CLASSES),
        help='keep tests whose cost_class is one of these values',
    )


def _one_marker(item: pytest.Item, name: str) -> pytest.Mark | None:
    found = list(item.iter_markers(name=name))
    if len(found) != 1:
        return None
    return found[0]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    errors: list[str] = []
    allowed_costs = set(config.getoption('--cost-class') or [])
    kept: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        names = _marker_names(item)
        level = names & LEVELS
        kind = names & TYPES
        priority = names & PRIORITIES
        node = item.nodeid
        if len(level) != 1:
            errors.append(f'{node}: need exactly one of {sorted(LEVELS)}, got {sorted(level)}')
        if len(kind) != 1:
            errors.append(f'{node}: need exactly one of {sorted(TYPES)}, got {sorted(kind)}')
        if len(priority) != 1:
            errors.append(f'{node}: need exactly one of {sorted(PRIORITIES)}, got {sorted(priority)}')
        if 'location' in names and 'unit' not in names:
            errors.append(f'{node}: location must be paired with unit')
        if names & {'integration', 'e2e'} and 'location' in names:
            errors.append(f'{node}: integration/e2e cannot use location')
        cost = _one_marker(item, 'cost')
        if cost is None or cost.kwargs.get('cost_class') not in COST_CLASSES:
            errors.append(f'{node}: need exactly one cost(cost_class=c1|c2|c3|c4)')
        result = _one_marker(item, 'result_type')
        detail = result.kwargs.get('detail') if result is not None else None
        if (
            result is None
            or not result.args
            or result.args[0] not in RESULT_TYPES
            or detail not in RESULT_DETAILS
        ):
            errors.append(
                f'{node}: need result_type(categorical|numeric, detail=summary|metrics|samples)'
            )
        if 'physical' in names and not (names & {'runtime', 'memory'}):
            errors.append(f'{node}: physical needs runtime or memory')
        if allowed_costs and cost is not None and cost.kwargs.get('cost_class') not in allowed_costs:
            deselected.append(item)
        else:
            kept.append(item)
    if errors:
        raise pytest.UsageError('TESTING.md marker contract:\n' + '\n'.join(errors))
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept


def pytest_configure(config: pytest.Config) -> None:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_id = f'{stamp}_{uuid.uuid4().hex[:6]}'
    out_dir = RESULTS_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    config._rpipe_run_id = run_id
    config._rpipe_results_dir = out_dir
    manifest = {
        'run_id': run_id,
        'started_at': datetime.now(timezone.utc).isoformat(),
        'cwd': str(Path.cwd()),
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'command': list(config.invocation_params.args),
        'git_sha': os.environ.get('GITHUB_SHA') or _git_sha(),
        'markexpr': getattr(config.option, 'markexpr', '') or '',
        'plan_id': 'ad_hoc',
        'batch_cost_class': None,
        'environment_id': 'local',
        'cost_class': list(getattr(config.option, 'cost_class', None) or []),
    }
    (out_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    (out_dir / 'events.jsonl').write_text('', encoding='utf-8')
    (out_dir / 'artifacts').mkdir(exist_ok=True)


def _git_sha() -> str | None:
    head = REPO / '.git' / 'HEAD'
    if not head.is_file():
        return None
    raw = head.read_text(encoding='utf-8').strip()
    if raw.startswith('ref:'):
        ref = REPO / '.git' / raw.split(' ', 1)[1].strip()
        if ref.is_file():
            return ref.read_text(encoding='utf-8').strip()
        return None
    return raw


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    report: pytest.TestReport = outcome.get_result()
    if report.when != 'call' and not (report.when == 'setup' and report.failed):
        return
    config = item.config
    out_dir: Path = getattr(config, '_rpipe_results_dir', None)
    run_id: str = getattr(config, '_rpipe_run_id', '')
    if out_dir is None:
        return
    names = _marker_names(item)
    level = next(iter(names & LEVELS), '')
    kind = next(iter(names & TYPES), '')
    priority = next(iter(names & PRIORITIES), '')
    cost_mark = _one_marker(item, 'cost')
    result_mark = _one_marker(item, 'result_type')
    cost_class = cost_mark.kwargs.get('cost_class') if cost_mark is not None else None
    result_type = result_mark.args[0] if result_mark is not None and result_mark.args else None
    result_detail = result_mark.kwargs.get('detail') if result_mark is not None else None
    status = _status(report, result_type)
    duration_ms = int((report.duration or 0) * 1000)
    failure = None
    if report.failed:
        failure = {
            'longrepr': str(report.longrepr)[:4000],
            'when': report.when,
        }
    if status == 'measured':
        verdict = None
        category = None
    elif status in {'passed', 'xpass'}:
        verdict = 'pass'
        category = 'pass'
    elif status in {'failed', 'xfail'}:
        verdict = 'fail'
        category = 'fail'
    else:
        verdict = None
        category = None
    event = {
        'run_id': run_id,
        'attempt': 1,
        'test_id': item.nodeid,
        'level': level,
        'type': kind,
        'priority': priority,
        'tags': sorted(names - LEVELS - TYPES - PRIORITIES - {'cost', 'result_type'}),
        'target': item.location[0],
        'result_type': result_type,
        'plan_id': 'ad_hoc',
        'cost_class': cost_class,
        'result_detail': result_detail,
        'environment_id': 'local',
        'verdict': verdict,
        'category': category,
        'metrics': [],
        'cost': {
            'estimated_wall_time_ms': (
                cost_mark.kwargs.get('estimated_wall_time_ms') if cost_mark is not None else None
            ),
            'actual_wall_time_ms': duration_ms,
        },
        'status': status,
        'duration_ms': duration_ms,
        'started_at': datetime.now(timezone.utc).isoformat(),
        'failure': failure,
        'artifacts': [],
    }
    with (out_dir / 'events.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + '\n')


def _status(report: pytest.TestReport, result_type: str | None) -> str:
    if report.skipped:
        return 'skipped'
    if report.passed:
        if getattr(report, 'wasxfail', False):
            return 'xpass'
        if result_type == 'numeric':
            return 'measured'
        return 'passed'
    if getattr(report, 'wasxfail', False):
        return 'xfail'
    return 'failed'


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    config = session.config
    out_dir: Path | None = getattr(config, '_rpipe_results_dir', None)
    if out_dir is None or not (out_dir / 'events.jsonl').is_file():
        return
    counts: dict[str, int] = {}
    for line in (out_dir / 'events.jsonl').read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        status = json.loads(line).get('status') or 'unknown'
        counts[status] = counts.get(status, 0) + 1
    lines = [
        '# Test report',
        '',
        f'- run_id: `{getattr(config, "_rpipe_run_id", "")}`',
        '- plan_id: `ad_hoc`',
        f'- exit_status: {exitstatus}',
        '',
        '| status | count |',
        '|---|---:|',
    ]
    for status in sorted(counts):
        lines.append(f'| {status} | {counts[status]} |')
    lines.append('')
    lines.append('数值测量完成记 `measured`，质量判定为 null。通过率的分母不含 skipped 与 measured。')
    lines.append('')
    (out_dir / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
