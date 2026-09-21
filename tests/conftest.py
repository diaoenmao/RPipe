"""TESTING.md: require three-axis markers and persist per-case results."""

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

REPO = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO / '.tmp' / 'test-results'


def _marker_names(item: pytest.Item) -> set[str]:
    return {marker.name for marker in item.iter_markers()}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    errors: list[str] = []
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
    if errors:
        raise pytest.UsageError('TESTING.md marker contract:\n' + '\n'.join(errors))


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
    status = _status(report)
    failure = None
    if report.failed:
        failure = {
            'longrepr': str(report.longrepr)[:4000],
            'when': report.when,
        }
    event = {
        'run_id': run_id,
        'test_id': item.nodeid,
        'level': level,
        'type': kind,
        'priority': priority,
        'tags': sorted(names),
        'target': item.location[0],
        'status': status,
        'duration_ms': int((report.duration or 0) * 1000),
        'started_at': datetime.now(timezone.utc).isoformat(),
        'failure': failure,
        'artifacts': [],
    }
    with (out_dir / 'events.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + '\n')


def _status(report: pytest.TestReport) -> str:
    if report.skipped:
        return 'skipped'
    if report.passed:
        if getattr(report, 'wasxfail', False):
            return 'xpass'
        return 'passed'
    if getattr(report, 'wasxfail', False):
        return 'xfail'
    return 'failed'
