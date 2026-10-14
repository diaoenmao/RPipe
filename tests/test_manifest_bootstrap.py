"""Native test entry completes when optional local Git metadata is unreadable."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.content, pytest.mark.p1,
              pytest.mark.cost(cost_class='c2'),
              pytest.mark.result_type('categorical', detail='summary')]


@pytest.mark.parametrize('head,ci_sha', [(b'\xff', None), (b'ref:', None), (b'\xff', 'a' * 40)])
def test_native_test_entry_keeps_optional_corrupt_metadata(tmp_path, head, ci_sha):
    """Copied native tests/run.py → real pytest session → executed fixture and null/explicit SHA manifest."""
    source = Path(__file__).resolve().parent.parent
    (tmp_path / 'tests').mkdir()
    for name in ['tests/run.py', 'tests/conftest.py', 'pyproject.toml']:
        shutil.copy2(source / name, tmp_path / name)
    (tmp_path / '.git').mkdir()
    (tmp_path / '.git/HEAD').write_bytes(head)
    (tmp_path / 'tests/test_minimal.py').write_text(
        "from pathlib import Path\nimport pytest\n"
        "pytestmark=[pytest.mark.unit,pytest.mark.content,pytest.mark.p1,"
        "pytest.mark.cost(cost_class='c1'),pytest.mark.result_type('categorical',detail='summary')]\n"
        "def test_executes():\n    Path('EXECUTED').write_text('ok')\n", encoding='utf-8')
    env = os.environ.copy()
    env.pop('GITHUB_SHA', None)
    if ci_sha is not None:
        env['GITHUB_SHA'] = ci_sha
    child = subprocess.run(
        [sys.executable, 'tests/run.py', '--core', '--', 'tests/test_minimal.py',
         '--basetemp', str(tmp_path / 'p')], cwd=tmp_path, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)
    assert child.returncode == 0, child.stdout
    assert (tmp_path / 'EXECUTED').read_text() == 'ok'
    manifests = list((tmp_path / '.tmp/test-results').glob('*/manifest.json'))
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text())['git_sha'] == ci_sha
    events = [json.loads(line) for line in (manifests[0].parent / 'events.jsonl').read_text().splitlines()]
    assert len(events) == 1 and events[0]['status'] == 'passed'
