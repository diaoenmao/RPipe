"""Git provenance must never report a failed status query as clean."""
import shutil
import subprocess

import pytest

from rpipe.structure.artifact import provenance

pytestmark = [pytest.mark.unit, pytest.mark.content, pytest.mark.p1,
              pytest.mark.structure_layer, pytest.mark.module_artifact,
              pytest.mark.cost(cost_class='c2'),
              pytest.mark.result_type('categorical', detail='summary')]


@pytest.mark.skipif(shutil.which('git') is None, reason='requires Git for actual repository states')
@pytest.mark.parametrize('state', ['clean', 'dirty', 'packed', 'worktree', 'corrupt-index'])
def test_capture_git_reports_real_repository_state(tmp_path, state):
    repo = tmp_path / 'repo'
    repo.mkdir()
    hooks = tmp_path / 'hooks'
    hooks.mkdir()
    def git(*args):
        return subprocess.check_output(
            ['git', '-c', f'core.hooksPath={hooks}', '-c', 'commit.gpgsign=false',
             '-c', 'user.name=provenance-test', '-c', 'user.email=provenance@example.invalid', *args],
            cwd=repo, text=True, stderr=subprocess.PIPE, timeout=10).strip()
    git('init', '--quiet')
    (repo / 'tracked.txt').write_text('fixture\n')
    git('add', 'tracked.txt')
    git('commit', '--quiet', '-m', 'fixture')
    head = git('rev-parse', 'HEAD')
    if state == 'packed':
        git('pack-refs', '--all')
    elif state == 'dirty':
        (repo / 'tracked.txt').write_text('changed\n')
    elif state == 'worktree':
        linked = tmp_path / 'linked'
        git('worktree', 'add', '--quiet', '--detach', str(linked))
        repo = linked
    elif state == 'corrupt-index':
        (repo / '.git/index').write_bytes(b'broken synthetic index')
        assert git('rev-parse', 'HEAD') == head
        status = subprocess.run(['git', 'status', '--porcelain'], cwd=repo,
                                capture_output=True, text=True, timeout=10)
        assert status.returncode != 0 and not status.stdout
    expected = {} if state == 'corrupt-index' else {'head': head, 'dirty': state == 'dirty'}
    assert provenance.capture_git(repo) == expected


@pytest.mark.parametrize('stdout', ['', ' M incomplete.txt\n'])
def test_capture_git_does_not_trust_failed_status_stdout(tmp_path, monkeypatch, stdout):
    def run(command, **kwargs):
        if command[1] == 'rev-parse':
            return subprocess.CompletedProcess(command, 0, 'a' * 40 + '\n', '')
        assert command == ['git', 'status', '--porcelain'] and kwargs['timeout'] == 30
        return subprocess.CompletedProcess(command, 128, stdout, 'status failed')
    monkeypatch.setattr(provenance.subprocess, 'run', run)
    assert provenance.capture_git(tmp_path) == {}


@pytest.mark.parametrize('stage', ['rev-parse', 'status'])
@pytest.mark.parametrize('error', ['missing', 'timeout'])
def test_capture_git_unavailable_metadata_is_optional(tmp_path, monkeypatch, stage, error):
    calls = []
    def run(command, **kwargs):
        calls.append(command[1])
        assert kwargs['timeout'] == (10 if command[1] == 'rev-parse' else 30)
        if command[1] == stage:
            if error == 'missing':
                raise FileNotFoundError('git unavailable')
            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
        return subprocess.CompletedProcess(command, 0, 'a' * 40 + '\n', '')
    monkeypatch.setattr(provenance.subprocess, 'run', run)
    assert provenance.capture_git(tmp_path) == {}
    assert calls == (['rev-parse'] if stage == 'rev-parse' else ['rev-parse', 'status'])
