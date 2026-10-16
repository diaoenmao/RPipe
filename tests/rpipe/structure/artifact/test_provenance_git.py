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


@pytest.mark.skipif(shutil.which('git') is None, reason='requires Git for actual repository states')
@pytest.mark.parametrize('encoding', ['ascii', 'utf-8'])
@pytest.mark.parametrize('filename', ['plain.txt', '研究-λ.txt'], ids=['ascii-name', 'unicode-name'])
@pytest.mark.parametrize('route', ['capture', 'publish'])
def test_git_filename_decode_does_not_abort_optional_metadata(tmp_path, monkeypatch, encoding, filename, route):
    import json

    repo = tmp_path / 'repo'
    repo.mkdir()
    hooks = tmp_path / 'hooks'
    hooks.mkdir()
    real_run = subprocess.run
    def git(*args):
        result = real_run(['git', '-c', f'core.hooksPath={hooks}', '-c', 'commit.gpgsign=false',
                          '-c', 'user.name=metadata-test', '-c', 'user.email=metadata@example.invalid', *args],
                         cwd=repo, capture_output=True, timeout=10)
        assert result.returncode == 0, result.stderr
        return result.stdout
    git('init', '--quiet')
    tracked = repo / 'tracked.txt'
    tracked.write_bytes(b'fixture input')
    git('add', 'tracked.txt')
    git('commit', '--quiet', '-m', 'fixture')
    git('config', 'core.quotepath', 'false')
    head = git('rev-parse', 'HEAD').decode('ascii').strip()
    peer = repo / filename
    peer.write_bytes(b'untracked peer')
    before = {path: path.read_bytes() for path in [tracked, peer]}
    raw = git('status', '--porcelain')
    assert filename.encode('utf-8') in raw
    def decode(command, **kwargs):
        # Real Git bytes and return codes; explicitly model the parent locale decoder.
        kwargs['encoding'] = encoding
        return real_run(command, **kwargs)
    monkeypatch.setattr(provenance.subprocess, 'run', decode)
    monkeypatch.setattr(provenance, 'capture_environment', lambda: {'fixture': 'no device probing'})
    if route == 'capture':
        result = provenance.capture_git(repo)
    else:
        target = repo / 'provenance.json'
        target.write_bytes(b'previous synthetic provenance')
        assert provenance.write_provenance(repo, {}) == target
        result = json.loads(target.read_text(encoding='utf-8'))['git']
    assert result == {'head': head, 'dirty': True}
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize('stage', ['rev-parse', 'status'])
def test_git_capture_does_not_hide_unexpected_programmer_error(tmp_path, monkeypatch, stage):
    primary = RuntimeError('unexpected Git driver bug')
    def run(command, **kwargs):
        if command[1] == stage:
            raise primary
        return subprocess.CompletedProcess(command, 0, 'a' * 40 + '\n', '')
    monkeypatch.setattr(provenance.subprocess, 'run', run)
    with pytest.raises(RuntimeError) as caught:
        provenance.capture_git(tmp_path)
    assert caught.value is primary
