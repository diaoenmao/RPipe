"""Local test manifests identify ordinary, packed and linked Git checkouts."""
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import conftest as records

pytestmark = [pytest.mark.unit, pytest.mark.content, pytest.mark.p1,
              pytest.mark.cost(cost_class='c2'),
              pytest.mark.result_type('categorical', detail='summary')]


@pytest.mark.skipif(shutil.which('git') is None, reason='requires local Git for real metadata layouts')
@pytest.mark.parametrize('layout', ['loose', 'packed', 'detached', 'worktree'])
def test_manifest_git_sha_handles_real_git_layouts(tmp_path, monkeypatch, layout):
    repo = tmp_path / 'repo'
    hooks = tmp_path / 'hooks'
    hooks.mkdir()
    def git(*args):
        return subprocess.check_output(
            ['git', '-c', f'core.hooksPath={hooks}', '-c', 'commit.gpgsign=false',
             '-c', 'user.name=metadata-test', '-c', 'user.email=metadata@example.invalid', *args],
            cwd=repo if repo.exists() else tmp_path, text=True, stderr=subprocess.PIPE).strip()
    git('init', '--quiet', str(repo))
    git('commit', '--quiet', '--allow-empty', '-m', 'metadata fixture')
    expected = git('rev-parse', 'HEAD')
    root = repo
    if layout == 'packed':
        git('pack-refs', '--all')
    elif layout == 'detached':
        git('checkout', '--quiet', '--detach')
    elif layout == 'worktree':
        root = tmp_path / 'linked'
        git('worktree', 'add', '--quiet', '--detach', str(root))
        assert (root / '.git').is_file()
    monkeypatch.setattr(records, 'REPO', root)
    assert records._git_sha() == expected


def test_snapshot_without_git_never_attributes_parent_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(records, 'REPO', tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail('a snapshot must not search its parent checkout')
    monkeypatch.setattr(records, 'subprocess', SimpleNamespace(run=forbidden), raising=False)
    assert records._git_sha() is None


@pytest.mark.parametrize('error', [FileNotFoundError('git missing'),
                                  subprocess.TimeoutExpired('git', 5),
                                  subprocess.CalledProcessError(128, 'git')])
def test_git_fallback_unavailable_is_optional_metadata(tmp_path, monkeypatch, error):
    (tmp_path / '.git').write_text('gitdir: absent')
    monkeypatch.setattr(records, 'REPO', tmp_path)
    def unavailable(*args, **kwargs):
        assert kwargs['timeout'] == 5
        raise error
    monkeypatch.setattr(records, 'subprocess', SimpleNamespace(run=unavailable), raising=False)
    assert records._git_sha() is None
