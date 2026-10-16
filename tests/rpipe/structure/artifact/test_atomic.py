from pathlib import Path
import pytest

from rpipe.structure.artifact._atomic import atomic_replace, atomic_write_text

pytestmark = [pytest.mark.unit, pytest.mark.content, pytest.mark.p1,
              pytest.mark.structure_layer, pytest.mark.module_artifact,
              pytest.mark.cost(cost_class='c1'),
              pytest.mark.result_type('categorical', detail='summary')]


@pytest.mark.parametrize('winerror', [5, 32, 33])
def test_atomic_replace_transient_windows_denial_eventually_commits(tmp_path, monkeypatch, winerror):
    source, target = tmp_path / 'new', tmp_path / 'old'
    source.write_text('new', encoding='utf-8')
    target.write_text('old', encoding='utf-8')
    replace, pauses, calls = Path.replace, [], []

    def held(path, destination):
        calls.append(path)
        if len(calls) <= 2:
            assert target.read_text(encoding='utf-8') == 'old'
            error = PermissionError('file temporarily held')
            error.winerror = winerror
            raise error
        return replace(path, destination)

    monkeypatch.setattr(Path, 'replace', held)
    monkeypatch.setattr('rpipe.structure.artifact._atomic.time.sleep', pauses.append)
    assert atomic_replace(source, target) == target
    assert target.read_text(encoding='utf-8') == 'new' and not source.exists()
    assert len(calls) == 3 and pauses == [0.05, 0.1]


@pytest.mark.parametrize('winerror,expected_calls', [(5, 6), (None, 1), (112, 1)])
def test_atomic_write_permanent_failure_preserves_target_and_cleans_stage(tmp_path, monkeypatch, winerror, expected_calls):
    target, calls = tmp_path / 'old', []
    target.write_text('old', encoding='utf-8')
    error = PermissionError('permanent failure')
    error.winerror = winerror

    def denied(path, destination):
        calls.append(path)
        raise error

    monkeypatch.setattr(Path, 'replace', denied)
    monkeypatch.setattr('rpipe.structure.artifact._atomic.time.sleep', lambda _: None)
    with pytest.raises(PermissionError) as caught:
        atomic_write_text(target, 'new')
    assert caught.value is error and len(calls) == expected_calls
    assert target.read_text(encoding='utf-8') == 'old'
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize('existing', [False, True])
@pytest.mark.parametrize('error_type', [OSError, KeyboardInterrupt])
def test_partial_text_write_failure_cleans_only_its_stage(tmp_path, monkeypatch, stream_close_fault, existing, error_type):
    target = tmp_path / 'result.txt'
    if existing:
        target.write_text('old', encoding='utf-8')
    peer = tmp_path / 'result.txt.other.tmp'
    peer.write_text('another writer', encoding='utf-8')
    error = error_type('injected partial text write failure')
    stream_close_fault(lambda path, mode: path.suffix == '.tmp' and mode == 'w', error, None)
    with pytest.raises(error_type) as caught:
        atomic_write_text(target, 'new')
    assert caught.value is error
    assert peer.read_text(encoding='utf-8') == 'another writer'
    assert set(tmp_path.iterdir()) == ({target, peer} if existing else {peer})
    if existing:
        assert target.read_text(encoding='utf-8') == 'old'


@pytest.mark.parametrize('existing', [False, True])
def test_text_encoding_failure_preserves_target_and_peer(tmp_path, existing):
    target, peer = tmp_path / 'result.txt', tmp_path / 'peer.tmp'
    if existing:
        target.write_bytes(b'original')
    peer.write_bytes(b'peer staging')
    with pytest.raises(UnicodeEncodeError):
        atomic_write_text(target, 'plain\u4e2d\u6587', encoding='ascii')
    assert set(tmp_path.iterdir()) == ({target, peer} if existing else {peer})
    assert peer.read_bytes() == b'peer staging'
    if existing:
        assert target.read_bytes() == b'original'


def test_atomic_text_success_replaces_target_and_keeps_peer_stage(tmp_path):
    target = tmp_path / 'result.txt'
    target.write_text('old', encoding='utf-8')
    peer = tmp_path / 'result.txt.other.tmp'
    peer.write_text('another writer', encoding='utf-8')
    assert atomic_write_text(target, 'new') == target
    assert target.read_text(encoding='utf-8') == 'new'
    assert peer.read_text(encoding='utf-8') == 'another writer'
    assert set(tmp_path.iterdir()) == {target, peer}


# Exact protocol exemptions: append offsets, checkpoint recovery and lock inode
# ownership cannot use whole-file text replacement (design section 5.3).
_TEXT_PROTOCOLS = {
    ('structure/artifact/_atomic.py', 'atomic_write_text'): 'the common text publisher',
    ('structure/system/logger.py', 'Logger.__init__'): 'create legacy append log',
    ('structure/system/logger.py', 'Logger._emit'): 'append and flush log records',
    ('structure/algorithm/tracker.py', 'AlgorithmTracker.__init__'): 'create legacy append ledger',
    ('structure/algorithm/tracker.py', 'AlgorithmTracker.begin_run'): 'restore byte-offset prefix',
    ('structure/algorithm/tracker.py', 'AlgorithmTracker.append_jsonl'): 'append curve records',
}


def _direct_text_writes(source):
    import ast

    tree = ast.parse(source)
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update((a.asname or a.name, a.name) for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            aliases.update((a.asname or a.name, f'{node.module}.{a.name}') for a in node.names)
    found, scope = [], []

    class Writes(ast.NodeVisitor):
        def visit_ClassDef(self, node):
            self.visit_scope(node)

        def visit_FunctionDef(self, node):
            self.visit_scope(node)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_scope(self, node):
            scope.append(node.name)
            self.generic_visit(node)
            scope.pop()

        def visit_Call(self, node):
            name = ast.unparse(node.func)
            root, _, rest = name.partition('.')
            name = aliases.get(root, root) + ('.' + rest if rest else '')
            direct = isinstance(node.func, ast.Attribute) and node.func.attr == 'write_text'
            if name in ('json.dump', 'yaml.safe_dump', 'yaml.dump'):
                direct = direct or len(node.args) > 1 or any(k.arg == 'stream' for k in node.keywords)
            if name in ('open', 'builtins.open', 'io.open') or name.endswith('.open'):
                index = 1 if name in ('open', 'builtins.open', 'io.open') else 0
                mode = next((k.value for k in node.keywords if k.arg == 'mode'),
                            node.args[index] if len(node.args) > index else None)
                if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
                    direct = direct or ('b' not in mode.value and any(c in mode.value for c in 'wax+'))
            if direct:
                found.append(('.'.join(scope), node.lineno))
            self.generic_visit(node)

    Writes().visit(tree)
    return found


def test_replaceable_text_artifacts_only_use_atomic_publisher():
    import rpipe

    root = Path(rpipe.__file__).parent
    found = [(p.relative_to(root).as_posix(), function, line)
             for p in root.rglob('*.py')
             for function, line in _direct_text_writes(p.read_text(encoding='utf-8'))
             if (p.relative_to(root).as_posix(), function) not in _TEXT_PROTOCOLS]
    assert not found, found


@pytest.mark.parametrize('source,blocked', [
    ('from pathlib import Path as P; P("out").write_text("x")', True),
    ('from builtins import open as writer; writer("out", "w")', True),
    ('import io as stream; stream.open("out", mode="a")', True),
    ('import json as j; j.dump({}, handle)', True),
    ('import yaml as y; y.safe_dump({}, stream=handle)', True),
    ('path.open("rb"); text.replace("a", "b")', False),
    ('atomic_write_text(path, "x"); yaml.safe_dump({})', False),
])
def test_text_architecture_guard_detects_aliases_without_banning_read_or_encoding(source, blocked):
    assert bool(_direct_text_writes(source)) is blocked


@pytest.mark.parametrize('leaf', ['index.json', 'scripts/jobs.json', 'scripts/launch.sh', 'scripts/launch.ps1'])
def test_public_text_publishers_preserve_target_when_commit_is_refused(tmp_path, monkeypatch, leaf):
    from rpipe.structure.artifact import _atomic
    from rpipe.structure.artifact.index import write_index
    from rpipe.structure.make.schedule import write_launch_scripts

    target = tmp_path / leaf
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b'old evidence')
    peer = target.parent / 'peer.tmp'
    peer.write_bytes(b'peer staging')
    original = _atomic.atomic_replace
    error = OSError('refused publication')

    def refused(source, dest):
        if Path(dest) == target:
            assert target.read_bytes() == b'old evidence'
            raise error
        return original(source, dest)

    monkeypatch.setattr(_atomic, 'atomic_replace', refused)
    with pytest.raises(OSError) as caught:
        if leaf == 'index.json':
            write_index(tmp_path, {'study': 'new', 'experiments': []})
        else:
            write_launch_scripts(tmp_path, [], round_size=1, num_gpus=0)
    assert caught.value is error
    assert target.read_bytes() == b'old evidence'
    assert peer.read_bytes() == b'peer staging'
    assert set(target.parent.glob('*.tmp')) == {peer}


def test_index_encoding_failure_does_not_truncate_existing_plan(tmp_path):
    from rpipe.structure.artifact.index import write_index

    target = tmp_path / 'index.json'
    target.write_bytes(b'old plan')
    with pytest.raises(TypeError):
        write_index(tmp_path, {'id': 'known', 'study': 'new', 'bad': object()})
    assert target.read_bytes() == b'old plan'


@pytest.mark.parametrize('newline', [None, '\n'])
def test_atomic_text_newline_option_preserves_native_and_script_bytes(tmp_path, newline):
    import os

    target = tmp_path / 'unicode.txt'
    atomic_write_text(target, '中文\nsecond\n', newline=newline)
    assert target.read_bytes() == ('中文' + (os.linesep if newline is None else '\n')
                                  + 'second' + (os.linesep if newline is None else '\n')).encode('utf-8')


@pytest.mark.skipif(__import__('os').name == 'nt', reason='POSIX executable mode bits')
def test_atomic_text_preserves_existing_script_mode(tmp_path):
    target = tmp_path / 'launch.sh'
    target.write_text('old', encoding='utf-8')
    target.chmod(0o750)
    atomic_write_text(target, 'new', newline='\n')
    assert target.stat().st_mode & 0o777 == 0o750


def _study_sources():
    # Scan source paths only, never generated Run archives or shared inputs.
    root = Path(__file__).resolve().parents[4] / 'studies'
    for study in sorted(root.iterdir()):
        for name in ('recipe.py',):
            path = study / name
            if path.is_file():
                yield root, path
        for name in ('structure', 'flow'):
            for path in sorted((study / name).rglob('*.py')):
                yield root, path


def test_study_replaceable_text_artifacts_use_same_atomic_publisher():
    exemptions = {('main_exp/flow/execute/precision.py', 'main.execute'):
                  'live subprocess stdout/stderr stream, not replaceable report'}
    found = [(path.relative_to(root).as_posix(), function, line)
             for root, path in _study_sources()
             for function, line in _direct_text_writes(path.read_text(encoding='utf-8'))
             if (path.relative_to(root).as_posix(), function) not in exemptions]
    assert not found, found


def _probe_json_writer():
    import ast
    import json
    import os
    from typing import Any

    root = Path(__file__).resolve().parents[4]
    path = root / 'studies/main_probe/structure/algorithm/paired.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'save_json')
    namespace = dict(Path=Path, Any=Any, json=json, os=os, atomic_write_text=atomic_write_text)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['save_json']


@pytest.mark.parametrize('failure', ['partial', 'replace', 'encoding'])
def test_study_json_publisher_failure_preserves_evidence_and_peer(tmp_path, monkeypatch, stream_close_fault, failure):
    import rpipe.structure.artifact._atomic as atomic

    writer = _probe_json_writer()
    target, peer = tmp_path / 'EXECUTION.json', tmp_path / 'EXECUTION.json.writing'
    target.write_bytes(b'old evidence')
    peer.write_bytes(b'another publisher')
    error = OSError('publication refused')
    if failure == 'partial':
        stream_close_fault(lambda path, mode: path.suffix == '.tmp' and mode == 'w', error, None)
    elif failure == 'replace':
        import os
        def refused(*args):
            raise error
        monkeypatch.setattr(atomic, 'atomic_replace', refused)
        monkeypatch.setattr(Path, 'replace', refused)
        monkeypatch.setattr(os, 'replace', refused)
    payload = {'invalid': object()} if failure == 'encoding' else {'new': '中文'}
    with pytest.raises(TypeError if failure == 'encoding' else OSError):
        writer(target, payload)
    assert target.read_bytes() == b'old evidence'
    assert peer.read_bytes() == b'another publisher'
    assert set(tmp_path.iterdir()) == {target, peer}


def test_study_json_publisher_success_preserves_payload_format_and_peer(tmp_path):
    import json
    import os

    target, peer = tmp_path / 'EXECUTION.json', tmp_path / 'EXECUTION.json.writing'
    peer.write_bytes(b'another publisher')
    body = {'text': '中文', 'rows': [1, 2]}
    assert _probe_json_writer()(target, body) is None
    expected = (json.dumps(body, ensure_ascii=False, indent=2) + '\n').replace('\n', os.linesep).encode('utf-8')
    assert target.read_bytes() == expected
    assert peer.read_bytes() == b'another publisher'
    assert set(tmp_path.iterdir()) == {target, peer}


@pytest.mark.parametrize('backend', ['dynamic', 'prebound'])
def test_study_json_concurrent_publishers_keep_distinct_stages_and_peer(tmp_path, monkeypatch, backend):
    from concurrent.futures import ThreadPoolExecutor
    import json
    import os
    import threading
    import rpipe.structure.artifact._atomic as atomic

    target, peer = tmp_path / 'EXECUTION.json', tmp_path / 'EXECUTION.json.writing'
    target.write_bytes(b'old evidence')
    peer.write_bytes(b'another publisher')
    writer = _probe_json_writer()
    barrier, stages, replace = threading.Barrier(2), [], os.replace
    if backend == 'prebound':
        # Python 3.10 pathlib captures os.replace before the test patches it.
        monkeypatch.setattr(Path, 'replace', lambda source, destination: replace(source, destination) or Path(destination))
    commit, local = atomic.atomic_replace, threading.local()
    def synchronized(source, destination):
        stages.append(source)
        barrier.wait(timeout=10)
        local.publishing = True
        try:
            return commit(source, destination)
        finally:
            local.publishing = False
    def native_replace(source, destination):
        if getattr(local, 'publishing', False):
            return replace(source, destination)
        return synchronized(source, destination)
    monkeypatch.setattr(atomic, 'atomic_replace', synchronized)
    monkeypatch.setattr(os, 'replace', native_replace)
    bodies = [{'writer': 'A'}, {'writer': 'B'}]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(writer, target, body) for body in bodies]
        assert [future.result(timeout=15) for future in futures] == [None, None]
    assert len(set(stages)) == 2
    assert json.loads(target.read_text(encoding='utf-8')) in bodies
    assert peer.read_bytes() == b'another publisher'
    assert set(tmp_path.iterdir()) == {target, peer}


@pytest.mark.parametrize('kind', ['peer', 'target-alias', 'stat-refusal'])
def test_text_stage_collision_preserves_unowned_bytes(tmp_path, monkeypatch, kind):
    from types import SimpleNamespace
    from rpipe.structure.artifact import _atomic
    monkeypatch.setattr(_atomic.uuid, 'uuid4', lambda: SimpleNamespace(hex='0' * 32))
    peer = tmp_path / ('0' * 32 + '.tmp')
    target = peer if kind == 'target-alias' else tmp_path / 'target.txt'
    target.write_bytes(b'old target')
    if peer != target:
        peer.write_bytes(b'peer stage')
    before = {p: p.read_bytes() for p in [target, peer]}
    error = PermissionError('target stat refused')
    if kind == 'stat-refusal':
        stat = Path.stat

        def denied(path, *a, **kw):
            if path == target:
                raise error
            return stat(path, *a, **kw)
        monkeypatch.setattr(Path, 'stat', denied)
    with pytest.raises(PermissionError if kind == 'stat-refusal' else FileExistsError) as caught:
        _atomic.atomic_write_text(target, 'new target')
    if kind == 'stat-refusal':
        assert caught.value is error
    assert {p: p.read_bytes() for p in before} == before


def test_concurrent_nonce_collision_cannot_delete_an_active_writer_stage(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    import threading
    from rpipe.structure.artifact import _atomic

    target = tmp_path / 'target.txt'
    target.write_text('old', encoding='utf-8')
    stage = tmp_path / ('0' * 32 + '.tmp')
    monkeypatch.setattr(_atomic.uuid, 'uuid4', lambda: SimpleNamespace(hex='0' * 32))
    ready, release = threading.Event(), threading.Event()
    publish = _atomic.atomic_replace

    def hold(source, destination):
        if ready.is_set():
            raise AssertionError('colliding writer reached publication')
        ready.set()
        assert release.wait(5)
        return publish(source, destination)

    monkeypatch.setattr(_atomic, 'atomic_replace', hold)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(_atomic.atomic_write_text, target, 'first')
        try:
            assert ready.wait(5)
            with pytest.raises(FileExistsError):
                _atomic.atomic_write_text(target, 'second')
            assert stage.read_text(encoding='utf-8') == 'first'
        finally:
            release.set()
        assert first.result() == target
    assert target.read_text(encoding='utf-8') == 'first'
    assert not stage.exists()


def test_owned_text_stage_chmod_refusal_preserves_old_target_and_peer(tmp_path, monkeypatch):
    from rpipe.structure.artifact import _atomic

    target, peer = tmp_path / 'target.txt', tmp_path / 'peer.tmp'
    target.write_bytes(b'old target')
    peer.write_bytes(b'peer evidence')
    error = PermissionError('stage mode update refused')
    stages = []

    def refused(path, mode, *args, **kwargs):
        stages.append(path)
        assert path not in (target, peer) and path.read_bytes() == b'new target'
        raise error

    monkeypatch.setattr(Path, 'chmod', refused)
    with pytest.raises(PermissionError) as caught:
        _atomic.atomic_write_text(target, 'new target')
    assert caught.value is error and len(stages) == 1
    assert target.read_bytes() == b'old target' and peer.read_bytes() == b'peer evidence'
    assert set(tmp_path.iterdir()) == {target, peer}


@pytest.mark.parametrize('writer', ['text', 'config', 'result'])
@pytest.mark.parametrize('readonly', [False, True], ids=['writable-control', 'native-readonly'])
def test_atomic_text_native_permissions_cleanup_owned_stage(tmp_path, monkeypatch, writer, readonly):
    import os
    from rpipe.structure.artifact import _atomic
    from rpipe.structure.artifact.config import write_config
    from rpipe.structure.artifact.result import write_result

    if readonly and os.name != 'nt':
        pytest.skip('Windows read-only replacement and unlink semantics')
    target, peer = tmp_path / 'target.txt', tmp_path / 'peer.tmp'
    target.write_bytes(b'old evidence')
    peer.write_bytes(b'other writer evidence')
    peer.chmod(0o444)
    if readonly:
        target.chmod(0o444)
    before = {p: (p.read_bytes(), p.stat().st_mode) for p in (target, peer)}
    monkeypatch.setattr(_atomic.time, 'sleep', lambda _: None)
    def write():
        if writer == 'config':
            return write_config(target, {'new': 'configuration'})
        if writer == 'result':
            return write_result(target, {'status': 'succeeded', 'control': {}, 'metrics': {}, 'paths': {}})
        return _atomic.atomic_write_text(target, 'new text')
    if readonly:
        with pytest.raises(PermissionError):
            write()
        assert (target.read_bytes(), target.stat().st_mode) == before[target]
    else:
        assert write() == target and target.read_bytes() != before[target][0]
    assert (peer.read_bytes(), peer.stat().st_mode) == before[peer]
    assert set(tmp_path.iterdir()) == {target, peer}


@pytest.mark.parametrize('cleanup', ['recover', 'chmod-denied', 'unlink-denied', 'io-denied'])
def test_atomic_text_permission_cleanup_is_owned_bounded_and_best_effort(tmp_path, monkeypatch, cleanup):
    from rpipe.structure.artifact import _atomic

    target, peer = tmp_path / 'target.txt', tmp_path / 'peer.tmp'
    target.write_bytes(b'old target')
    peer.write_bytes(b'peer bytes')
    target.chmod(0o444)
    peer.chmod(0o444)
    before = {p: (p.read_bytes(), p.stat().st_mode) for p in (target, peer)}
    error = PermissionError('publication refused')
    stages, removals, modes = [], [], []
    unlink, chmod = Path.unlink, Path.chmod
    def refuse(stage, destination):
        stages.append(stage)
        raise error
    def remove(path, *args, **kwargs):
        assert path not in (target, peer)
        removals.append(path)
        if cleanup == 'io-denied':
            raise OSError('cleanup IO failure')
        if cleanup == 'unlink-denied' or not path.stat().st_mode & 0o200:
            raise PermissionError('read-only staging')
        return unlink(path, *args, **kwargs)
    def mode(path, value, *args, **kwargs):
        assert path not in (target, peer)
        modes.append(value)
        if cleanup == 'chmod-denied' and value == 0o600:
            raise PermissionError('cleanup mode refused')
        return chmod(path, value, *args, **kwargs)
    monkeypatch.setattr(_atomic, 'atomic_replace', refuse)
    monkeypatch.setattr(Path, 'unlink', remove)
    monkeypatch.setattr(Path, 'chmod', mode)
    with pytest.raises(PermissionError) as caught:
        _atomic.atomic_write_text(target, 'new candidate')
    assert caught.value is error and len(stages) == 1
    assert {p: (p.read_bytes(), p.stat().st_mode) for p in before} == before
    if cleanup == 'recover':
        assert set(tmp_path.iterdir()) == {target, peer}
        assert len(removals) == 2 and modes[-1] == 0o600
    else:
        assert stages[0].read_bytes() == b'new candidate'
        assert len(removals) == (2 if cleanup == 'unlink-denied' else 1)


@pytest.mark.parametrize('existing', [False, True])
def test_atomic_text_readonly_collision_never_changes_unowned_stage(tmp_path, monkeypatch, existing):
    from types import SimpleNamespace
    from rpipe.structure.artifact import _atomic

    target, stage = tmp_path / 'target.txt', tmp_path / 'occupied.tmp'
    stage.write_bytes(b'other writer candidate')
    stage.chmod(0o444)
    if existing:
        target.write_bytes(b'old evidence')
    before = {p: (p.read_bytes(), p.stat().st_mode) for p in tmp_path.iterdir()}
    monkeypatch.setattr(_atomic.uuid, 'uuid4', lambda: SimpleNamespace(hex='occupied'))
    with pytest.raises(FileExistsError):
        _atomic.atomic_write_text(target, 'new candidate')
    assert set(tmp_path.iterdir()) == set(before)
    assert {p: (p.read_bytes(), p.stat().st_mode) for p in before} == before


@pytest.mark.parametrize('existing', [False, True])
@pytest.mark.parametrize('outcome', ['error', 'cancel', 'normal', 'programming'])
def test_atomic_stream_close_preserves_write_error_and_cancel(tmp_path, stream_close_fault, existing, outcome):
    import errno
    target, peer = tmp_path / 'result.txt', tmp_path / 'peer.tmp'
    if existing:
        target.write_bytes(b'old target')
    peer.write_bytes(b'peer evidence')
    primary = KeyboardInterrupt('cancel text write') if outcome == 'cancel' else None if outcome == 'normal' else OSError(errno.ENOSPC, 'partial text write')
    secondary = RuntimeError('unexpected close error') if outcome == 'programming' else OSError(errno.EIO, 'close IO')
    expected = secondary if outcome in ('normal', 'programming') else primary
    handles, calls = stream_close_fault(lambda path, mode: path.suffix == '.tmp' and mode == 'w', primary, secondary)
    with pytest.raises(type(expected)) as caught:
        atomic_write_text(target, 'new 中文\n')
    assert caught.value is expected
    assert len(handles) == 1 and handles[0].closed and calls.count('close') == 1
    assert set(tmp_path.iterdir()) == ({target, peer} if existing else {peer})
    assert peer.read_bytes() == b'peer evidence'
    if existing:
        assert target.read_bytes() == b'old target'


@pytest.mark.parametrize('existing', [False, True])
def test_atomic_encoding_error_survives_close_io(tmp_path, stream_close_fault, existing):
    target = tmp_path / 'result.txt'
    if existing:
        target.write_bytes(b'old target')
    handles, calls = stream_close_fault(lambda path, mode: path.suffix == '.tmp' and mode == 'w', None, OSError('close IO'))
    with pytest.raises(UnicodeEncodeError):
        atomic_write_text(target, '中文', encoding='ascii')
    assert len(handles) == 1 and handles[0].closed and calls.count('close') == 1
    assert list(tmp_path.iterdir()) == ([target] if existing else [])
    if existing:
        assert target.read_bytes() == b'old target'


@pytest.fixture(params=['main_exp', 'main_probe'])
def source_archive_exporter(tmp_path, request):
    """Execute actual exporter AST with one synthetic Git blob and no historical imports, data or bootstrap."""
    import ast
    import contextlib
    import copy
    import hashlib
    import importlib
    import json
    import os
    import sys
    from types import SimpleNamespace

    root = Path(__file__).resolve().parents[4]
    payload = b'# synthetic source blob\n'
    oid = hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).hexdigest()
    tree = f'100644 blob {oid}\tsrc/evidence.py\0'.encode()
    def git(args, **kwargs):
        if args[1] == 'ls-tree':
            return tree
        if args[1:3] == ['cat-file', '--batch']:
            assert kwargs['input'] == (oid + '\n').encode()
            return f'{oid} blob {len(payload)}\n'.encode() + payload + b'\n'
        assert args[1:3] == ['cat-file', 'blob'] and args[3] == oid
        return payload
    namespace = dict(Path=Path, contextlib=contextlib, copy=copy, hashlib=hashlib,
                     importlib=importlib, json=json, os=os, sys=sys,
                     subprocess=SimpleNamespace(check_output=git),
                     bootstrap=lambda: None, cwd=lambda _: contextlib.nullcontext(),
                     MAIN='fixture-ref', ROOT=tmp_path, COMMIT='fixture-ref', REPO=tmp_path,
                     ARCHIVE_ROOT=tmp_path / 'archive')
    main_exp = request.param == 'main_exp'
    path = root / ('studies/main_exp/structure/api/__init__.py' if main_exp
                   else 'studies/main_probe/structure/algorithm/paired.py')
    names = ('export_source',) if main_exp else ('save_json', 'MainArchive')
    source_tree = ast.parse(path.read_text(encoding='utf-8'))
    definitions = [n for n in source_tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *definitions], type_ignores=[])),
                 str(path), 'exec'), namespace)
    if main_exp:
        invoke = namespace['export_source']
        archive = namespace['ARCHIVE_ROOT']
        manifest = archive / 'SOURCE_MANIFEST.json'
    else:
        namespace['MainArchive'].ROOTS = ()
        invoke = lambda: namespace['MainArchive'](tmp_path)
        archive = tmp_path / 'source-main-98648f3'
        manifest = tmp_path / 'ARCHIVE_MANIFEST.json'
    target = archive / 'src/evidence.py'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_bytes(b'previous manifest evidence')
    return invoke, target, manifest, payload


@pytest.mark.parametrize('case', ['fresh', 'identical', 'different', 'race-different', 'race-identical'])
def test_source_archive_creation_does_not_overwrite_racing_owner(source_archive_exporter, monkeypatch, case):
    """Both source exporters preserve a real competing thread's bytes and previous manifest after losing creation ownership."""
    import hashlib
    import json
    import threading

    invoke, target, manifest, payload = source_archive_exporter
    peer = payload if case in ('identical', 'race-identical') else b'peer partial source evidence'
    if case in ('identical', 'different'):
        target.parent.mkdir(parents=True)
        target.write_bytes(peer)
    old_manifest = manifest.read_bytes()
    original = Path.exists
    errors = []
    def exists(path):
        observed = original(path)
        if path == target and case.startswith('race-') and not observed:
            def competing_writer():
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(peer)
                except BaseException as exc:
                    errors.append(exc)
            worker = threading.Thread(target=competing_writer)
            worker.start()
            worker.join(5)
            assert not worker.is_alive() and not errors
        return observed
    with monkeypatch.context() as fault:
        fault.setattr(Path, 'exists', exists)
        if case == 'different' or case.startswith('race-'):
            with pytest.raises(ValueError if case == 'different' else FileExistsError):
                invoke()
        else:
            invoke()
    if case == 'different' or case.startswith('race-'):
        assert target.read_bytes() == peer and manifest.read_bytes() == old_manifest
        if case == 'race-identical':
            invoke()  # Retry after the winner is complete reuses identical source bytes.
            assert target.read_bytes() == payload
    else:
        assert target.read_bytes() == payload
        row = json.loads(manifest.read_text(encoding='utf-8'))['files'][0]
        assert row['path'] == 'src/evidence.py'
        assert row['sha256'] == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize('primary_type', [OSError, RuntimeError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize('close_fails', [False, True])
def test_source_archive_partial_write_keeps_primary_and_diagnostic_bytes(source_archive_exporter, stream_close_fault, primary_type, close_fails):
    """Exclusive raw source creation retains failed owned bytes and the primary error even if closing raises IO."""
    invoke, target, manifest, payload = source_archive_exporter
    original = primary_type('source write failed first')
    secondary = OSError('source close failed second') if close_fails else None
    handles, _ = stream_close_fault(lambda path, mode: path == target and mode in ('wb', 'xb'), original, secondary)
    old_manifest = manifest.read_bytes()
    with pytest.raises(primary_type) as caught:
        invoke()
    assert caught.value is original
    assert target.read_bytes() == payload[:3]
    assert manifest.read_bytes() == old_manifest
    assert len(handles) == 1 and handles[0].closed


def test_source_archive_successful_write_close_error_is_strict(source_archive_exporter, stream_close_fault):
    """A complete blob followed by close IO is still a failed export; never publish a new manifest early."""
    invoke, target, manifest, payload = source_archive_exporter
    error = OSError('source close refused')
    handles, _ = stream_close_fault(lambda path, mode: path == target and mode in ('wb', 'xb'), None, error)
    old_manifest = manifest.read_bytes()
    with pytest.raises(OSError) as caught:
        invoke()
    assert caught.value is error and target.read_bytes() == payload
    assert manifest.read_bytes() == old_manifest
    assert len(handles) == 1 and handles[0].closed
