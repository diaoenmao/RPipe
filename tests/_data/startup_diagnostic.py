"""Manual CPU startup/IO diagnostic; output must be new. No speed acceptance gate."""
from __future__ import annotations

import contextlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


def handles():
    if os.name != 'nt':
        return len(list(Path('/proc/self/fd').iterdir())) if Path('/proc/self/fd').exists() else None
    import ctypes
    from ctypes import wintypes
    count = wintypes.DWORD()
    call = ctypes.windll.kernel32.GetProcessHandleCount
    call.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    if not call(wintypes.HANDLE(-1), ctypes.byref(count)):
        raise ctypes.WinError()
    return count.value


def system_times():
    if os.name != 'nt':
        return None
    import ctypes
    from ctypes import wintypes
    values = [wintypes.FILETIME() for _ in range(3)]
    if not ctypes.windll.kernel32.GetSystemTimes(*(ctypes.byref(v) for v in values)):
        raise ctypes.WinError()
    return [(v.dwHighDateTime << 32) + v.dwLowDateTime for v in values]


def child(mode, root):
    root.mkdir()
    rows = []

    def measure(name, action, count=1):
        before_handles, before_system = handles(), system_times()
        wall, cpu = time.perf_counter(), time.process_time()
        result = action()
        cpu, wall = time.process_time() - cpu, time.perf_counter() - wall
        after_system = system_times()
        busy = None
        if before_system is not None:
            delta = [b-a for a, b in zip(before_system, after_system)]
            total = delta[1] + delta[2]
            busy = 100 * (total-delta[0]) / total if total else None
        rows.append({'name': name, 'count': count, 'wall_s': wall, 'cpu_s': cpu,
                     'system_busy_percent': busy, 'handles_before': before_handles,
                     'handles_after': handles()})
        return result

    if mode == 'python':
        measure('python_noop', lambda: None)
        return rows

    def load_cli():
        from rpipe.flow.cli import main
        return main
    main = measure('import_cli', load_cli)
    from rpipe.structure.artifact import load_result, write_config
    from rpipe.structure.artifact._atomic import atomic_write_text
    from rpipe.structure.artifact.shared import cache_lock, prepared_resource

    def invoke(argv):
        with contextlib.redirect_stdout(io.StringIO()):
            assert main(argv) == 0

    for index in range(3):
        def help_call():
            try:
                invoke(['--help'])
            except SystemExit as exc:
                assert exc.code == 0
        measure('same_process_help', help_call)
    for index in range(2):
        study = root / f'study-{index}'
        study.mkdir()
        shutil.copyfile(Path(__file__).with_name('launch_retry_recipe.py'), study / 'recipe.py')
        write_config(study / 'study.yaml', {'study': 'startup', 'seeds': [0], 'recipe': 'recipe.py',
                     'freeze': True, 'flow': {'prepare_shared': False}})
        write_config(study / 'experiment_config.yaml', {
            'version': f'startup-{root.name}-{index}', 'data': {'name': 'Toy', 'source': 'stub'},
            'model': {'name': 'none', 'source': 'retry_fixture'},
            'algorithm': {'mode': 'inference', 'source': 'retry_fixture', 'config': {'fail': False}},
            'system': {'device': 'cpu', 'deterministic': True, 'cudnn_benchmark': False,
                       'cudnn_allow_tf32': False, 'matmul_allow_tf32': False}})
        measure(f'make_{index}', lambda: invoke(['make', str(study), '--num-gpus', '0']))
        job = json.loads((study / 'scripts/jobs.json').read_text())['jobs'][0]
        measure(f'run_one_{index}', lambda: invoke(['run-one', str(study), job['run_id']]))
        assert load_result(study / 'runs' / job['run_id'] / 'result.json')['status'] == 'succeeded'

    target = root / 'atomic.json'
    text = json.dumps({'payload': 'x' * 2048})
    measure('atomic_first', lambda: atomic_write_text(target, text))
    def writes():
        for _ in range(50):
            atomic_write_text(target, text)
    measure('atomic_50', writes, 50)
    lock_root = root / 'lock'
    def locks():
        for _ in range(50):
            with cache_lock(lock_root, timeout=0):
                pass
    measure('lock_50', locks, 50)
    def build(staging):
        (staging / 'input').write_bytes(b'x' * 65536)
    def resource():
        return prepared_resource(root / 'cache', {'revision': 'fixed'}, ['input'], build)
    first = measure('cache_initial', resource)
    for _ in range(3):
        assert measure('cache_reuse', resource) == first
    assert target.read_text() == text and not list(root.glob('*.tmp'))
    return rows


def main():
    if sys.argv[1] == '--child':
        print(json.dumps(child(sys.argv[2], Path(sys.argv[3])), allow_nan=False))
        return
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=False)
    report = {'scope': 'engineering CPU diagnostic, no speed verdict or historical cause claim',
              'os': platform.platform(), 'python': sys.version, 'cpu_count': os.cpu_count(),
              'head': os.environ.get('GITHUB_SHA'), 'base': '9324014985680a0d186863cd135faa53a2e7bcbf',
              'dependencies': {n: importlib.metadata.version(n) for n in ['torch', 'numpy', 'PyYAML']},
              'environment': {n: os.environ.get(n) for n in ['CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS',
                              'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']}, 'samples': []}
    deadline = time.monotonic() + 120
    for repeat in range(3):
        for mode in (['python', 'rpipe'] if repeat % 2 == 0 else ['rpipe', 'python']):
            workspace = root / f'{repeat}-{mode}'
            wall = time.perf_counter()
            result = subprocess.run([sys.executable, '-X', 'importtime', __file__, '--child', mode, str(workspace)],
                                    capture_output=True, text=True, encoding='utf-8',
                                    timeout=min(45, max(.1, deadline-time.monotonic())))
            elapsed = time.perf_counter() - wall
            (root / f'{workspace.name}.stdout').write_text(result.stdout, encoding='utf-8')
            (root / f'{workspace.name}.importtime').write_text(result.stderr, encoding='utf-8')
            selected = [line for line in result.stderr.splitlines()
                        if any(line.rstrip().endswith(n) for n in ['numpy', 'torch', 'rpipe', 'rpipe.flow.cli'])]
            sample = {'repeat': repeat, 'mode': mode, 'process_wall_s': elapsed,
                      'exit_code': result.returncode, 'importtime_selected': selected,
                      'rows': json.loads(result.stdout) if result.returncode == 0 else [],
                      'failure': result.stderr[-3000:] if result.returncode else None}
            report['samples'].append(sample)
            (root / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
            print(json.dumps(sample, allow_nan=False), flush=True)
            assert result.returncode == 0, sample['failure']
    print(json.dumps({'completed': True, 'report': str(root / 'result.json')}))


if __name__ == '__main__':
    main()
