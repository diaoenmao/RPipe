"""Verify an installed distribution through its public, offline CLI.

Invoke with the Python that owns the installed wheel/sdist and a new workspace.
The Toy/Stub route verifies packaging and artifact contracts, not ML accuracy.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if workspace.exists():
        parser.error('workspace must be new; retained smoke artifacts are never overwritten')
    workspace.mkdir(parents=True)
    # Import after cwd changes; never insert the repository src path.
    os.chdir(workspace)
    import rpipe

    repo_src = Path(__file__).resolve().parents[1] / 'src'
    module_file = Path(rpipe.__file__).resolve()
    if module_file.is_relative_to(repo_src.resolve()):
        raise RuntimeError(f'expected an installed distribution, imported source checkout: {module_file}')
    if importlib.metadata.version('rpipe') != rpipe.__version__:
        raise RuntimeError('distribution metadata and imported version differ')
    scripts = Path(sysconfig.get_path('scripts'))
    console = scripts / ('rpipe.exe' if os.name == 'nt' else 'rpipe')
    if not console.is_file():
        raise RuntimeError(f'installed console script missing: {console}')

    evidence = {'installed_module': str(module_file), 'version': rpipe.__version__,
                'python': sys.executable, 'scope': 'offline Toy/Stub packaging and CLI contract',
                'gpu_used': False, 'commands': [], 'passed': False}

    def command(argv):
        launch_errors = []
        for attempt in range(20):
            try:
                completed = subprocess.run(argv, cwd=workspace, env=dict(os.environ),
                                           capture_output=True, text=True,
                                           encoding='utf-8', errors='replace', timeout=120)
                break
            except OSError as exc:
                # A Windows sharing violation means CreateProcess did not start;
                # retain it and retry only this bounded pre-launch failure.
                if getattr(exc, 'winerror', None) not in (32, 33) or attempt == 19:
                    raise
                launch_errors.append({'attempt': attempt + 1, 'error': str(exc), 'winerror': exc.winerror})
                time.sleep(0.25)
        record = {'argv': [str(v) for v in argv], 'exit_code': completed.returncode,
                  'stdout': completed.stdout, 'stderr': completed.stderr, 'prelaunch_retries': launch_errors}
        evidence['commands'].append(record)
        (workspace / 'SMOKE_RESULT.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
        if completed.returncode:
            raise RuntimeError(f'command failed: {argv}\n{completed.stdout}\n{completed.stderr}')
        return completed.stdout

    command([sys.executable, '-m', 'rpipe', '--help'])
    command([str(console), '--help'])
    study = workspace / 'toy'
    study.mkdir()
    (study / 'experiment_config.yaml').write_text('''version: installed-smoke-v1
data:
  name: Toy
  source: stub
model:
  name: linear
algorithm:
  source: custom_torch
  mode: train
  num_steps: 2
  progress_unit: step
system:
  device: cpu
  deterministic: true
  cudnn_benchmark: false
''', encoding='utf-8')
    (study / 'study.yaml').write_text('''study: installed_toy
origin: foreign
seeds: [0, 1]
axes:
  algorithm.mode: [train, eval]
''', encoding='utf-8')
    command([sys.executable, '-m', 'rpipe', 'make', str(study), '--num-gpus', '0'])
    index = json.loads((study / 'index.json').read_text(encoding='utf-8'))
    runs = [r for experiment in index['experiments'] for r in experiment['runs']]
    if len(runs) != 4 or len({r['id'] for r in runs}) != 4:
        raise RuntimeError('make did not produce four unique train/eval × seed Runs')
    config_before = {r['id']: (study / 'runs' / r['id'] / 'config.yaml').read_bytes() for r in runs}
    command([str(console), 'launch', str(study), '--num-gpus', '0'])
    result_before = {}
    for row in runs:
        run_dir = study / 'runs' / row['id']
        body = json.loads((run_dir / 'result.json').read_text(encoding='utf-8'))
        if body['status'] != 'succeeded' or body['structure']['data']['source'] != 'stub':
            raise RuntimeError(f'invalid Run status/source: {row["id"]}')
        if (run_dir / 'config.yaml').read_bytes() != config_before[row['id']]:
            raise RuntimeError('launch mutated a generated config')
        if not (run_dir / 'assets/logs/run.log').is_file() or not (run_dir / 'assets/tracker/scalars.jsonl').is_file():
            raise RuntimeError('Run log/tracker missing')
        result_before[row['id']] = (run_dir / 'result.json').read_bytes()
    command([sys.executable, '-m', 'rpipe', 'process', str(study)])
    aggregate = json.loads((study / 'process.json').read_text(encoding='utf-8'))
    if not aggregate['complete'] or len(aggregate['experiments']) != 2:
        raise RuntimeError('Study process not complete')
    if any(e['n'] != 2 or e['n_planned'] != 2 for e in aggregate['experiments']):
        raise RuntimeError('Study process has wrong seed counts')
    command([str(console), 'status', str(study)])
    command([sys.executable, '-m', 'rpipe', 'logs', str(study)])
    command([str(console), 'report', str(study)])
    if not (study / 'docs/NUMBERS.md').is_file():
        raise RuntimeError('public report did not produce NUMBERS.md')
    # A repeated launch must skip completed Runs instead of duplicating Flow logs.
    command([sys.executable, '-m', 'rpipe', 'launch', str(study), '--num-gpus', '0'])
    for row in runs:
        run_dir = study / 'runs' / row['id']
        if (run_dir / 'result.json').read_bytes() != result_before[row['id']]:
            raise RuntimeError('second launch rewrote a successful Run')
        if (run_dir / 'assets/logs/run.log').read_text(encoding='utf-8').count('[flow] start phases=') != 1:
            raise RuntimeError('second launch duplicated a Flow')
    evidence.update(passed=True, runs_succeeded=4, experiments_complete=2,
                    generated_configs_unchanged=True, completed_runs_skipped_on_second_launch=True)
    (workspace / 'SMOKE_RESULT.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'installed_module': str(module_file), 'workspace': str(workspace),
                      'commands': len(evidence['commands']), 'runs_succeeded': 4}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
