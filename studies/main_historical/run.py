"""Run the archived README recipe through the current RPipe Flow.

Study-specific registries must be installed in every child process. The public
RPipe CLI stays unchanged. Interrupted training is never silently resumed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parent.parent
sys.path[:0] = [str(ROOT / '.tmp' / 'runtime'), str(ROOT / 'src'), str(STUDY)]
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONPATH'] = os.pathsep.join(sys.path[:3])
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('PYTHONUTF8', '1')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    from rpipe.structure.artifact._atomic import atomic_write_text
    atomic_write_text(Path(path), json.dumps(value, indent=2, ensure_ascii=False))


def source_hashes():
    files = sorted((ROOT / 'src').rglob('*.py'))
    # Post-run plot/report utilities do not participate in the numerical chain;
    # their own outputs record provenance separately from this launch gate.
    files += [STUDY / name for name in ('recipe.py', 'run.py', 'prepare_data.py')]
    files += sorted(STUDY.glob('*.yaml')) + sorted(STUDY.glob('*.txt'))
    files += [STUDY / 'docs' / name for name in ('TARGET.md', 'REFERENCE_CURVES.json') if (STUDY / 'docs' / name).exists()]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def plan_hashes():
    files = [STUDY / 'index.json'] + sorted((STUDY / 'runs').glob('*/config.yaml'))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def check_unchanged():
    saved = json.loads((STUDY / 'docs' / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    if saved['files'] != source_hashes() or saved['plan_files'] != plan_hashes():
        raise RuntimeError('source or expanded plan changed after make; retained Runs require their original manifest')


def environment():
    import numpy
    import torch
    import torchvision
    return {
        'recorded_at_utc': utc_now(), 'python': sys.version,
        'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'numpy': numpy.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG'],
        'cpu_threads': 2, 'historical_ref': '4ccb28d0496110253e9f8e3f3df658853f07996b',
        'dev_ref': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    }


def configs():
    from rpipe.structure.artifact import load_config
    index = json.loads((STUDY / 'index.json').read_text(encoding='utf-8'))
    rows = []
    for exp in index['experiments']:
        for run in exp['runs']:
            rid = run['id']
            rows.append((rid, load_config(STUDY / 'runs' / rid / 'config.yaml')))
    return rows


def state(rid):
    result = STUDY / 'runs' / rid / 'result.json'
    if result.is_file():
        return json.loads(result.read_text(encoding='utf-8')).get('status', 'unknown')
    log = STUDY / 'runs' / rid / 'assets' / 'logs' / 'run.log'
    return 'started' if log.exists() else 'pending'


def make():
    from recipe import export_source
    from rpipe.structure.make import expand_study
    export_source()
    out = expand_study(STUDY)
    write_json(STUDY / 'docs' / 'ENVIRONMENT.json', environment())
    write_json(STUDY / 'docs' / 'SOURCE_MANIFEST.json', {'created_at_utc': utc_now(), 'files': source_hashes(), 'plan_files': plan_hashes()})
    print(f"Expanded {len(out['configs'])} Runs; no training started.", flush=True)


def one(rid):
    import numpy
    import torch
    from recipe import register
    from rpipe.structure.artifact import load_config
    from rpipe.flow.cli import launch_one
    check_unchanged()
    cfg = load_config(STUDY / 'runs' / rid / 'config.yaml')
    if state(rid) == 'succeeded':
        print(f'skip succeeded {rid}', flush=True)
        return
    if cfg['algorithm']['mode'] == 'train':
        ckpt = STUDY / 'runs' / rid / 'assets' / 'checkpoints'
        if ckpt.exists() and any(ckpt.iterdir()):
            raise RuntimeError(f'{rid}: interrupted training has checkpoints; preserve it and use a new version for a fresh continuous run')
        # The base default loads latest only if present; the guard above proves
        # there is none. Do not mutate the immutable on-disk configuration.
    torch.set_num_threads(2)
    register(data_root=STUDY / 'shared' / 'data', seed=int(cfg['seed']), sampler_steps=80000)
    launch_one(STUDY, rid)


def launch(seeds):
    preflight = json.loads((STUDY / 'docs' / 'PREFLIGHT.json').read_text(encoding='utf-8'))
    expected = {(d, m) for d in ('MNIST', 'CIFAR10') for m in ('linear', 'mlp', 'cnn', 'resnet18')}
    if preflight.get('passed') is not True or {(r['data'], r['model']) for r in preflight.get('runs', [])} != expected or len(preflight.get('runs', [])) != 8:
        raise RuntimeError('the same-device original/native preflight must pass before launch')
    if preflight.get('source_hashes') != source_hashes():
        raise RuntimeError('source/configuration changed after preflight; rerun preflight before launch')
    if preflight.get('torch') != environment()['torch']:
        raise RuntimeError('Torch environment changed after preflight')
    check_unchanged()
    rows = [(rid, cfg) for rid, cfg in configs() if int(cfg['seed']) in seeds]
    execution = {'started_at_utc': utc_now(), 'seeds': seeds, 'events': [], 'groups': [], 'status': 'running'}
    dest = STUDY / 'docs' / 'EXECUTION.json'
    write_json(dest, execution)
    for mode in ('train', 'eval'):
        for model in ('linear', 'mlp', 'cnn', 'resnet18'):
            limit = 2 if model == 'resnet18' else 4
            for data_name in ('MNIST', 'CIFAR10'):
                candidates = [(rid, cfg) for rid, cfg in rows if cfg['algorithm']['mode'] == mode and cfg['model']['name'] == model and cfg['data']['name'] == data_name and state(rid) != 'succeeded']
                if mode == 'eval':
                    from rpipe.structure.algorithm.resume import sibling_train_ids
                    ready = []
                    for rid, cfg in candidates:
                        parents = sibling_train_ids(STUDY, rid)
                        if not parents or any(state(parent) != 'succeeded' for parent in parents):
                            execution['events'].append({'run': rid, 'event': 'skipped_dependency', 'parents': parents, 'at_utc': utc_now()})
                        else:
                            ready.append((rid, cfg))
                    candidates = ready
                for offset in range(0, len(candidates), limit):
                    check_unchanged()
                    batch = candidates[offset:offset + limit]
                    group = {'mode': mode, 'model': model, 'data': data_name, 'runs': [rid for rid, _ in batch], 'start_utc': utc_now()}
                    execution['groups'].append(group)
                    write_json(dest, execution)
                    print(f"start {mode} {data_name}/{model}: {group['runs']}", flush=True)
                    children = []
                    for rid, cfg in batch:
                        assets = STUDY / 'runs' / rid / 'assets' / 'logs'
                        assets.mkdir(parents=True, exist_ok=True)
                        stream = (assets / 'launcher.log').open('a', encoding='utf-8')
                        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                        child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), 'one', rid], cwd=ROOT, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT, creationflags=flags)
                        children.append((rid, child, stream))
                        execution['events'].append({'run': rid, 'event': 'start', 'pid': child.pid, 'at_utc': utc_now()})
                    write_json(dest, execution)
                    while any(child.poll() is None for _, child, _ in children):
                        time.sleep(2)
                    for rid, child, stream in children:
                        stream.close()
                        execution['events'].append({'run': rid, 'event': 'exit', 'exit_code': child.returncode, 'status': state(rid), 'at_utc': utc_now()})
                        print(f'exit {rid}: {child.returncode}, {state(rid)}', flush=True)
                    group['end_utc'] = utc_now()
                    write_json(dest, execution)
    execution['status'] = 'succeeded' if all(state(rid) == 'succeeded' for rid, _ in rows) else 'incomplete'
    execution['finished_at_utc'] = utc_now()
    write_json(dest, execution)
    process()
    if execution['status'] != 'succeeded':
        raise RuntimeError('matrix incomplete; inspect retained per-Run logs')


def process():
    from rpipe.flow.process import run_study
    from rpipe.structure.artifact.readout import write_numbers
    run_study(STUDY)
    write_numbers(STUDY)


def status():
    counts = {}
    for rid, cfg in configs():
        value = state(rid)
        counts[value] = counts.get(value, 0) + 1
        if value == 'started':
            log = STUDY / 'runs' / rid / 'assets' / 'logs' / 'run.log'
            lines = log.read_text(encoding='utf-8', errors='replace').splitlines()
            print(f"{rid} {cfg['data']['name']}/{cfg['model']['name']} seed={cfg['seed']} {lines[-1] if lines else ''}")
    print(json.dumps(counts, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('make', 'status', 'process', 'preflight'):
        commands.add_parser(name)
    item = commands.add_parser('one')
    item.add_argument('run_id')
    item = commands.add_parser('launch')
    item.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3])
    args = parser.parse_args()
    if args.command == 'one':
        one(args.run_id)
    elif args.command == 'launch':
        launch(args.seeds)
    elif args.command == 'preflight':
        from recipe import preflight
        before = source_hashes()
        output = ROOT / '.tmp' / ('historical-preflight-20261004-' + uuid.uuid4().hex[:8])
        report = preflight(data_root=STUDY / 'shared' / 'data', output_dir=output, steps=600, device='cuda')
        if before != source_hashes():
            raise RuntimeError('source/configuration changed during preflight; retained output is not a valid launch gate')
        report['source_hashes'] = before
        report['output_dir'] = str(output)
        report['environment'] = environment()
        write_json(STUDY / 'docs' / 'PREFLIGHT.json', report)
        if not report.get('passed'):
            raise RuntimeError('preflight failed')
    else:
        globals()[args.command]()


if __name__ == '__main__':
    main()
