"""Independently verify the saved 600-step probe, without importing its recipe.

This reads CPU copies of checkpoints and raw evidence; it never trains a model.
Missing or incomplete evidence exits nonzero and cannot produce a passing gate.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any


STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parent.parent
COMMIT = '4ccb28d0496110253e9f8e3f3df658853f07996b'
STEPS = [200, 400, 600]
PAIRS = {(data, model) for data in ('MNIST', 'CIFAR10')
         for model in ('linear', 'mlp', 'cnn', 'resnet18')}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def equal_tree(left: Any, right: Any, path: str = '', *, stats: dict[str, int] | None = None) -> None:
    """Compare all tensor values, including optimizer momentum, exactly."""
    import torch

    stats = stats if stats is not None else {'tensors': 0, 'elements': 0}
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        require(isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor), f'{path}: tensor type differs')
        require(left.shape == right.shape and left.dtype == right.dtype, f'{path}: tensor shape/dtype differs')
        require(torch.equal(left, right), f'{path}: tensor values differ')
        stats['tensors'] += 1
        stats['elements'] += left.numel()
    elif isinstance(left, dict) or isinstance(right, dict):
        require(isinstance(left, dict) and isinstance(right, dict), f'{path}: mapping type differs')
        require(set(left) == set(right), f'{path}: mapping keys differ')
        for key in left:
            equal_tree(left[key], right[key], f'{path}.{key}', stats=stats)
    elif isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        require(type(left) is type(right) and len(left) == len(right), f'{path}: sequence type/length differs')
        for index, (a, b) in enumerate(zip(left, right)):
            equal_tree(a, b, f'{path}[{index}]', stats=stats)
    else:
        require(left == right, f'{path}: scalar values differ')


def source_hashes() -> dict[str, str]:
    files = sorted((ROOT / 'src').rglob('*.py'))
    files += [STUDY / name for name in ('recipe.py', 'run.py', 'prepare_data.py')]
    files += sorted(STUDY.glob('*.yaml')) + sorted(STUDY.glob('*.txt'))
    files += [STUDY / 'docs' / name for name in ('TARGET.md', 'REFERENCE_CURVES.json')
              if (STUDY / 'docs' / name).is_file()]
    return {path.relative_to(ROOT).as_posix(): digest(path) for path in files}


def verify_sources(preflight: dict[str, Any]) -> dict[str, Any]:
    recorded = preflight.get('source_hashes')
    current = source_hashes()
    require(isinstance(recorded, dict) and recorded == current, 'numerical source/configuration differs from preflight')
    require(preflight.get('recipe_sha256') == digest(STUDY / 'recipe.py'), 'recipe hash differs')
    archive = ROOT / '.tmp' / 'historical-source-4ccb28d'
    tree = subprocess.check_output(['git', 'ls-tree', '-r', '-z', COMMIT], cwd=ROOT)
    rows = []
    for entry in tree.split(b'\0'):
        if not entry:
            continue
        attributes, raw_name = entry.split(b'\t', 1)
        _, kind, oid = attributes.decode('ascii').split()
        if kind != 'blob':
            continue
        name = raw_name.decode('utf-8')
        original = subprocess.check_output(['git', 'cat-file', 'blob', oid], cwd=ROOT)
        path = archive / name
        require(path.is_file() and path.read_bytes() == original, f'archived Git blob differs: {name}')
        rows.append({'path': name, 'git_blob': oid, 'sha256': hashlib.sha256(original).hexdigest()})
    require(len(rows) == 36, f'expected 36 immutable historical files, found {len(rows)}')
    return {'passed': True, 'current_files': len(current), 'historical_files': len(rows),
            'historical_ref': COMMIT, 'archive_files': rows}


def verify_data() -> dict[str, Any]:
    path = STUDY / 'docs' / 'DATA_MANIFEST.json'
    manifest = read_json(path)
    expected_path = ROOT / manifest['expected_manifest']
    require(digest(expected_path) == manifest['expected_manifest_sha256'], 'historical data manifest changed')
    original = read_json(expected_path)
    expected = {name.replace('\\', '/').removeprefix('data/'): value for name, value in original.items()}
    rows = manifest['files']
    actual = {row['path']: row for row in rows}
    require(len(rows) == len(actual) == len(expected) == 16 and set(actual) == set(expected),
            'expected precisely 16 historical raw dataset files')
    for name, expected_hash in expected.items():
        raw = STUDY / 'shared' / 'data' / name
        row = actual[name]
        require(raw.is_file() and raw.stat().st_size == row['bytes'], f'raw data length differs: {name}')
        require(digest(raw) == row['sha256'] == expected_hash, f'raw data SHA256 differs: {name}')
    require(manifest['split_counts'] == {'MNIST': {'train': 60000, 'test': 10000, 'classes': 10},
                                       'CIFAR10': {'train': 50000, 'test': 10000, 'classes': 10}},
            'full dataset split sizes differ')
    return {'passed': True, 'files': 16, 'historical_sha256_matches': 16,
            'manifest_sha256': digest(path), 'split_counts': manifest['split_counts']}


def verify_case(row: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    import torch

    cell = output_dir / f'{row["data"]}_{row["model"]}'
    original_path = cell / 'original.pt'
    current_path = cell / 'current' / 'assets' / 'checkpoints' / 'latest.pt'
    original = torch.load(original_path, map_location='cpu', weights_only=False)
    current = torch.load(current_path, map_location='cpu', weights_only=False)
    require(original['step'] == current['step'] == 600, 'checkpoint step must be 600')
    model_stats, optimizer_stats = {'tensors': 0, 'elements': 0}, {'tensors': 0, 'elements': 0}
    equal_tree(original['model_state_dict'], current['model'], 'model', stats=model_stats)
    equal_tree(original['optimizer_state_dict'], current['optimizer'], 'optimizer', stats=optimizer_stats)
    equal_tree(original['scheduler_state_dict'], current['scheduler'], 'scheduler')
    scheduler = current['scheduler']
    require(scheduler['T_max'] == 80000 and scheduler['last_epoch'] == 600, 'scheduler horizon/progress differs')
    require(scheduler['_step_count'] == 601 and scheduler['base_lrs'] == [.01], 'scheduler update count/base LR differs')
    for group in current['optimizer']['param_groups']:
        require(group['initial_lr'] == .01 and group['momentum'] == .9 and group['weight_decay'] == .0005
                and group['nesterov'] is True and group['lr'] == scheduler['_last_lr'][0], 'SGD recipe differs')
    require(all('momentum_buffer' in value for value in current['optimizer']['state'].values()),
            'SGD state must contain momentum buffers')

    segments = row['original_segments']
    new_segments = row['current_segments']
    require([segment['step'] for segment in segments] == STEPS, 'original evaluation segments differ')
    require([segment['step'] for segment in new_segments] == STEPS, 'current evaluation segments differ')
    scalar_path = cell / 'current' / 'assets' / 'tracker' / 'scalars.jsonl'
    records = [json.loads(line) for line in scalar_path.read_text(encoding='utf-8').splitlines() if line]
    starts = [record for record in records if record.get('event') == 'start']
    require(len(starts) == 1 and starts[0]['keep_until'] == 0, 'probe must have exactly one fresh tracker start')
    numeric = [record for record in records if 'split' in record]
    require(len(numeric) == 12 and len(records) == 13, 'expected 12 metric records plus one start marker')
    metrics = []
    for split, segment_samples in [('train', 50000), ('test', 10000)]:
        for metric in ('Loss', 'Accuracy'):
            a = original['logger_state_dict']['history'][f'{split}/{metric}']
            b = current['tracker']['splits'][split][metric]['history']
            require(len(a) == len(b) == 3, f'{split}/{metric}: expected three checkpoint history values')
            points = [record for record in numeric if record['split'] == split and record['name'] == metric]
            require([record['optimizer_step'] for record in points] == STEPS, f'{split}/{metric}: scalar coordinates differ')
            expected_batch_steps = [200, 440, 680] if split == 'train' else [240, 480, 720]
            require([record['step'] for record in points] == expected_batch_steps,
                    f'{split}/{metric}: actual train/test batch ledger differs')
            for index, (old_value, current_value) in enumerate(zip(a, b)):
                require(math.isfinite(old_value) and math.isfinite(current_value), f'{split}/{metric}: nonfinite metric')
                require(old_value == segments[index][split][metric], f'{split}/{metric}: original checkpoint differs from JSON')
                require(current_value == new_segments[index][split][metric], f'{split}/{metric}: current checkpoint differs from JSON')
                require(points[index]['mean'] == current_value, f'{split}/{metric}: scalar differs from checkpoint history')
                delta = abs(old_value - current_value)
                check = {'step': STEPS[index], 'split': split, 'metric': metric,
                         'original': old_value, 'current': current_value, 'absolute_difference': delta}
                if metric == 'Loss':
                    require(delta <= 1e-6, f'{split}/Loss: exceeds 1e-6 at step {STEPS[index]}')
                else:
                    old_count = round(old_value * segment_samples / 100)
                    new_count = round(current_value * segment_samples / 100)
                    require(old_count == new_count, f'{split}/Accuracy: correct count differs at step {STEPS[index]}')
                    check.update(samples=segment_samples, original_correct=old_count, current_correct=new_count)
                metrics.append(check)
    require(current['tracker']['step'] == 720 and current['tracker']['progress']['step'] == 600,
            'final tracker batch/optimizer progress differs')
    input_counts = {}
    for split, samples, batches in [('train', 150000, 600), ('test', 30000, 120)]:
        old = row['original_inputs'][split]
        new = row['current_inputs'][split]
        require(old == new and old['samples'] == samples and old['batches'] == batches,
                f'{split}: measured input hashes/sample/batch counts differ')
        require(all(len(old[f'{name}_sha256']) == 64 for name in ('indices', 'images', 'targets')),
                f'{split}: missing input hashes')
        input_counts[split] = {'samples_each': samples, 'batches_each': batches,
                               'per_segment_samples': 50000 if split == 'train' else 10000,
                               'per_segment_batches': 200 if split == 'train' else 40}
    return {'data': row['data'], 'model': row['model'], 'passed': True,
            'parameters_and_buffers_exact': True, 'model_tensor_count': model_stats['tensors'],
            'model_elements': model_stats['elements'], 'optimizer_exact': True,
            'optimizer_tensor_count': optimizer_stats['tensors'], 'scheduler_exact': True,
            'optimizer_steps': STEPS, 'input_counts': input_counts, 'metrics': metrics,
            'original_checkpoint_sha256': digest(original_path), 'current_checkpoint_sha256': digest(current_path),
            'scalars_sha256': digest(scalar_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', type=Path, default=STUDY / 'docs' / 'PREFLIGHT.json')
    parser.add_argument('--output', type=Path, default=STUDY / 'docs' / 'PREFLIGHT_VERIFICATION.json')
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT / '.tmp' / 'runtime'))
    import numpy  # Load before torch for this Windows host's native runtime.
    import torch
    torch.set_num_threads(2)
    report: dict[str, Any] = {'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'passed': False,
                              'verification_script_sha256': digest(Path(__file__)), 'torch': torch.__version__,
                              'numpy': numpy.__version__, 'device': 'cpu', 'errors': [], 'runs': []}
    try:
        preflight = read_json(args.preflight)
        report['preflight_sha256'] = digest(args.preflight)
        require(preflight.get('passed') is True, 'preflight is missing its completed passing status')
        require(preflight.get('historical_commit') == COMMIT and preflight.get('steps') == 600
                and preflight.get('sampler_steps') == 80000 and preflight.get('seed') == 0,
                'preflight recipe/horizon/seed differs')
        rows = preflight.get('runs', [])
        require(len(rows) == 8 and {(row['data'], row['model']) for row in rows} == PAIRS,
                'preflight must contain all eight distinct completed cells')
        require(all(row.get('passed') is True for row in rows), 'preflight has a failed cell')
        require(preflight.get('torch') == torch.__version__, 'verification Torch differs from probe')
        report['sources'] = verify_sources(preflight)
        report['data'] = verify_data()
        output_dir = Path(preflight['output_dir'])
        report['probe_output_dir'] = str(output_dir)
        for row in rows:
            try:
                result = verify_case(row, output_dir)
            except Exception as error:
                result = {'data': row['data'], 'model': row['model'], 'passed': False,
                          'error': f'{type(error).__name__}: {error}'}
            report['runs'].append(result)
            print(f'verify {row["data"]}/{row["model"]}: passed={result["passed"]}', flush=True)
        report['passed'] = len(report['runs']) == 8 and all(row['passed'] for row in report['runs'])
    except Exception as error:
        report['errors'].append(f'{type(error).__name__}: {error}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'independent verification passed={report["passed"]}; report: {args.output}', flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
