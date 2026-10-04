"""Publish full measured curves while preserving the immutable historical PNGs.

Only NumPy/Matplotlib and stored JSON are used; no training, GPU, or inference.
The old reference paths in frozen scientific records remain historical paths.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
DOCS = STUDY / 'docs'
REFERENCE_DIR = DOCS / 'reference'
MANIFEST = REFERENCE_DIR / 'PUBLISHED_FIGURES.json'
DATASETS = ('MNIST', 'CIFAR10')
MODELS = ('linear', 'mlp', 'cnn', 'resnet18')
SEEDS = (0, 1, 2, 3)
STEPS = list(range(200, 80001, 200))
ARCHIVE_NAMES = {data: f'{data}_Accuracy_mean_4ccb28d.png' for data in DATASETS}
COLORS = {'linear': '#d62728', 'mlp': '#ed9700', 'cnn': '#3454d1', 'resnet18': '#009eaf'}
LABELS = {'linear': 'Linear', 'mlp': 'MLP', 'cnn': 'CNN', 'resnet18': 'ResNet18'}
LINESTYLES = {'linear': '-', 'mlp': '--', 'cnn': ':', 'resnet18': '-.'}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def png_size(payload: bytes) -> list[int]:
    require(payload[:8] == b'\x89PNG\r\n\x1a\n', 'not a PNG')
    return list(struct.unpack('>II', payload[16:24]))


def verify_original(payload: bytes, row: dict) -> None:
    require(hashlib.sha256(payload).hexdigest() == row['sha256'], 'original PNG SHA differs')
    blob = hashlib.sha1(b'blob ' + str(len(payload)).encode('ascii') + b'\0' + payload).hexdigest()
    require(blob == row['git_blob'], 'original PNG Git blob differs')
    require(png_size(payload) == row['image_size_px'], 'original PNG dimensions differ')


def resolve_historical_reference(data: str, reference: dict | None = None) -> Path:
    """Resolve a frozen original-image record after root assets are published."""
    reference = reference or read(DOCS / 'REFERENCE_CURVES.json')
    require(data in DATASETS, 'unknown dataset')
    archive = REFERENCE_DIR / ARCHIVE_NAMES[data]
    require(archive.is_file(), f'missing archived reference: {archive}')
    verify_original(archive.read_bytes(), reference['datasets'][data])
    return archive


def preserve_references(reference: dict) -> dict:
    """Copy original bytes before replacement; never overwrite a differing archive."""
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    archived = {}
    for data in DATASETS:
        row = reference['datasets'][data]
        destination = REFERENCE_DIR / ARCHIVE_NAMES[data]
        if destination.exists():
            payload = destination.read_bytes()
        else:
            current = ROOT / row['source_path']
            payload = current.read_bytes() if current.is_file() else b''
            if hashlib.sha256(payload).hexdigest() != row['sha256']:
                # This also supports regeneration in a clone whose assets are already new.
                payload = subprocess.check_output(
                    ['git', 'cat-file', 'blob', row['git_blob']], cwd=ROOT)
            verify_original(payload, row)
            with destination.open('xb') as handle:
                handle.write(payload)
        verify_original(payload, row)
        archived[data] = {
            'historical_source_path': row['source_path'],
            'archived_path': relative(destination),
            'sha256': row['sha256'], 'git_blob': row['git_blob'],
            'image_size_px': row['image_size_px'], 'bytes': len(payload),
        }
    return archived


def validate_measured_snapshot() -> tuple[dict, dict, list[dict]]:
    comparison = read(DOCS / 'COMPARISON.json')
    reference = read(DOCS / 'REFERENCE_CURVES.json')
    final = read(DOCS / 'FINAL_RESULT.json')
    require(comparison['complete'] is True and comparison['final_gates_applied'] is True
            and comparison['passed'] is True and comparison['errors'] == []
            and comparison['partial_mode'] is False, 'comparison is not a passing complete matrix')
    require(comparison['run_counts'] == {'train': {'succeeded': 32}, 'eval': {'succeeded': 32}},
            'formal train/eval counts differ')
    require(comparison['source_sha'] == reference['source_sha'] == final['historical_commit'],
            'historical source binding differs')
    require(comparison['reference_file_sha256'] == sha(DOCS / 'REFERENCE_CURVES.json'),
            'comparison reference binding differs')
    require(comparison['comparison_script_sha256'] == sha(STUDY / 'compare.py'),
            'comparison program binding differs')
    for name in ['COMPARISON.json', 'SOURCE_MANIFEST.json']:
        require(final['evidence_sha256'][name] == sha(DOCS / name),
                f'final result binding differs: {name}')
    require(final['complete'] is True and final['passed'] is True and final['errors'] == [],
            'final record is not complete and passing')
    runs = comparison['runs']
    require(len(runs) == 64 and len({row['id'] for row in runs}) == 64,
            'expected 64 unique formal runs')
    table = {(row['data'], row['model'], row['mode'], row['seed']): row for row in runs}
    expected = {(data, model, mode, seed) for data in DATASETS for model in MODELS
                for mode in ('train', 'eval') for seed in SEEDS}
    require(set(table) == expected and all(row['status'] == 'succeeded' and not row['errors']
                                         for row in runs), 'formal matrix is incomplete')
    experiments = {(row['data'], row['model']): row for row in comparison['experiments']}
    require(len(experiments) == 8 and set(experiments) ==
            {(data, model) for data in DATASETS for model in MODELS}, 'expected eight curves')
    records = []
    for data in DATASETS:
        for model in MODELS:
            row = experiments[data, model]
            require(row['optimizer_steps'] == STEPS and row['seed_count_by_step'] == [4] * 400
                    and row['seeds_with_observations'] == list(SEEDS)
                    and row['completed_train_seeds'] == list(SEEDS),
                    f'{data}/{model}: coordinates/seeds differ')
            train = [table[data, model, 'train', seed] for seed in SEEDS]
            require(all(item['optimizer_steps'] == STEPS and item['points'] == 400
                        and item['last_optimizer_step'] == 80000 for item in train),
                    f'{data}/{model}: incomplete seed curves')
            matrix = np.asarray([item['accuracy_pct'] for item in train], dtype=np.float64)
            mean = np.asarray(row['mean_accuracy_pct'], dtype=np.float64)
            std = np.asarray(row['population_std_accuracy_pp'], dtype=np.float64)
            require(matrix.shape == (4, 400) and mean.shape == std.shape == (400,)
                    and np.all(np.isfinite(matrix)) and np.all(np.isfinite(mean))
                    and np.all(np.isfinite(std)) and np.all(std >= 0),
                    f'{data}/{model}: invalid point array')
            require(np.all((matrix >= 0) & (matrix <= 100.00001)), 'invalid percent Accuracy')
            mean_error = float(np.max(np.abs(mean - np.mean(matrix, axis=0))))
            std_error = float(np.max(np.abs(std - np.std(matrix, axis=0, ddof=0))))
            require(mean_error <= 1e-12 and std_error <= 1e-12,
                    f'{data}/{model}: aggregate differs from four stored seed vectors')
            records.append({
                'data': data, 'model': model, 'train_run_ids': [item['id'] for item in train],
                'seeds': list(SEEDS), 'optimizer_steps': STEPS,
                'seed_count_by_step': [4] * 400,
                'mean_accuracy_pct': row['mean_accuracy_pct'],
                'population_std_accuracy_pp': row['population_std_accuracy_pp'],
                'recomputed_mean_maximum_abs_difference_pp': mean_error,
                'recomputed_std_maximum_abs_difference_pp': std_error,
                'mean_plus_std_max_pct': float(np.max(mean + std)),
                'mean_minus_std_min_pct': float(np.min(mean - std)),
            })
    return comparison, reference, records


def render(data: str, records: list[dict], destination: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'path.simplify': False, 'axes.spines.top': False,
                         'axes.spines.right': False}):
        figure, axis = plt.subplots(figsize=(11, 6.5), dpi=180)
        figure.subplots_adjust(left=0.09, right=0.94, bottom=0.17, top=0.82)
        low, high = [], []
        for row in records:
            if row['data'] != data:
                continue
            model = row['model']
            x = np.asarray(row['optimizer_steps'])
            mean = np.asarray(row['mean_accuracy_pct'])
            std = np.asarray(row['population_std_accuracy_pp'])
            axis.fill_between(x, mean - std, mean + std,
                              color=COLORS[model], alpha=0.13, linewidth=0)
            axis.plot(x, mean, color=COLORS[model], linestyle=LINESTYLES[model],
                      linewidth=2.2, label=LABELS[model])
            low.append(float(np.min(mean - std)))
            high.append(float(np.max(mean + std)))
        padding = (max(high) - min(low)) * 0.055
        axis.set_ylim(min(low) - padding, max(high) + padding)
        axis.set_xlim(0, 80000)
        axis.set_xticks([0, 20000, 40000, 60000, 80000])
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
        axis.set_xlabel('Optimizer step', labelpad=10)
        axis.set_ylabel('Test accuracy (%)', labelpad=10)
        axis.grid(color='#d8dee7', linewidth=0.8, alpha=0.85)
        axis.set_axisbelow(True)
        axis.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.015),
                    frameon=False, columnspacing=2.3)
        figure.suptitle(f'{data} | Full test accuracy', fontsize=19, fontweight='bold', y=0.97)
        figure.text(0.5, 0.89, 'Four seeds (0-3) | 400 evaluations | Measured 2026-10-04',
                    ha='center', fontsize=12, color='#475569')
        figure.text(0.09, 0.066,
                    'Line: mean. Shading: population std (ddof=0). All recorded points; no smoothing.',
                    fontsize=10, color='#475569')
        if max(high) > 100:
            figure.text(0.09, 0.037,
                        'Mean +/- std bands may exceed 100%; the band is not an accuracy observation.',
                        fontsize=9, color='#475569')
        temporary = ROOT / '.tmp/publish-figures' / destination.name
        temporary.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(temporary, metadata={
            'Software': 'RPipe publish_figures.py',
            'Description': 'Complete measured four-seed mean and population std, optimizer steps 200..80000',
        })
        plt.close(figure)
    require(destination.resolve().parent == (ROOT / 'asset').resolve(), 'unexpected output directory')
    for attempt in range(5):
        try:
            os.replace(temporary, destination)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.2 * (attempt + 1))


def publish() -> dict:
    comparison, reference, records = validate_measured_snapshot()
    archived = preserve_references(reference)
    output = {}
    for data in DATASETS:
        path = ROOT / 'asset' / f'{data}_Accuracy_mean.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        render(data, records, path)
        output[data] = {'path': relative(path), 'sha256': sha(path),
                        'image_size_px': png_size(path.read_bytes()),
                        'points_per_model': 400, 'models': list(MODELS)}
    import matplotlib
    report = {
        'schema': 'rpipe.main_historical.published_figures.v1',
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'passed': True,
        'plotting_script': relative(Path(__file__)), 'plotting_script_sha256': sha(Path(__file__)),
        'versions': {'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
        'inputs': {relative(DOCS / name): sha(DOCS / name) for name in
                   ['COMPARISON.json', 'REFERENCE_CURVES.json', 'SOURCE_MANIFEST.json', 'FINAL_RESULT.json']},
        'historical_source_commit': comparison['source_sha'],
        'frozen_scientific_records_modified': False,
        'historical_reference_mapping': archived,
        'published_images': output,
        'metric': 'Training-run full-test Accuracy; independent own-best eval is not the curve endpoint',
        'axis': {'name': 'optimizer_step', 'first': 200, 'last': 80000, 'spacing': 200, 'points': 400},
        'aggregation': {'seeds': list(SEEDS), 'count_every_point': 4,
                        'mean': 'arithmetic mean', 'std': 'population', 'ddof': 0,
                        'smoothing': False, 'interpolation': False, 'point_deletion': False,
                        'path_simplification': False, 'band_clipping': False,
                        'historical_png_std_used': False},
        'curves': records,
        'compatibility': {
            'old_source_paths': 'Frozen source_path and original_images.path identify pre-publication root assets',
            'resolver': 'publish_figures.resolve_historical_reference validates the delivered archived bytes',
            'training_and_comparison': 'Existing numerical helpers read pinned Git blobs and frozen JSON; root asset images are not part of the 99-file source gate',
        },
        'gpu_operations': 0, 'training_or_inference_executed': False,
    }
    MANIFEST.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return report


def verify() -> dict:
    _, reference, records = validate_measured_snapshot()
    manifest = read(MANIFEST)
    errors = []
    for name, expected in manifest['inputs'].items():
        if sha(ROOT / name) != expected:
            errors.append(f'published input changed: {name}')
    if sha(Path(__file__)) != manifest['plotting_script_sha256']:
        errors.append('plotting script changed after publication')
    if records != manifest['curves']:
        errors.append('published curves differ from complete stored four-seed vectors')
    for data in DATASETS:
        original = resolve_historical_reference(data, reference)
        if sha(original) != manifest['historical_reference_mapping'][data]['sha256']:
            errors.append(f'archived original changed: {data}')
        output = manifest['published_images'][data]
        path = ROOT / output['path']
        if sha(path) != output['sha256'] or png_size(path.read_bytes()) != output['image_size_px']:
            errors.append(f'published image changed: {data}')
    return {'schema': 'rpipe.main_historical.published_figures_verification.v1',
            'verified_at_utc': datetime.now(timezone.utc).isoformat(),
            'passed': not errors, 'errors': errors,
            'published_manifest_sha256': sha(MANIFEST),
            'comparison_sha256': sha(DOCS / 'COMPARISON.json'),
            'curves': len(records), 'seeds_per_curve': 4, 'points_per_curve': 400,
            'reference_images_verified': 2, 'published_images_verified': 2,
            'gpu_operations': 0, 'training_or_inference_executed': False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='Read-only check of published/archived images and fixed snapshot bindings')
    parser.add_argument('--verify-output', type=Path,
                        default=ROOT / '.tmp/publish-figures/VERIFICATION.json')
    args = parser.parse_args()
    if args.verify:
        report = verify()
        args.verify_output.parent.mkdir(parents=True, exist_ok=True)
        args.verify_output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report['passed'] else 1
    report = publish()
    print(json.dumps({'passed': report['passed'], 'published_images': report['published_images'],
                      'historical_reference_mapping': report['historical_reference_mapping']},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
