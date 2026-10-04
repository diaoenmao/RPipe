"""Dataset profile written next to shared data.

One file per dataset: split sizes, class counts, pixel range, and the train
mean / std that Normalize reads. This is a look at the data, not only a
standardize recipe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from rpipe.structure.data.config import DataConfig
from rpipe.structure.data.factory import DataFactory, vision_root
from rpipe.structure.make.expand import load_study_yaml
from rpipe.structure.origin import normalize_origin

STATS_NAME = 'stats.yaml'


def summarize_batches(batches: list[tuple[Any, Any]]) -> dict[str, Any]:
    """Channel mean / std / min / max and class counts over NCHW batches in 0–1."""
    import torch

    count = 0
    sum_c = None
    sumsq = None
    min_c = None
    max_c = None
    classes: dict[int, int] = {}
    shape: list[int] | None = None
    for images, targets in batches:
        images = images.detach().float()
        if images.ndim != 4:
            raise ValueError(f'expected NCHW, got {tuple(images.shape)}')
        if shape is None:
            shape = [int(x) for x in images.shape[1:]]
        flat = images.transpose(0, 1).reshape(images.size(1), -1)
        batch_sum = flat.sum(dim=1)
        batch_sq = flat.square().sum(dim=1)
        batch_min = flat.min(dim=1).values
        batch_max = flat.max(dim=1).values
        if sum_c is None:
            sum_c = batch_sum
            sumsq = batch_sq
            min_c = batch_min
            max_c = batch_max
        else:
            sum_c = sum_c + batch_sum
            sumsq = sumsq + batch_sq
            min_c = torch.minimum(min_c, batch_min)
            max_c = torch.maximum(max_c, batch_max)
        count += int(flat.size(1))
        for label in targets.detach().reshape(-1).tolist():
            key = int(label)
            classes[key] = classes.get(key, 0) + 1
    if count <= 0 or sum_c is None or sumsq is None or min_c is None or max_c is None:
        raise ValueError('no samples')
    mean = sum_c / count
    var = sumsq / count - mean.square()
    std = var.clamp_min(0).sqrt()
    rows = [{'label': label, 'count': classes[label]} for label in sorted(classes)]
    return {
        'count': int(sum(classes.values())),
        'shape': shape or [],
        'classes': rows,
        'pixel': {
            'min': _round_list(min_c),
            'max': _round_list(max_c),
            'mean': _round_list(mean),
            'std': _round_list(std),
        },
    }


def load_stats(path: Path | str) -> dict[str, Any] | None:
    file = Path(path)
    if not file.is_file():
        return None
    body = yaml.safe_load(file.read_text(encoding='utf-8')) or {}
    if not isinstance(body, dict) or not body.get('mean') or not body.get('std'):
        return None
    return body


def profile_study(study_dir: Path | str) -> list[Path]:
    """Download each vision dataset if needed, then write ``stats.yaml`` beside it."""
    study = Path(study_dir)
    declared = load_study_yaml(study)
    origin = normalize_origin(declared.get('origin'))
    shared = study / 'shared' / 'data'
    written: list[Path] = []
    for name in _data_names(declared):
        root = vision_root(name) or name
        folder = shared / root
        data = DataFactory.build(
            DataConfig.from_mapping(
                {
                    'name': name,
                    'source': 'torch',
                    'config': {'batch_size': 256, 'augment': False, 'num_workers': 0},
                }
            ),
            shared,
            origin=origin,
        )
        splits = {}
        for split in ('train', 'test'):
            batches = [(images, targets) for images, targets in _iter_pairs(data, split)]
            if batches:
                splits[split] = summarize_batches(batches)
        if 'train' not in splits:
            raise RuntimeError(f'{name}: train split is empty')
        train_pixel = splits['train']['pixel']
        body = {
            'name': name,
            'summary': _summary_line(name, splits),
            'mean': list(train_pixel['mean']),
            'std': list(train_pixel['std']),
            'splits': splits,
        }
        path = folder / STATS_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(body, sort_keys=False, allow_unicode=True), encoding='utf-8')
        written.append(path)
    return written


def _iter_pairs(data: Any, split: str):
    loader = getattr(data, '_loaders', {}).get(split)
    if loader is None:
        return
    for batch in loader:
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            yield batch[0], batch[1]


def _data_names(study: dict[str, Any]) -> list[str]:
    names: list[str] = []
    axes = study.get('axes') if isinstance(study.get('axes'), dict) else {}
    axis_names = axes.get('data.name') if isinstance(axes, dict) else None
    if isinstance(axis_names, list):
        names.extend(str(item) for item in axis_names)
    fixed = study.get('fixed') if isinstance(study.get('fixed'), dict) else {}
    data = fixed.get('data') if isinstance(fixed, dict) else None
    if isinstance(data, dict) and data.get('name'):
        names.append(str(data['name']))
    seen: list[str] = []
    for name in names:
        if name not in seen:
            seen.append(name)
    if not seen:
        raise ValueError('study.yaml has no data.name')
    return seen


def _summary_line(name: str, splits: dict[str, Any]) -> str:
    parts = [name]
    for split, body in splits.items():
        shape = 'x'.join(str(x) for x in body.get('shape') or [])
        classes = body.get('classes') or []
        counts = [int(row['count']) for row in classes]
        balance = ''
        if counts and len(set(counts)) == 1:
            balance = f', {len(counts)} classes of {counts[0]}'
        elif counts:
            balance = f', {len(counts)} classes, counts {min(counts)}–{max(counts)}'
        parts.append(f'{split} {body.get("count")} ({shape}{balance})')
    return '; '.join(parts)


def _round_list(value: Any) -> list[float]:
    return [round(float(item), 6) for item in value.tolist()]
