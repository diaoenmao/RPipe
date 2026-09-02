"""Study-level learning curves from tracker_state history (process sidecar)."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from rpipe.structure.artifact.asset import kinds
from rpipe.structure.artifact.paths import DOCS_DIRNAME, RUNS_DIRNAME

FIGURES_DIRNAME = 'figures'
LEARNING_CURVES_NAME = 'learning_curves.png'

_PANELS = (
    ('test', 'Accuracy', 'Test accuracy'),
    ('train', 'Accuracy', 'Train accuracy'),
    ('test', 'Loss', 'Test loss'),
    ('train', 'Loss', 'Train loss'),
)


def figures_dir(study_dir: Path | str) -> Path:
    return Path(study_dir) / DOCS_DIRNAME / FIGURES_DIRNAME


def learning_curves_path(study_dir: Path | str) -> Path:
    return figures_dir(study_dir) / LEARNING_CURVES_NAME


def _label(factors: dict[str, Any]) -> str:
    if not factors:
        return 'experiment'
    parts: list[str] = []
    for key, value in factors.items():
        parts.append(f'{str(key).split(".")[-1]}={value}')
    return ' '.join(parts)


def _history(study_dir: Path, run_dir: str, split: str, name: str) -> list[float]:
    path = study_dir / RUNS_DIRNAME / run_dir / 'assets' / kinds.TRACKER_STATE
    if not path.is_file():
        return []
    try:
        body = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError, TypeError):
        return []
    splits = body.get('splits') if isinstance(body, dict) else None
    if not isinstance(splits, dict):
        return []
    meter = ((splits.get(split) or {}).get(name) or {})
    history = meter.get('history')
    if not isinstance(history, list):
        return []
    out: list[float] = []
    for value in history:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out.append(float(value))
    return out


def _mean_std(series: list[list[float]]) -> tuple[list[float], list[float]]:
    length = min(len(row) for row in series)
    means: list[float] = []
    stds: list[float] = []
    for i in range(length):
        values = [row[i] for row in series]
        means.append(float(statistics.fmean(values)))
        stds.append(float(statistics.stdev(values)) if len(values) >= 2 else 0.0)
    return means, stds


def collect_curve_groups(study_dir: Path, index: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not index:
        return []
    groups: list[dict[str, Any]] = []
    for exp in index.get('experiments') or []:
        factors = dict(exp.get('factors') or {})
        run_dirs = [
            str(run.get('run_dir') or run.get('id') or '')
            for run in (exp.get('runs') or [])
        ]
        run_dirs = [name for name in run_dirs if name]
        if not run_dirs:
            continue
        groups.append({'label': _label(factors), 'run_dirs': run_dirs})
    return groups


def write_learning_curves(
    study_dir: Path | str,
    index: dict[str, Any] | None,
    *,
    title: str | None = None,
) -> Path | None:
    """Write ``docs/figures/learning_curves.png``. None if no epoch history."""
    study_dir = Path(study_dir)
    groups = collect_curve_groups(study_dir, index)
    drawn = False
    for group in groups:
        for split, name, _title in _PANELS:
            if any(_history(study_dir, run_dir, split, name) for run_dir in group['run_dirs']):
                drawn = True
                break
        if drawn:
            break
    if not drawn:
        return None

    import matplotlib.pyplot as plt

    dest = learning_curves_path(study_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    palette = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b')
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), sharex=True)
    panel_axes = (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1])
    for ax, (split, name, panel_title) in zip(panel_axes, _PANELS):
        for i, group in enumerate(groups):
            series = [
                row
                for row in (
                    _history(study_dir, run_dir, split, name) for run_dir in group['run_dirs']
                )
                if row
            ]
            if not series:
                continue
            means, stds = _mean_std(series)
            epochs = list(range(1, len(means) + 1))
            color = palette[i % len(palette)]
            ax.plot(epochs, means, color=color, label=group['label'], linewidth=1.8)
            lo = [m - s for m, s in zip(means, stds)]
            hi = [m + s for m, s in zip(means, stds)]
            ax.fill_between(epochs, lo, hi, color=color, alpha=0.18)
        ax.set_title(panel_title)
        ax.set_xlabel('epoch')
        ax.grid(True, alpha=0.3)
        if name == 'Accuracy':
            ax.set_ylim(0.0, 1.02)
    axes[0, 0].legend(loc='best', fontsize=8)
    fig.suptitle(title or f'{study_dir.name} · mean ± std', fontsize=11)
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    return dest
