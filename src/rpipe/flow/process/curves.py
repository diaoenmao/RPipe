"""Study-level curves from report observations, falling back to tracker history."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rpipe.flow.process.aggregate import collect_from_index, load_curve_series, summarize_curve_series
from rpipe.structure.artifact.paths import DOCS_DIRNAME
from rpipe.structure.artifact.result import STATUS_SUCCEEDED

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
        parts.append(f'{str(key).removesuffix(".name").split(".")[-1]}={value}')
    return ' '.join(parts)


def _history(study_dir: Path, run_dir: str, split: str, name: str) -> list[float]:
    return load_curve_series(study_dir, run_dir).get(split, {}).get(name, {}).get('values', [])


def collect_curve_groups(study_dir: Path, index: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not index:
        return []
    groups: list[dict[str, Any]] = []
    for exp in collect_from_index(study_dir, index):
        factors = dict(exp.get('factors') or {})
        if str(factors.get('algorithm.mode') or '').lower() == 'eval':
            continue
        run_dirs = [
            str(run.get('run_dir') or run.get('id') or '')
            for run in (exp.get('runs') or []) if run.get('status') == STATUS_SUCCEEDED
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
    """Write ``docs/figures/learning_curves.png``. None if no report observations or history."""
    study_dir = Path(study_dir)
    groups = collect_curve_groups(study_dir, index)
    panels = {}
    for group in groups:
        for split, name, _title in _PANELS:
            series = [
                row for run_dir in group['run_dirs']
                if (row := load_curve_series(study_dir, run_dir).get(split, {}).get(name))
            ]
            summary = summarize_curve_series(series)
            if summary:
                for unit, values in (summary.get('by_unit') or {summary['unit']: summary}).items():
                    panels.setdefault((unit, split, name), []).append((group['label'], values))
    units = [unit for unit in ('step', 'epoch', 'observation') if any(key[0] == unit for key in panels)]
    if not units:
        return None

    import matplotlib.pyplot as plt

    dest = learning_curves_path(study_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axes = plt.subplots(2, 2 * len(units), figsize=(9.2 * len(units), 6.4))
    colors = {group['label']: palette[i % len(palette)] for i, group in enumerate(groups)}
    for column, unit in enumerate(units):
        for panel, (split, name, panel_title) in enumerate(_PANELS):
            ax = axes[panel // 2, column * 2 + panel % 2]
            for label, summary in panels.get((unit, split, name), []):
                means, stds, points = summary['mean'], summary['std'], summary['x']
                counts = summary['n_at_point']
                varying = len(set(counts)) > 1
                color = colors[label]
                label = f'{label} · {unit} · n={min(counts)}' + (f'–{max(counts)}' if varying else '')
                ax.plot(points, means, color=color, label=label, linewidth=1.8,
                        marker='o' if len(means) == 1 or varying else None)
                ax.fill_between(points, [m - s for m, s in zip(means, stds)],
                                [m + s for m, s in zip(means, stds)], color=color, alpha=0.18)
                if varying:
                    for x, y, n in zip(points, means, counts):
                        ax.annotate(f'n={n}', (x, y), xytext=(0, 6), textcoords='offset points',
                                    ha='center', fontsize=7, color=color)
            ax.set_title(panel_title)
            ax.set_xlabel({'step': 'optimizer step', 'epoch': 'epoch', 'observation': 'history point'}[unit])
            ax.grid(True, alpha=0.3)
            if name == 'Accuracy':
                ax.set_ylim(0.0, 102.0)
                ax.set_ylabel('Accuracy (%)')
    handles = {}
    for ax in fig.axes:
        lines, labels = ax.get_legend_handles_labels()
        handles.update(zip(labels, lines))
    fig.legend(handles.values(), handles.keys(), loc='lower center', ncol=2, fontsize=8)
    fig.suptitle(title or f'{study_dir.name} · mean ± std', fontsize=11)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    return dest
