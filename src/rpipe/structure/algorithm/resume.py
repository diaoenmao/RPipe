"""Resume target resolution (algorithm-layer; system only loads files)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rpipe.structure.algorithm.config import AlgorithmConfig


def resume_stem(config: AlgorithmConfig, *, mode: str) -> str | None:
    """Return checkpoint stem/path to load, or None to skip resume."""
    explicit = config.setting('resume_from')
    if explicit:
        return str(explicit)
    raw = config.setting('resume')
    if raw is None:
        return 'latest' if mode == 'train' else 'best'
    if raw is False:
        return None
    text = str(raw).strip().lower()
    if text in ('', 'false', '0', 'none', 'off'):
        return None
    return str(raw)


def study_dir_from_assets(assets_dir: Path | str | None) -> Path | None:
    if assets_dir is None:
        return None
    assets = Path(assets_dir).resolve()
    if assets.name != 'assets':
        return None
    # studies/<name>/runs/<id>/assets
    try:
        return assets.parent.parent.parent
    except Exception:
        return None


def sibling_train_checkpoint(
    system: Any,
    *,
    stem: str,
    current_id: str | None = None,
) -> Path | None:
    """Eval Run: ``best.pt`` lives on the matching train Run, not this assets dir."""
    assets_dir = getattr(system, 'assets_dir', None)
    study_dir = study_dir_from_assets(assets_dir)
    if study_dir is None or not (study_dir / 'index.json').is_file():
        return None
    run_id = current_id or Path(assets_dir).resolve().parent.name
    from rpipe.structure.artifact.index import load_index

    try:
        index = load_index(study_dir)
    except (OSError, TypeError, ValueError):
        return None
    current_factors: dict[str, Any] | None = None
    current_seed: Any = None
    for exp in index.get('experiments') or []:
        for run in exp.get('runs') or []:
            rid = str(run.get('run_dir') or run.get('id') or '')
            if rid == run_id:
                current_factors = dict(exp.get('factors') or {})
                current_seed = run.get('seed')
                break
        if current_factors is not None:
            break
    if current_factors is None:
        return None
    want = dict(current_factors)
    want['algorithm.mode'] = 'train'
    name = stem if stem.endswith('.pt') else f'{stem}.pt'
    folder_name = stem[:-3] if stem.endswith('.pt') else stem
    for exp in index.get('experiments') or []:
        if dict(exp.get('factors') or {}) != want:
            continue
        for run in exp.get('runs') or []:
            if run.get('seed') != current_seed:
                continue
            rid = str(run.get('run_dir') or run.get('id') or '')
            base = study_dir / 'runs' / rid / 'assets' / 'checkpoints'
            path = base / name
            if path.is_file():
                return path
            folder = base / folder_name
            if folder.is_dir() and ((folder / 'model.pt').is_file() or (folder / 'meta.json').is_file()):
                return folder
    return None


def resolve_checkpoint_ref(
    ref: str,
    system: Any,
    *,
    mode: str,
) -> str:
    """Turn stem / relative path / ``sibling`` into a path or stem system can load."""
    text = str(ref).strip()
    if text.lower() in ('sibling', 'train'):
        stem = 'best' if mode == 'eval' else 'latest'
        found = sibling_train_checkpoint(system, stem=stem)
        return str(found) if found is not None else stem
    path = Path(text)
    if path.is_file() or path.is_dir():
        return str(path)
    assets_dir = getattr(system, 'assets_dir', None)
    study_dir = study_dir_from_assets(assets_dir)
    if study_dir is not None:
        relative = study_dir / text
        if relative.is_file():
            return str(relative)
        if not text.endswith('.pt'):
            alt = study_dir / f'{text}.pt'
            if alt.is_file():
                return str(alt)
    return text


def apply_module_state(module: Any, payload: dict[str, Any] | None) -> None:
    if module is None or not payload:
        return
    state = payload.get('model')
    if state is None:
        return
    module.load_state_dict(state)
