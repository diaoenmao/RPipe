"""Compare two Run directories: result metrics, tracker history, checkpoint state."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from rpipe.structure.artifact._atomic import atomic_write_text
from rpipe.structure.artifact.asset import kinds
from rpipe.structure.artifact.paths import ASSETS_DIRNAME, RESULT_NAME

CHECKPOINT_PARTS = ('model', 'optimizer', 'scheduler')


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _close(a: float, b: float, atol: float, rtol: float) -> tuple[bool, float]:
    if math.isnan(a) or math.isnan(b):
        same = math.isnan(a) and math.isnan(b)
        return same, 0.0 if same else math.inf
    diff = abs(a - b)
    return diff <= atol + rtol * abs(b), diff


def _item(passed: bool, max_abs_diff: float, **extra: Any) -> dict[str, Any]:
    return {'passed': passed, 'max_abs_diff': max_abs_diff, **extra}


def _missing(where: str) -> dict[str, Any]:
    return {'passed': False, 'max_abs_diff': None, 'missing': where}


def _tree(a: Any, b: Any, atol: float, rtol: float, path: str, problems: list[str]) -> float:
    """Return the max abs diff; append mismatch paths to ``problems``."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a runtime dependency
        torch = None
    if torch is not None and isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if tuple(a.shape) != tuple(b.shape):
            problems.append(f'{path}: shape {tuple(a.shape)} != {tuple(b.shape)}')
            return math.inf
        if a.numel() == 0:
            return 0.0
        left, right = a.detach().cpu().double(), b.detach().cpu().double()
        both_nan = torch.isnan(left) & torch.isnan(right)
        diff = (left - right).abs().masked_fill(both_nan, 0.0)
        worst = float(diff.max())
        bound = atol + rtol * right.abs().masked_fill(both_nan, 0.0)
        if math.isnan(worst) or bool((diff > bound).any()):
            problems.append(f'{path}: max abs diff {worst:g}')
        return worst
    if isinstance(a, dict) and isinstance(b, dict):
        worst = 0.0
        for key in sorted(set(a) | set(b), key=str):
            if key not in a or key not in b:
                problems.append(f'{path}.{key}: only in {"a" if key in a else "b"}')
                worst = math.inf
                continue
            worst = max(worst, _tree(a[key], b[key], atol, rtol, f'{path}.{key}', problems))
        return worst
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            problems.append(f'{path}: length {len(a)} != {len(b)}')
            return math.inf
        worst = 0.0
        for i, (x, y) in enumerate(zip(a, b)):
            worst = max(worst, _tree(x, y, atol, rtol, f'{path}[{i}]', problems))
        return worst
    if _number(a) and _number(b):
        ok, diff = _close(float(a), float(b), atol, rtol)
        if not ok:
            problems.append(f'{path}: {a!r} != {b!r}')
        return diff
    if a != b:
        problems.append(f'{path}: {a!r} != {b!r}')
        return math.inf
    return 0.0


def _compare_tree(a: Any, b: Any, atol: float, rtol: float, label: str) -> dict[str, Any]:
    problems: list[str] = []
    worst = _tree(a, b, atol, rtol, label, problems)
    return _item(not problems, worst, problems=problems[:20], n_problems=len(problems))


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    body = json.loads(path.read_text(encoding='utf-8'))
    return body if isinstance(body, dict) else None


def _history(run: Path) -> dict[str, dict[str, list[Any]]] | None:
    body = _load_json(run / ASSETS_DIRNAME / kinds.TRACKER_STATE)
    if body is None:
        return None
    out: dict[str, dict[str, list[Any]]] = {}
    for split, meters in (body.get('splits') or {}).items():
        for name, meter in (meters or {}).items():
            if isinstance(meter, dict) and isinstance(meter.get('history'), list):
                out.setdefault(str(split), {})[str(name)] = list(meter['history'])
    return out


def _checkpoint(run: Path, name: str) -> dict[str, Any] | None:
    import torch

    path = run / ASSETS_DIRNAME / kinds.CHECKPOINTS / f'{name}.pt'
    if not path.is_file():
        return None
    payload = torch.load(path, map_location='cpu', weights_only=False)
    return payload if isinstance(payload, dict) else {'model': payload}


def compare_runs(
    run_a: Path | str,
    run_b: Path | str,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
    checkpoints: list[str] | None = None,
) -> dict[str, Any]:
    a, b = Path(run_a).resolve(), Path(run_b).resolve()
    for run in (a, b):
        if not run.is_dir():
            raise FileNotFoundError(f'missing Run dir: {run}')
    if atol < 0 or rtol < 0:
        raise ValueError('atol and rtol must be >= 0')
    items: dict[str, dict[str, Any]] = {}

    result_a, result_b = _load_json(a / RESULT_NAME), _load_json(b / RESULT_NAME)
    if result_a is None or result_b is None:
        items['metrics'] = _missing('a' if result_a is None else 'b')
    else:
        ma, mb = result_a.get('metrics') or {}, result_b.get('metrics') or {}
        keys = sorted(key for key in set(ma) & set(mb) if _number(ma[key]) and _number(mb[key]))
        items['metrics'] = _compare_tree(
            {key: ma[key] for key in keys}, {key: mb[key] for key in keys}, atol, rtol, 'metrics',
        )
        items['metrics']['keys'] = keys

    history_a, history_b = _history(a), _history(b)
    if history_a is None or history_b is None:
        items['history'] = _missing('a' if history_a is None else 'b')
    else:
        items['history'] = _compare_tree(history_a, history_b, atol, rtol, 'history')

    for name in checkpoints or ['latest']:
        payload_a, payload_b = _checkpoint(a, name), _checkpoint(b, name)
        for part in CHECKPOINT_PARTS:
            key = f'checkpoint.{name}.{part}'
            if payload_a is None or payload_b is None:
                items[key] = _missing('a' if payload_a is None else 'b')
                continue
            left, right = payload_a.get(part), payload_b.get(part)
            if left is None and right is None:
                continue
            if left is None or right is None:
                items[key] = _missing('a' if left is None else 'b')
                continue
            items[key] = _compare_tree(left, right, atol, rtol, part)

    return {
        'run_a': str(a),
        'run_b': str(b),
        'atol': atol,
        'rtol': rtol,
        'items': items,
        'passed': all(item['passed'] for item in items.values()),
    }


def format_compare(report: dict[str, Any]) -> str:
    lines = []
    for key, item in report['items'].items():
        state = 'pass' if item['passed'] else 'FAIL'
        if item.get('missing'):
            detail = f'missing in {item["missing"]}'
        else:
            detail = f'max abs diff {item["max_abs_diff"]:g}'
        lines.append(f'{state}  {key}  {detail}')
    lines.append('passed' if report['passed'] else 'failed')
    return '\n'.join(lines) + '\n'


def write_compare(path: Path | str, report: dict[str, Any]) -> Path:
    def finite(value: Any) -> Any:
        return None if isinstance(value, float) and not math.isfinite(value) else value

    clean = {
        **report,
        'items': {
            key: {name: finite(value) for name, value in item.items()}
            for key, item in report['items'].items()
        },
    }
    return atomic_write_text(Path(path), json.dumps(clean, indent=2, ensure_ascii=False))
