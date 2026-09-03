"""Atomic text write (temp file then replace)."""

from __future__ import annotations

import time
from pathlib import Path


def atomic_write_text(path: Path | str, text: str, encoding: str = 'utf-8') -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + '.tmp')
    tmp.write_text(text, encoding=encoding)
    last_error: OSError | None = None
    for attempt in range(6):
        try:
            tmp.replace(target)
            return target
        except PermissionError as error:
            last_error = error
            time.sleep(0.05 * (attempt + 1))
    if last_error is not None:
        raise last_error
    return target
