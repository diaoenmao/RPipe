"""Atomic text write (unique temp file then replace)."""

from __future__ import annotations

import time
import uuid
from pathlib import Path


def atomic_replace(source: Path | str, target: Path | str) -> Path:
    """Replace once, allowing a short Windows sharing/access denial to clear."""
    source, target = Path(source), Path(target)
    for attempt in range(6):
        try:
            source.replace(target)
            return target
        except OSError as error:
            if getattr(error, 'winerror', None) not in (5, 32, 33) or attempt == 5:
                raise
            time.sleep(0.05 * (attempt + 1))


def atomic_write_text(path: Path | str, text: str, encoding: str = 'utf-8', *, newline: str | None = None) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f'{uuid.uuid4().hex}.tmp')
    try:
        try:
            mode = target.stat().st_mode
        except FileNotFoundError:
            mode = None
        tmp.write_text(text, encoding=encoding, newline=newline)
        if mode is not None:
            tmp.chmod(mode)
        return atomic_replace(tmp, target)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
