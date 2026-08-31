"""Atomic text write (temp file then replace)."""

from __future__ import annotations

from pathlib import Path


def atomic_write_text(path: Path | str, text: str, encoding: str = 'utf-8') -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + '.tmp')
    tmp.write_text(text, encoding=encoding)
    tmp.replace(target)
    return target
