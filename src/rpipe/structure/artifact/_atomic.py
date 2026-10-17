"""Atomic text write (unique temp file then replace)."""

from __future__ import annotations

from contextlib import closing

import time
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path


@contextmanager
def _owned_stream(manager):
    with ExitStack() as stack:
        value = manager.__enter__()

        def close(exc_type, exc, tb):
            try:
                return manager.__exit__(exc_type, exc, tb)
            except OSError:
                if exc_type is None:
                    raise

        stack.push(close)
        yield value


def _read_text(path: Path | str, *, errors: str = 'strict') -> str:
    with _owned_stream(closing(Path(path).open('r', encoding='utf-8', errors=errors))) as handle:
        return handle.read()


def _read_bytes(path: Path | str) -> bytes:
    with _owned_stream(closing(Path(path).open('rb'))) as handle:
        return handle.read()


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
    owned = False
    try:
        try:
            mode = target.stat().st_mode
        except FileNotFoundError:
            mode = None
        tmp.touch(exist_ok=False)
        owned = True
        with _owned_stream(closing(tmp.open('w', encoding=encoding, newline=newline))) as handle:
            handle.write(text)
        if mode is not None:
            tmp.chmod(mode)
        return atomic_replace(tmp, target)
    finally:
        if owned:
            try:
                tmp.unlink(missing_ok=True)
            except PermissionError:
                try:
                    tmp.chmod(0o600)
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
            except OSError:
                pass
