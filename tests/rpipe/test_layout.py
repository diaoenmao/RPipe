"""tests/rpipe must copy the src/rpipe directory skeleton (TESTING.md §5)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.location,
    pytest.mark.p1,
]

REPO = Path(__file__).resolve().parents[2]
SRC_RPIPE = REPO / 'src' / 'rpipe'
TESTS_RPIPE = REPO / 'tests' / 'rpipe'


def _package_dirs(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    rels: set[str] = set()
    for path in root.rglob('*'):
        if not path.is_dir():
            continue
        if '__pycache__' in path.parts:
            continue
        rel = path.relative_to(root).as_posix()
        if rel == '.':
            continue
        rels.add(rel)
    return rels


def test_tests_rpipe_directory_tree_matches_src_rpipe():
    assert SRC_RPIPE.is_dir(), f'missing source package root: {SRC_RPIPE}'
    assert TESTS_RPIPE.is_dir(), f'missing tests mirror root: {TESTS_RPIPE}'

    src_dirs = _package_dirs(SRC_RPIPE)
    test_dirs = _package_dirs(TESTS_RPIPE)

    missing_in_tests = sorted(src_dirs - test_dirs)
    extra_in_tests = sorted(test_dirs - src_dirs)

    assert not missing_in_tests, (
        'tests/rpipe is missing directories present under src/rpipe:\n'
        + '\n'.join(f'  - {p}' for p in missing_in_tests)
    )
    assert not extra_in_tests, (
        'tests/rpipe has directories not present under src/rpipe:\n'
        + '\n'.join(f'  - {p}' for p in extra_in_tests)
    )
