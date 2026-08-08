"""RPipe package entry.

The installable library lives here. Experiment orchestration is a sibling
package: ``python -m experiments`` / ``rpipe-run``.
"""

from __future__ import annotations


def main(argv=None):
    print(
        'rpipe is a library package (data / model / algorithm / system).\n'
        'Run experiment suites with:\n'
        '  python -m experiments --suite smoke --device cpu\n'
        '  rpipe-run --suite smoke --device cpu'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
