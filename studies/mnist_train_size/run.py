"""Study mnist_train_size — prefer: python -m rpipe run studies/mnist_train_size"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main() -> int:
    from rpipe.cli import run_study

    out = run_study(Path(__file__).resolve().parent)
    for path in out['configs']:
        print(path)
    print(out['index'])
    for path in out['results']:
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
