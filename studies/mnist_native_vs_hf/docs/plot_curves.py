"""Redraw Study learning curves (same path process writes)."""

from pathlib import Path

from rpipe.flow.process.curves import write_learning_curves
from rpipe.structure.artifact.index import load_index


def main() -> None:
    study = Path(__file__).resolve().parents[1]
    dest = write_learning_curves(study, load_index(study), title='mnist_native_vs_hf · mean ± std')
    print(dest)


if __name__ == '__main__':
    main()
