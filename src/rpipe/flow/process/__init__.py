"""process: Run-individual after write; Study-total via run_study / ``rpipe process``."""

from rpipe.flow.process.aggregate import process_path, run_process_path
from rpipe.flow.process.individual import run
from rpipe.flow.process.study import run_study

__all__ = ['process_path', 'run', 'run_process_path', 'run_study']
