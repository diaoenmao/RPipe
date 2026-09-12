"""Study-scoped process: sibling results + history mean/std/min/max. One process after launch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rpipe.flow.process.aggregate import (
    attach_histories,
    collect_from_index,
    collect_from_runs,
    process_path,
    summarize,
)
from rpipe.flow.process.curves import write_learning_curves
from rpipe.structure.artifact._atomic import atomic_write_text
from rpipe.structure.artifact.index import load_index
from rpipe.structure.artifact.result.format import encode_result


def run_study(study_dir: Path | str) -> dict[str, Any]:
    study_dir = Path(study_dir)
    index = None
    try:
        index = load_index(study_dir)
    except (OSError, TypeError):
        index = None

    if index is not None:
        groups = collect_from_index(study_dir, index)
        study_name = str(index.get('study') or study_dir.name)
    else:
        groups = collect_from_runs(study_dir)
        study_name = study_dir.name

    body = summarize(groups, study=study_name, source_run=None)
    body['scope'] = 'study'
    attach_histories(study_dir, groups, body.get('experiments') or [])
    figure = write_learning_curves(study_dir, index, title=f'{study_name} · mean ± std')
    if figure is not None:
        body['figures'] = {
            'learning_curves': str(figure.relative_to(study_dir)).replace('\\', '/')
        }
    atomic_write_text(process_path(study_dir), encode_result(body))
    return body
