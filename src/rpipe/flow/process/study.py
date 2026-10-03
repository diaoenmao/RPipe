"""Study-scoped process: sibling results + history mean/std/min/max. One process after launch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rpipe.flow.process.aggregate import (
    attach_histories,
    collect_from_index,
    process_path,
    summarize,
)
from rpipe.flow.process.curves import write_learning_curves
from rpipe.structure.artifact._atomic import atomic_write_text
from rpipe.structure.artifact.index import index_path, load_index
from rpipe.structure.artifact.result.format import encode_result


def run_study(study_dir: Path | str) -> dict[str, Any]:
    study_dir = Path(study_dir)
    try:
        index = load_index(study_dir)
        if not isinstance(index.get('experiments'), list):
            raise TypeError('Study index must contain an experiments list')
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(
            f'Cannot process Study without a usable current index: {index_path(study_dir)}; '
            f'check study.yaml and rebuild with python -m rpipe make "{study_dir}"'
        ) from exc

    groups = collect_from_index(study_dir, index)
    study_name = str(index.get('study') or study_dir.name)

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
