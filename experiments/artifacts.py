"""Structured run artifacts for downstream AI report generation.

This module intentionally does NOT write narrative markdown reports.
It dumps machine-readable manifests that an LLM/agent can consume later.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from rpipe.schema import assert_valid_run_manifest
from rpipe.system import makedir_exist_ok


def write_run_manifest(
    suite: dict[str, Any],
    *,
    stages: list[str],
    device: str,
    result_paths: list[str] | None = None,
    plot_paths: list[str] | None = None,
    notes: list[str] | None = None,
    output_root: str = 'output',
) -> str:
    report_dir = Path(output_root) / 'artifacts'
    makedir_exist_ok(str(report_dir))
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = report_dir / f"{suite['name']}_{stamp}.json"

    plot_dir = Path(output_root) / 'vis' / 'png' / 'lc'
    if plot_paths is None and plot_dir.is_dir():
        plot_paths = [str(plot_dir / p) for p in sorted(plot_dir.iterdir()) if p.suffix == '.png']

    payload = {
        'schema': 'rpipe.run_manifest.v1',
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'suite': {
            'name': suite['name'],
            'description': suite.get('description', ''),
            'data_names': suite.get('data_names', []),
            'model_names': suite.get('model_names', []),
            'num_experiments': suite.get('num_experiments'),
            'init_seed': suite.get('init_seed'),
            'hyper': suite.get('hyper') or {},
        },
        'run': {
            'stages': stages,
            'device': device,
            'cwd': os.getcwd(),
        },
        'artifacts': {
            'result_paths': result_paths or [],
            'plot_paths': plot_paths or [],
            'excel': [
                str(Path(output_root) / 'result' / 'result_mean.xlsx'),
                str(Path(output_root) / 'result' / 'result_history.xlsx'),
            ],
            'processed_result': str(Path(output_root) / 'result' / 'processed_result'),
            'stats_dir': str(Path(output_root) / 'stats'),
            'exp_dir': str(Path(output_root) / 'exp'),
        },
        'notes': notes or [],
        'report_hint': (
            'Feed this manifest + result blobs / Excel into an LLM agent to draft the experiment report. '
            'Do not expect rule-based markdown from RPipe itself.'
        ),
    }
    assert_valid_run_manifest(payload)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print('[artifacts] wrote {}'.format(path))
    return str(path)
