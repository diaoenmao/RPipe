from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

from rpipe.config import load_yaml, repo_root


def suites_path() -> Path:
    return repo_root() / 'configs' / 'suites' / 'default.yaml'


def list_suites(path=None) -> list[str]:
    raw = load_yaml(path or suites_path())
    return sorted((raw.get('suites') or {}).keys())


def load_suite(name: str, path=None) -> dict[str, Any]:
    raw = load_yaml(path or suites_path())
    suites = raw.get('suites') or {}
    if name not in suites:
        raise ValueError('Unknown suite {!r}. Available: {}'.format(name, sorted(suites)))
    suite = dict(suites[name])
    suite['name'] = name
    suite.setdefault('init_seed', 0)
    suite.setdefault('num_experiments', 1)
    suite.setdefault('hyper', {})
    suite.setdefault('description', '')
    return suite


def expand_control_names(data_names, model_names) -> list[str]:
    return ['_'.join(pair) for pair in itertools.product(data_names, model_names)]


def expand_controls(suite) -> list[tuple[str, str]]:
    control_names = expand_control_names(suite['data_names'], suite['model_names'])
    seeds = [str(s) for s in range(suite['init_seed'], suite['init_seed'] + suite['num_experiments'])]
    return list(itertools.product(seeds, control_names))
