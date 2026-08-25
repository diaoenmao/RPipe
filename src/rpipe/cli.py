"""Thin CLI: expand study.yaml, write index, launch Flow.

Not a library pillar. Structure + Flow already cover Run execution;
this module only maps a Study directory onto those two.
"""

from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path
from typing import Any

import yaml

from rpipe.flow import FlowContext, FlowRunner
from rpipe.structure.artifact import (
    artifact_layout,
    build_index,
    experiment_entries,
    load_config,
    write_config,
    write_index,
)
from rpipe.structure.artifact.layout import ensure_study_layout
from rpipe.structure.control import experiment_config_from_mapping, run_config_from_merge


def set_dotted(mapping: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split('.')
    cur: dict[str, Any] = mapping
    for key in keys[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[keys[-1]] = value


def cartesian(axes: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not axes:
        return [{}]
    keys = list(axes.keys())
    values = [axes[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def format_description(template: str, patch: dict[str, Any], experiment: str) -> str:
    class _Safe(dict):
        def __missing__(self, key: str) -> str:
            return '{' + key + '}'

    flat = {'experiment': experiment}
    for dotted, value in patch.items():
        if dotted in ('description', 'tags'):
            continue
        flat[dotted] = value
        flat[str(dotted).split('.')[-1]] = value
    try:
        return template.format_map(_Safe(flat))
    except Exception:
        return template


def match_when(axis_flat: dict[str, Any], when: dict[str, Any]) -> bool:
    for key, expected in when.items():
        if axis_flat.get(key) != expected:
            return False
    return True


def axis_keys_from_study(study: dict[str, Any]) -> list[str]:
    raw = dict(study.get('axes') or {})
    raw.pop('seed', None)
    return list(raw.keys())


def expand_patches(study: dict[str, Any]) -> list[dict[str, Any]]:
    """``axes`` → Experiment cells; ``seeds`` → Runs under each cell."""
    fixed = dict(study.get('fixed') or {})
    raw_axes = dict(study.get('axes') or {})
    axis_seed_values = raw_axes.pop('seed', None)
    axes = raw_axes
    tag_rules = list(study.get('tags') or [])
    exp_name = ''
    exp = study.get('experiment')
    if isinstance(exp, dict):
        exp_name = str(exp.get('name') or '')
    elif isinstance(exp, str):
        exp_name = exp
    desc_t = str(study.get('run_description') or '{experiment} seed={seed}')

    if study.get('seeds') is not None:
        seeds = list(study['seeds'])
    elif axis_seed_values is not None:
        seeds = list(axis_seed_values)
    elif 'seed' in fixed:
        seeds = [fixed['seed']]
    else:
        seeds = [0]

    fixed_no_seed = {k: v for k, v in fixed.items() if k != 'seed'}

    patches: list[dict[str, Any]] = []
    for combo in cartesian(axes):
        for seed in seeds:
            patch: dict[str, Any] = copy.deepcopy(fixed_no_seed)
            axis_flat: dict[str, Any] = {}
            for dotted, value in combo.items():
                set_dotted(patch, dotted, value)
                axis_flat[dotted] = value
            patch['seed'] = seed
            axis_flat['seed'] = seed
            tags: list[str] = []
            for rule in tag_rules:
                if not isinstance(rule, dict):
                    continue
                when = dict(rule.get('when') or {})
                when_no_seed = {k: v for k, v in when.items() if k != 'seed'}
                if match_when(axis_flat, when_no_seed) and (
                    'seed' not in when or axis_flat.get('seed') == when.get('seed')
                ):
                    tags.extend(list(rule.get('tags') or []))
            if tags:
                patch['tags'] = tags
            patch['description'] = format_description(
                desc_t, {**axis_flat, **patch}, exp_name
            )
            patches.append(patch)
    return patches


def load_study_yaml(study_dir: Path) -> dict[str, Any]:
    path = study_dir / 'study.yaml'
    if not path.is_file():
        raise FileNotFoundError(f'missing study.yaml: {path}')
    data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    if not isinstance(data, dict):
        raise ValueError(f'study.yaml must be a mapping: {path}')
    return data


def write_run_configs(study_dir: Path, patches: list[dict[str, Any]]) -> list[Path]:
    base_path = study_dir / 'experiment_config.yaml'
    if not base_path.is_file():
        raise FileNotFoundError(f'missing experiment_config.yaml: {base_path}')
    base = experiment_config_from_mapping(load_config(base_path))
    written: list[Path] = []
    for patch in patches:
        run = run_config_from_merge(base, patch)
        cfg = run.to_mapping()
        layout = artifact_layout(study_dir, run.id)
        written.append(write_config(layout.config_path, cfg))
    return written


def write_study_index(
    study_dir: Path,
    study: dict[str, Any],
    config_paths: list[Path],
) -> Path:
    loaded = [(path, load_config(path)) for path in config_paths]
    experiments = experiment_entries(
        configs=loaded,
        axis_keys=axis_keys_from_study(study),
        study_dir=study_dir,
    )
    index = build_index(
        study=str(study.get('study') or study_dir.name),
        description=str(study.get('description') or ''),
        experiments=experiments,
    )
    return write_index(study_dir, index)


def launch_runs(
    study_dir: Path,
    config_paths: list[Path],
    phases: list[str] | None = None,
) -> list[Path]:
    results: list[Path] = []
    runner = FlowRunner(phases=phases)
    for path in config_paths:
        run_dir = path.parent.name
        layout = artifact_layout(study_dir, run_dir)
        cfg = load_config(layout.config_path)
        ctx = FlowContext(study_dir=study_dir, layout=layout, config=cfg)
        results.append(runner.run(ctx))
    return results


def run_study(
    study_dir: Path | str,
    *,
    skip_launch: bool = False,
    phases: list[str] | None = None,
) -> dict[str, Any]:
    """Expand → write index (grouped by Experiment) → launch Flow."""
    study_dir = ensure_study_layout(Path(study_dir).resolve())
    study = load_study_yaml(study_dir)
    patches = expand_patches(study)
    configs = write_run_configs(study_dir, patches)
    index_file = write_study_index(study_dir, study, configs)
    result_paths: list[Path] = []
    if not skip_launch:
        result_paths = launch_runs(study_dir, configs, phases=phases)
    return {
        'study_dir': study_dir,
        'index': index_file,
        'configs': configs,
        'results': result_paths,
    }


def _add_run_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('study_dir', type=Path, help='Path to studies/<name>/')
    parser.add_argument(
        '--skip-launch',
        action='store_true',
        help='Only write Configs + index.json',
    )
    parser.add_argument(
        '--phases',
        default='',
        help='comma-separated Flow phases (default: all)',
    )


def _execute_run(args: argparse.Namespace) -> int:
    phases = [s.strip() for s in args.phases.split(',') if s.strip()] or None
    out = run_study(args.study_dir, skip_launch=args.skip_launch, phases=phases)
    for path in out['configs']:
        print(path)
    print(out['index'])
    for path in out['results']:
        print(path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='rpipe',
        description='RPipe: expand a Study directory and run Flow',
    )
    sub = parser.add_subparsers(dest='cmd', required=True)

    run_p = sub.add_parser('run', help='Expand study.yaml, write index, launch Flow')
    _add_run_flags(run_p)

    study_p = sub.add_parser('study', help='Alias for run (kept for existing scripts)')
    study_sub = study_p.add_subparsers(dest='study_cmd', required=True)
    study_run = study_sub.add_parser('run', help='Same as ``rpipe run``')
    _add_run_flags(study_run)

    args = parser.parse_args(argv)
    if args.cmd == 'run' or (args.cmd == 'study' and args.study_cmd == 'run'):
        return _execute_run(args)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
