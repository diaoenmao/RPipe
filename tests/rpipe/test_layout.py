"""tests/rpipe must copy the src/rpipe directory skeleton (TESTING.md 目录与映射)."""

from __future__ import annotations

from pathlib import Path
import re

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.location,
    pytest.mark.p1,
    pytest.mark.cost(cost_class='c1'),
    pytest.mark.result_type('categorical', detail='summary'),
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


def _documented_paths():
    text = (REPO / 'docs/code/layout.md').read_text(encoding='utf-8')
    return set(re.findall(r'^\| `(<project>/[^`]+)` \|', text, re.MULTILINE))


def _document_path(path, project):
    names = {'sample-study': '<study>', 'sample-run': '<run_id>', 'mnist': '<dataset>'}
    return '<project>/' + '/'.join(names.get(part, part) for part in path.relative_to(project).parts)


@pytest.mark.parametrize('member', [
    'study', 'runs', 'run', 'config', 'result', 'assets', 'docs',
    'shared', 'data', 'model', 'index', 'provenance', 'activity',
    'study_process', 'run_process', 'numbers', 'figures', 'curves', 'scripts', 'jobs',
    'run_log', 'declaration', 'base_config',
    'TRACKER', 'TRACKER_STATE', 'TRACKER_JSONL', 'LOGS', 'CHECKPOINTS', 'SAMPLES', 'CACHE', 'WEIGHTS',
])
def test_default_paths_match_layout_documentation(tmp_path, member):
    from rpipe.flow.process.aggregate import process_path, run_process_path
    from rpipe.flow.process.curves import figures_dir, learning_curves_path
    from rpipe.structure.artifact.activity import activity_path
    from rpipe.structure.artifact.asset import kinds
    from rpipe.structure.artifact.index import index_path
    from rpipe.structure.artifact.layout import artifact_layout
    from rpipe.structure.artifact.provenance import DECLARATION_FILES, provenance_path
    from rpipe.structure.artifact.readout.report import numbers_path
    from rpipe.structure.artifact.readout.status import log_path
    from rpipe.structure.make.schedule import JOBS_NAME, scripts_dir

    project = tmp_path.resolve()
    study = project / 'studies' / 'sample-study'
    layout = artifact_layout(study, 'sample-run')
    scripts = scripts_dir(study)
    actual = {
        'study': study, 'runs': layout.root.parent, 'run': layout.root,
        'config': layout.config_path, 'result': layout.result_path, 'assets': layout.assets_dir,
        'docs': layout.docs_dir, 'shared': layout.shared_dir,
        'data': layout.shared_data_dir, 'model': layout.shared_model_dir,
        'index': index_path(study), 'provenance': provenance_path(study),
        'activity': activity_path(study), 'study_process': process_path(study),
        'run_process': run_process_path(layout.root), 'numbers': numbers_path(study),
        'figures': figures_dir(study), 'curves': learning_curves_path(study),
        'scripts': scripts, 'jobs': scripts / JOBS_NAME,
        'run_log': log_path(study, {}, layout.root.name),
        'declaration': study / DECLARATION_FILES[0], 'base_config': study / DECLARATION_FILES[1],
    }
    for name in ('TRACKER', 'TRACKER_STATE', 'TRACKER_JSONL', 'LOGS', 'CHECKPOINTS', 'SAMPLES', 'CACHE', 'WEIGHTS'):
        actual[name] = layout.assets_dir / getattr(kinds, name)
    documented = {row.rstrip('/') for row in _documented_paths()}
    path = _document_path(actual[member], project)
    assert path in documented, f'{member}: undocumented default path {path}; update layout.md first'


@pytest.mark.parametrize('kind', ['data', 'model'])
def test_private_shared_paths_match_layout_documentation(tmp_path, kind):
    from rpipe.structure.artifact.layout import artifact_layout
    from rpipe.structure.artifact.shared import shared_paths

    project = tmp_path.resolve()
    study = project / 'studies' / 'sample-study'
    study.mkdir(parents=True)
    (study / 'study.yaml').write_text(f'shared: {{{kind}: study}}\n', encoding='utf-8')
    layout = artifact_layout(study, 'sample-run')
    actual = getattr(layout, f'shared_{kind}_dir')
    assert actual == shared_paths(study)[kind] == study / 'shared' / kind
    assert _document_path(actual, project) in {row.rstrip('/') for row in _documented_paths()}
    other = 'model' if kind == 'data' else 'data'
    assert shared_paths(study)[other] == project / 'shared' / other


def test_profile_path_matches_documented_private_directory(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import torch
    from rpipe.structure.data import profile

    project = tmp_path.resolve()
    study = project / 'studies' / 'sample-study'
    study.mkdir(parents=True)
    (study / 'study.yaml').write_text('fixed: {data: {name: MNIST}}\n', encoding='utf-8')
    data = SimpleNamespace(_loaders={'train': [(torch.zeros(1, 1, 2, 2), torch.zeros(1, dtype=torch.long))]})
    monkeypatch.setattr(profile.DataFactory, 'build', lambda *args, **kwargs: data)
    [written] = profile.profile_study(study)
    assert _document_path(written, project) in _documented_paths()
    assert written.is_file()
    assert not (project / 'shared' / 'data' / 'mnist' / profile.STATS_NAME).exists()


def test_package_readme_and_release_material_versions_agree():
    import rpipe

    metadata = (REPO / 'pyproject.toml').read_text(encoding='utf-8')
    version = re.search(r'^version = "([^"]+)"$', metadata, re.MULTILINE).group(1)
    assert rpipe.__version__ == version
    readme = (REPO / 'README.md').read_text(encoding='utf-8')
    assert f'当前模板版本为 **{version}' in readme
    release = REPO / 'docs/releases' / f'v{version}.md'
    assert f'# RPipe v{version} ' in release.read_text(encoding='utf-8')
