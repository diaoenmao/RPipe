from pathlib import Path

import pytest

from rpipe.flow.cli import run_study
from rpipe.flow.context import FlowContext
from rpipe.flow.runner import PHASES, FlowRunner
from rpipe.structure.artifact import artifact_layout, load_config, load_result
from rpipe.structure.artifact.provenance import ProvenanceChangedError, check_frozen, load_provenance, provenance_changes
from rpipe.structure.make import expand_study, load_study_yaml

pytestmark = [pytest.mark.integration, pytest.mark.content, pytest.mark.p1,
              pytest.mark.flow_layer, pytest.mark.module_runner,
              pytest.mark.cost(cost_class='c1'), pytest.mark.result_type('categorical', detail='summary')]


def _study(root, *, freeze=False):
    study = root / 'stages'
    study.mkdir()
    (study / 'study.yaml').write_text('study: stages\nseeds: [0, 1]\nflow: {study_phases: true}\n'
                                      f'freeze: {str(freeze).lower()}\n', encoding='utf-8')
    (study / 'experiment_config.yaml').write_text(
        'experiment: stages\ndata: {name: Toy, source: stub}\nmodel: {name: linear}\n'
        'algorithm: {mode: train, num_steps: 2}\nsystem: {device: cpu}\n', encoding='utf-8')
    return study


def _hook(study, phase, code):
    path = study / phase / '__init__.py'
    path.parent.mkdir(exist_ok=True)
    path.write_text(code, encoding='utf-8')
    return path


def test_full_cpu_chain_runs_library_then_study_and_aggregates_once(tmp_path):
    study = _study(tmp_path)
    assertions = {'prepare': 'assert "data" in ctx.state and "model" in ctx.state',
                  'execute': 'assert "execute" in ctx.state',
                  'collect': 'assert "collected" in ctx.state',
                  'summarize': 'ctx.state["result_draft"]["study_extension"] = True',
                  'write': 'assert "result_draft" in ctx.state and not ctx.layout.result_path.exists()',
                  'process': 'assert (ctx.layout.root / "process.json").is_file()'}
    for phase in PHASES:
        _hook(study, phase, 'from pathlib import Path\ndef run(ctx):\n'
              '    if ctx.scope == "study":\n'
              '        assert ctx.state["process"]["complete"]\n'
              '        mark = ctx.study_dir / "study-process.txt"\n'
              '        mark.write_text(mark.read_text() + "done\\n" if mark.exists() else "done\\n")\n'
              '        return\n'
              f'    {assertions[phase]}\n'
              '    mark = ctx.layout.assets_dir / "study-phases.txt"\n'
              f'    mark.write_text((mark.read_text() if mark.exists() else "") + "{phase}\\n")\n')
    out = run_study(study)
    assert (study / 'study-process.txt').read_text() == 'done\n'
    for result_path in out['results']:
        assert load_result(result_path)['study_extension'] is True
        assert (result_path.parent / 'assets' / 'study-phases.txt').read_text().splitlines() == list(PHASES)
    files = load_provenance(study)['files']
    assert all(f'{phase}/__init__.py' in files for phase in PHASES)


def test_prepare_before_runs_before_recipe_and_factory(tmp_path):
    study = _study(tmp_path)
    _hook(study, 'prepare', '''def before(ctx):
    assert ctx.control is not None and "system" in ctx.state
    assert "data" not in ctx.state and "model" not in ctx.state
    (ctx.layout.assets_dir / "before.txt").write_text("ready")

def run(ctx):
    assert "data" in ctx.state and "model" in ctx.state
''')
    (study / 'recipe.py').write_text('''def register(ctx):
    assert (ctx.assets_dir / "before.txt").read_text() == "ready"
''', encoding='utf-8')
    with (study / 'study.yaml').open('a', encoding='utf-8') as stream:
        stream.write('recipe: recipe.py\n')
    out = run_study(study)
    assert all(load_result(path)['status'] == 'succeeded' for path in out['results'])


@pytest.mark.parametrize('stale_success', [False, True])
def test_write_gate_failure_cannot_leave_success_result(tmp_path, stale_success):
    from rpipe.structure.artifact.result import write_result

    study = _study(tmp_path)
    _hook(study, 'write', '''def run(ctx):
    assert "result_draft" in ctx.state
    (ctx.layout.assets_dir / "gate.txt").write_text("rejected")
    raise RuntimeError("write gate rejected")
''')
    config = expand_study(study)['configs'][0]
    ctx = FlowContext(study, artifact_layout(study, config.parent.name), load_config(config))
    if stale_success:
        write_result(ctx.layout.result_path, {'status': 'succeeded', 'control': {}, 'paths': {}, 'metrics': {'stale': 1}})
    with pytest.raises(RuntimeError, match='write gate rejected'):
        FlowRunner().run(ctx)
    assert load_result(ctx.layout.result_path)['status'] == 'failed'
    assert (ctx.layout.assets_dir / 'gate.txt').read_text() == 'rejected'
    assert not (ctx.layout.root / 'process.json').exists()


def test_write_extension_artifacts_are_registered_before_commit(tmp_path):
    study = _study(tmp_path)
    _hook(study, 'write', '''def run(ctx):
    (ctx.layout.assets_dir / "evidence.txt").write_text("observed")
    ctx.state["result_draft"]["evidence_complete"] = True
''')
    out = run_study(study)
    for path in out['results']:
        result = load_result(path)
        assert result['evidence_complete'] is True
        assert 'evidence.txt' in result['paths']['asset_files']


def test_study_prepare_failure_retains_original_error_and_failed_result(tmp_path):
    study = _study(tmp_path)
    _hook(study, 'prepare', 'def run(ctx):\n    raise RuntimeError("Study gate rejected")\n')
    config = expand_study(study)['configs'][0]
    ctx = FlowContext(study, artifact_layout(study, config.parent.name), load_config(config))
    with pytest.raises(RuntimeError, match='Study gate rejected'):
        FlowRunner().run(ctx)
    assert load_result(ctx.layout.result_path)['status'] == 'failed'
    assert 'Study gate rejected' in (ctx.layout.assets_dir / 'logs' / 'run.log').read_text(encoding='utf-8')


def test_study_process_failure_preserves_succeeded_runs_and_aggregation(tmp_path):
    study = _study(tmp_path)
    _hook(study, 'process', 'def run(ctx):\n    if ctx.scope == "study":\n        raise RuntimeError("final gate failed")\n')
    with pytest.raises(RuntimeError, match='final gate failed'):
        run_study(study)
    assert (study / 'process.json').is_file()
    assert all(load_result(path)['status'] == 'succeeded' for path in (study / 'runs').glob('*/result.json'))


@pytest.mark.parametrize('change', ['edit', 'add', 'delete'])
def test_freeze_includes_stage_helpers_and_blocks_before_selected_execute(tmp_path, monkeypatch, change):
    study = _study(tmp_path, freeze=True)
    _hook(study, 'prepare', 'def run(ctx):\n    pass\n')
    helper = study / 'prepare' / 'helper.py'
    if change != 'add':
        helper.write_text('VALUE = 1\n', encoding='utf-8')
    config = expand_study(study)['configs'][0]
    if change == 'delete':
        helper.unlink()
    else:
        helper.write_text('VALUE = 2\n', encoding='utf-8')
    declared = load_study_yaml(study)
    assert 'prepare/helper.py' in provenance_changes(study, declared)
    with pytest.raises(ProvenanceChangedError):
        check_frozen(study, declared)
    import rpipe.flow.execute as execute
    called = []
    monkeypatch.setattr(execute, 'run', lambda ctx: called.append(True))
    ctx = FlowContext(study, artifact_layout(study, config.parent.name), load_config(config))
    with pytest.raises(ProvenanceChangedError):
        FlowRunner(['execute']).run(ctx)
    assert not called
