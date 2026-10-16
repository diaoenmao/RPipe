from pathlib import Path

import pytest

from rpipe.flow import cli
from rpipe.flow.cli import launch_one, run_study
from rpipe.structure.artifact import load_result
from rpipe.structure.artifact.provenance import (
    ProvenanceChangedError,
    check_frozen,
    load_provenance,
    provenance_changes,
)
from rpipe.structure.make import expand_study, load_study_yaml

pytestmark = [
    pytest.mark.integration,
    pytest.mark.content,
    pytest.mark.p1,
    pytest.mark.flow_layer,
    pytest.mark.module_cli,
    pytest.mark.cost(cost_class='c1'),
    pytest.mark.result_type('categorical', detail='summary'),
]

RECIPE = '''
from pathlib import Path

from helper import MARK
from rpipe.structure.model.factory import ModelRegistry, _build_linear


def register(ctx):
    def build(model_config, assets_dir, data_meta=None):
        Path(ctx.assets_dir, MARK).write_text(f'{ctx.run_id} {ctx.seed}', encoding='utf-8')
        return _build_linear(model_config, assets_dir, data_meta)

    ModelRegistry.register('linear', 'recipe_src', build)
'''


def _study(root: Path, *, freeze: bool = False, recipe: str | None = RECIPE) -> Path:
    study = root / 'hooked'
    study.mkdir()
    lines = ['study: hooked', 'seeds: [0, 1]', 'provenance: {include: [notes/*.txt]}']
    if recipe is not None:
        lines.append('recipe: recipe.py')
        (study / 'recipe.py').write_text(recipe, encoding='utf-8')
        (study / 'helper.py').write_text("MARK = 'recipe_mark.txt'\n", encoding='utf-8')
    if freeze:
        lines.append('freeze: true')
    (study / 'study.yaml').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (study / 'notes').mkdir()
    (study / 'notes' / 'a.txt').write_text('a\n', encoding='utf-8')
    (study / 'experiment_config.yaml').write_text(
        'experiment: demo\n'
        'data: {name: Toy, source: stub}\n'
        'model: {name: linear, source: recipe_src}\n'
        'algorithm: {mode: train, num_steps: 2}\n'
        'system: {device: cpu}\n',
        encoding='utf-8',
    )
    return study


def test_recipe_registers_in_every_run_and_result_records_provenance(tmp_path: Path):
    """study.yaml recipe → prepare calls register per Run → result carries environment and provenance digest."""
    study = _study(tmp_path)
    out = run_study(study)
    assert len(out['results']) == 2
    digest = load_provenance(study)['digest']
    for path in out['results']:
        result = load_result(path)
        assert result['status'] == 'succeeded'
        assert result['provenance'] == digest
        assert result['environment']['python']
        mark = path.parent / 'assets' / 'recipe_mark.txt'
        run_id, seed = mark.read_text(encoding='utf-8').split()
        assert run_id == path.parent.name
        assert seed in ('0', '1')


def test_make_writes_provenance_with_sources_declarations_and_plan(tmp_path: Path):
    study = _study(tmp_path)
    body = load_provenance(expand_study(study)['study_dir'])
    files = body['files']
    assert 'rpipe/flow/cli.py' in files
    for name in ('study.yaml', 'experiment_config.yaml', 'recipe.py', 'notes/a.txt'):
        assert name in files
    assert 'index.json' in body['plan']
    assert sum(name.endswith('config.yaml') for name in body['plan']) == 2
    assert {'python', 'torch', 'cuda', 'gpu'} <= set(body['environment'])
    assert provenance_changes(study, load_study_yaml(study)) == []


@pytest.mark.parametrize('recipe,error', [
    ('', FileNotFoundError),
    ('x = 1\n', AttributeError),
])
def test_make_rejects_missing_or_invalid_recipe(tmp_path: Path, recipe: str, error: type):
    study = _study(tmp_path)
    if recipe:
        (study / 'recipe.py').write_text(recipe, encoding='utf-8')
    else:
        (study / 'recipe.py').unlink()
    with pytest.raises(error):
        expand_study(study)


def test_freeze_blocks_launch_and_run_one_after_source_change(tmp_path: Path, capsys):
    """freeze: true + edited recipe after make → launch gate and run-one both refuse."""
    study = _study(tmp_path, freeze=True)
    out = expand_study(study)
    declared = load_study_yaml(study)
    check_frozen(study, declared)
    (study / 'notes' / 'a.txt').write_text('changed\n', encoding='utf-8')
    assert provenance_changes(study, declared) == ['notes/a.txt']
    with pytest.raises(ProvenanceChangedError, match='notes/a.txt'):
        check_frozen(study, declared)
    assert cli._provenance_gate(study.resolve()) is False
    assert 'provenance changed' in capsys.readouterr().err
    run_dir = out['configs'][0].parent
    with pytest.raises(ProvenanceChangedError):
        launch_one(study.resolve(), run_dir.name)
    assert load_result(run_dir / 'result.json')['status'] == 'failed'


def test_without_freeze_change_only_warns(tmp_path: Path, capsys):
    study = _study(tmp_path)
    expand_study(study)
    (study / 'notes' / 'a.txt').write_text('changed\n', encoding='utf-8')
    assert cli._provenance_gate(study.resolve()) is True
    assert 'provenance: 1 changed' in capsys.readouterr().out


@pytest.mark.parametrize('form', ['bundle', 'pieces', 'payload'])
@pytest.mark.parametrize('relocated', [False, True], ids=['legacy-helper-state', 'relocated-new-helper-state'])
def test_recipe_checkpoint_state_keeps_legacy_and_relocated_imports(tmp_path, monkeypatch, relocated, form):
    import importlib.util
    import sys
    from rpipe.flow.context import FlowContext
    from rpipe.flow.prepare.run import run as prepare_run
    from rpipe.structure.artifact import artifact_layout, load_config
    from rpipe.structure.system.config import SystemConfig
    from rpipe.structure.system.factory import SystemFactory

    recipe = (
        'from helper import State\n'
        'from rpipe.structure.model.factory import ModelRegistry, _build_linear\n'
        'def register(ctx):\n'
        '    def build(model_config, assets_dir, data_meta=None):\n'
        '        model = _build_linear(model_config, assets_dir, data_meta)\n'
        '        model._recipe_state_type = State\n'
        '        return model\n'
        '    ModelRegistry.register("linear", "recipe_src", build)\n'
    )

    def create(parent, label):
        parent.mkdir()
        study = _study(parent, recipe=recipe)
        (study / 'helper.py').write_text(
            'from dataclasses import dataclass\n@dataclass\nclass State:\n'
            f'    value: int\n    label = {label!r}\n', encoding='utf-8')
        (study / 'torch.py').write_text('MARK = "unused local source"\n', encoding='utf-8')
        declaration = study / 'study.yaml'
        declaration.write_text(declaration.read_text(encoding='utf-8').replace(
            'notes/*.txt', 'notes/*.txt, helper.py'), encoding='utf-8')
        return study

    study = create(tmp_path / 'original', 'original')
    spec = importlib.util.spec_from_file_location('helper', study / 'helper.py')
    legacy_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, 'helper', legacy_module)
    spec.loader.exec_module(legacy_module)
    legacy = SystemFactory.build(SystemConfig.from_mapping({'device': 'cpu'}),
                                 study / 'runs' / 'legacy' / 'assets')
    import torch
    bundle = legacy.save_checkpoint({'model': {'_extra_state': legacy_module.State(7),
                                                'tensor': torch.tensor([1, 2])},
                                     'metadata': legacy_module.State(9)}, 'legacy')
    if form != 'bundle':
        import shutil
        folder = legacy.checkpoint_dir() / ('legacy-' + form)
        if form == 'pieces':
            shutil.copytree(bundle.with_suffix(''), folder)
        else:
            folder.mkdir()
            shutil.copy2(bundle, folder / 'payload.pt')
        bundle = folder
    monkeypatch.delitem(sys.modules, 'helper')
    input_root = legacy.checkpoint_dir()
    before = {path.relative_to(input_root).as_posix(): path.read_bytes()
              for path in input_root.rglob('*') if path.is_file()}

    def prepared(study):
        out = expand_study(study)
        config = out['configs'][0]
        ctx = FlowContext(study_dir=study, layout=artifact_layout(study, config.parent.name),
                          config=load_config(config))
        prepare_run(ctx)
        return ctx

    ctx = prepared(study)
    loaded = ctx.state['system'].load_checkpoint(str(bundle))
    assert loaded['model']['_extra_state'].value == 7
    assert loaded['metadata'].value == 9
    assert torch.equal(loaded['model']['tensor'], torch.tensor([1, 2]))
    if form == 'bundle':
        from rpipe.structure.artifact.readout.compare import _checkpoint
        displayed = _checkpoint(study / 'runs' / 'legacy', 'legacy')
        assert type(displayed['model']['_extra_state']) is ctx.state['model']._recipe_state_type
    assert type(loaded['model']['_extra_state']) is ctx.state['model']._recipe_state_type
    if relocated:
        bundle = ctx.state['system'].save_checkpoint(loaded, 'with-recipe-state')
        target = create(tmp_path / 'relocated', 'relocated')
        target_ctx = prepared(target)
        restored = target_ctx.state['system'].load_checkpoint(str(bundle))
        assert restored['model']['_extra_state'].value == 7
        assert type(restored['model']['_extra_state']) is target_ctx.state['model']._recipe_state_type
        assert restored['model']['_extra_state'].label == 'relocated'
        import shutil
        from rpipe.structure.artifact.readout.compare import _checkpoint
        shutil.copy2(bundle, target_ctx.state['system'].checkpoint_dir() / 'copied.pt')
        copied = target_ctx.state['system'].checkpoint_dir() / 'copied.pt'
        copied_before = copied.read_bytes()
        displayed = _checkpoint(target_ctx.layout.root, 'copied')
        assert type(displayed['model']['_extra_state']) is target_ctx.state['model']._recipe_state_type
        assert copied.read_bytes() == copied_before
    assert {path.relative_to(input_root).as_posix(): path.read_bytes()
            for path in input_root.rglob('*') if path.is_file()} == before
