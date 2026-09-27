import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.content,
    pytest.mark.p1,
    pytest.mark.structure_layer,
    pytest.mark.module_make,
]

from pathlib import Path

from rpipe.structure.make import (
    expand_patches,
    filter_jobs_by_mode,
    gpu_ids,
    load_launch_plan,
    load_study_yaml,
    plan_jobs,
    render_bash,
    run_succeeded,
    write_launch_scripts,
)
from rpipe.structure.artifact import artifact_layout


def _touch_config(study: Path, run_id: str, *, device: str = 'cuda') -> Path:
    layout = artifact_layout(study, run_id)
    layout.config_path.write_text(
        'id: {}\nsystem:\n  device: {}\n'.format(run_id, device),
        encoding='utf-8',
    )
    return layout.config_path


def _write_succeeded(study: Path, run_id: str) -> None:
    layout = artifact_layout(study, run_id)
    layout.result_path.write_text(
        '{"status": "succeeded", "control": {}, "metrics": {}, "paths": {}}\n',
        encoding='utf-8',
    )


def test_expand_patches_train_size_baseline():
    study = {
        'study': 'demo',
        'experiment': {'name': 'mnist_linear'},
        'fixed': {'seed': 0, 'algorithm': {'mode': 'train', 'lr': 0.1}},
        'axes': {'data.config.train_size': [500, 2000]},
        'tags': [{'when': {'data.config.train_size': 500}, 'tags': ['baseline']}],
        'run_description': 'size={train_size}',
    }
    patches = expand_patches(study)
    assert len(patches) == 2
    assert patches[0]['tags'] == ['baseline']
    assert patches[0]['data']['config']['train_size'] == 500
    assert 'tags' not in patches[1]
    assert patches[0]['description'] == 'size=500'


def test_plan_jobs_trains_before_eval(tmp_path: Path):
    study = tmp_path / 'study'
    eval_path = artifact_layout(study, 'e1').config_path
    eval_path.write_text('algorithm:\n  mode: eval\n', encoding='utf-8')
    train_path = artifact_layout(study, 't1').config_path
    train_path.write_text('algorithm:\n  mode: train\n', encoding='utf-8')
    jobs = plan_jobs(study, [eval_path, train_path], init_gpu=0, num_gpus=1)
    assert [j['run_id'] for j in jobs] == ['t1', 'e1']
    assert [j['mode'] for j in jobs] == ['train', 'eval']


def test_render_bash_waits_after_train_wave(tmp_path: Path):
    study = tmp_path / 'study'
    (tmp_path / 'pyproject.toml').write_text('[project]\nname = "rpipe"\n', encoding='utf-8')
    jobs = [
        {'run_id': 't0', 'gpu': '0', 'mode': 'train'},
        {'run_id': 't1', 'gpu': '0', 'mode': 'train'},
        {'run_id': 'e0', 'gpu': '0', 'mode': 'eval'},
        {'run_id': 'e1', 'gpu': '0', 'mode': 'eval'},
    ]
    text = render_bash(
        study,
        jobs,
        python_exe='/opt/python',
        round_size=4,
        split_round=65535,
    )[0]
    train_pos = text.rfind('"t1"')
    eval_pos = text.find('"e0"')
    wait_between = text.find('\nwait\n', train_pos, eval_pos)
    assert train_pos != -1 and eval_pos != -1
    assert wait_between != -1
    assert 'rpipe process' in text
    assert gpu_ids(2, 3) == ['2', '3', '4']
    assert gpu_ids(0, 0) == []


def test_console_new_uses_windows_flag_only_on_nt():
    from rpipe.structure.make.schedule import job_popen_kwargs, resolve_console

    assert resolve_console('shared') == 'shared'
    assert resolve_console('new') == 'new'
    shared = job_popen_kwargs('shared')
    assert 'creationflags' not in shared
    flags = job_popen_kwargs('new')
    import os

    if os.name == 'nt':
        assert 'creationflags' in flags
    else:
        assert flags == {}


def test_plan_jobs_skips_succeeded_and_assigns_gpu(tmp_path: Path):
    study = tmp_path / 'study'
    done = _touch_config(study, 'done')
    _write_succeeded(study, 'done')
    pending = _touch_config(study, 'pend')
    jobs = plan_jobs(study, [done, pending], init_gpu=1, num_gpus=2)
    assert [j['run_id'] for j in jobs] == ['pend']
    assert jobs[0]['gpu'] == '1'
    assert run_succeeded(study, 'done')
    assert not run_succeeded(study, 'pend')


def test_plan_jobs_does_not_assign_gpu_to_cpu_run(tmp_path: Path):
    study = tmp_path / 'study'
    config = _touch_config(study, 'cpu-run', device='cpu')
    jobs = plan_jobs(study, [config], init_gpu=0, num_gpus=1)
    assert jobs == [
        {
            'run_id': 'cpu-run',
            'config': config.as_posix(),
            'mode': 'train',
            'device': 'cpu',
        }
    ]


def test_render_bash_wait_every_round(tmp_path: Path):
    study = tmp_path / 'study'
    (tmp_path / 'pyproject.toml').write_text('[project]\nname = "rpipe"\n', encoding='utf-8')
    jobs = [{'run_id': f'r{i}', 'gpu': str(i % 2)} for i in range(4)]
    chunks = render_bash(
        study,
        jobs,
        python_exe='/opt/python',
        round_size=2,
        split_round=65535,
    )
    assert len(chunks) == 1
    text = chunks[0]
    assert text.count('\nwait\n') == 2
    assert text.count('run-one') == 4
    assert 'CUDA_VISIBLE_DEVICES="0"' in text
    assert 'CUDA_VISIBLE_DEVICES="1"' in text
    assert 'KMP_DUPLICATE_LIB_OK' not in text


def test_write_launch_scripts_split_round(tmp_path: Path):
    (tmp_path / 'pyproject.toml').write_text('[project]\nname = "rpipe"\n', encoding='utf-8')
    study = tmp_path / 'study'
    jobs = [{'run_id': f'r{i}', 'gpu': '0'} for i in range(4)]
    written = write_launch_scripts(
        study,
        jobs,
        round_size=2,
        split_round=1,
        python_exe='/opt/python',
        init_gpu=0,
        num_gpus=1,
    )
    assert written['n_jobs'] == 4
    assert len(written['bash']) == 2
    assert written['jobs_json'].is_file()
    assert written['ps1'].is_file()
    ps1 = written['ps1'].read_text(encoding='utf-8')
    assert 'rpipe launch' in ps1
    assert 'KMP_DUPLICATE_LIB_OK' not in ps1


def test_launch_job_env_only_sets_cuda_for_gpu(monkeypatch):
    from rpipe.structure.make.schedule import launch_job_env

    monkeypatch.delenv('KMP_DUPLICATE_LIB_OK', raising=False)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    cpu = launch_job_env({'run_id': 'cpu', 'device': 'cpu'})
    gpu = launch_job_env({'run_id': 'gpu', 'device': 'cuda', 'gpu': '2'})
    assert 'KMP_DUPLICATE_LIB_OK' not in cpu
    assert 'CUDA_VISIBLE_DEVICES' not in cpu
    assert gpu['CUDA_VISIBLE_DEVICES'] == '2'


def test_load_launch_plan_reuses_jobs_and_drops_succeeded(tmp_path: Path):
    (tmp_path / 'pyproject.toml').write_text('[project]\nname = "rpipe"\n', encoding='utf-8')
    study = tmp_path / 'study'
    jobs = [
        {'run_id': 't0', 'gpu': '0', 'mode': 'train'},
        {'run_id': 't1', 'gpu': '0', 'mode': 'train'},
        {'run_id': 'e0', 'gpu': '0', 'mode': 'eval'},
    ]
    batches = [jobs[:2], jobs[2:]]
    write_launch_scripts(
        study,
        jobs,
        round_size=2,
        python_exe='/opt/python',
        init_gpu=0,
        num_gpus=1,
        batches=batches,
    )
    _write_succeeded(study, 't0')
    plan = load_launch_plan(study, init_gpu=0, num_gpus=1, round_size=0)
    assert plan is not None
    assert [j['run_id'] for j in plan['job_list']] == ['t1', 'e0']
    assert [j['run_id'] for j in plan['batches'][0]] == ['t1']
    assert [j['run_id'] for j in plan['batches'][1]] == ['e0']
    assert load_launch_plan(study, init_gpu=0, num_gpus=2, round_size=0) is None
    assert load_launch_plan(study, init_gpu=0, num_gpus=1, round_size=4) is None


def test_filter_jobs_by_mode_keeps_eval_only():
    jobs = [
        {'run_id': 't0', 'mode': 'train'},
        {'run_id': 'e0', 'mode': 'eval'},
        {'run_id': 't1'},
    ]
    assert [j['run_id'] for j in filter_jobs_by_mode(jobs, ['eval'])] == ['e0']
    assert [j['run_id'] for j in filter_jobs_by_mode(jobs, None)] == ['t0', 'e0', 't1']


def test_load_launch_plan_include_done_keeps_succeeded(tmp_path: Path):
    (tmp_path / 'pyproject.toml').write_text('[project]\nname = "rpipe"\n', encoding='utf-8')
    study = tmp_path / 'study'
    jobs = [
        {'run_id': 't0', 'gpu': '0', 'mode': 'train'},
        {'run_id': 'e0', 'gpu': '0', 'mode': 'eval'},
    ]
    write_launch_scripts(
        study,
        jobs,
        round_size=2,
        python_exe='/opt/python',
        init_gpu=0,
        num_gpus=1,
        batches=[jobs[:1], jobs[1:]],
    )
    _write_succeeded(study, 't0')
    _write_succeeded(study, 'e0')
    skipped = load_launch_plan(study, init_gpu=0, num_gpus=1, round_size=0)
    assert skipped is not None
    assert skipped['job_list'] == []
    again = load_launch_plan(study, init_gpu=0, num_gpus=1, round_size=0, include_done=True)
    assert [j['run_id'] for j in again['job_list']] == ['t0', 'e0']
    assert [j['run_id'] for j in again['batches'][1]] == ['e0']


def test_launch_mode_eval_does_not_rewrite_jobs_json(tmp_path: Path, monkeypatch):
    from rpipe.flow import cli

    (tmp_path / 'pyproject.toml').write_text('[project]\nname = "rpipe"\n', encoding='utf-8')
    study = tmp_path / 'study'
    jobs = [
        {'run_id': 't0', 'gpu': '0', 'mode': 'train'},
        {'run_id': 'e0', 'gpu': '0', 'mode': 'eval'},
    ]
    write_launch_scripts(
        study,
        jobs,
        round_size=2,
        python_exe='/opt/python',
        init_gpu=0,
        num_gpus=1,
        batches=[jobs[:1], jobs[1:]],
    )
    _write_succeeded(study, 't0')
    launched: list[str] = []

    def fake_launch(_study, job_list, **_kwargs):
        launched.extend(str(job['run_id']) for job in job_list)
        return [0]

    monkeypatch.setattr(cli, 'launch_jobs', fake_launch)
    monkeypatch.setattr(cli, 'process_study', lambda *_args, **_kwargs: {'complete': False})
    monkeypatch.setattr(cli, 'process_path', lambda *_args, **_kwargs: study / 'process.json')
    assert cli.main(['launch', str(study), '--mode', 'eval', '--console', 'shared']) == 1
    assert launched == ['e0']
    stored = (study / 'scripts' / 'jobs.json').read_text(encoding='utf-8')
    assert '"t0"' in stored
    assert '"e0"' in stored


def test_cpu_make_skips_gpu_probe(tmp_path: Path, monkeypatch, capsys):
    from rpipe.flow import cli

    study = tmp_path / 'cpu-study'
    study.mkdir()
    (study / 'study.yaml').write_text(
        'study: cpu-study\norigin: domestic\naxes: {}\nseeds: [0]\n',
        encoding='utf-8',
    )
    (study / 'experiment_config.yaml').write_text(
        'experiment: demo\n'
        'data: {name: Toy, source: stub}\n'
        'model: {name: linear}\n'
        'algorithm: {mode: train, num_steps: 1}\n'
        'system: {device: cpu}\n',
        encoding='utf-8',
    )

    def fail_probe(*_args, **_kwargs):
        raise AssertionError('CPU make must not probe GPUs')

    monkeypatch.setattr(cli, 'probe_gpus', fail_probe)
    assert cli.main(['make', str(study)]) == 0
    out = capsys.readouterr().out
    assert 'make: origin domestic model https://hf-mirror.com' in out
    assert 'make: expand' in out
    assert 'make: pack' in out
    assert not (study / 'activity.json').is_file()
    jobs = (study / 'scripts' / 'jobs.json').read_text(encoding='utf-8')
    assert '"device": "cpu"' in jobs
    assert '"gpu"' not in jobs


def test_expand_rejects_origin_under_data():
    with pytest.raises(ValueError, match='Study'):
        expand_patches(
            {
                'origin': 'domestic',
                'fixed': {'data': {'name': 'CIFAR10', 'origin': 'foreign'}},
                'axes': {},
                'seeds': [0],
            }
        )
    repo = Path(__file__).resolve().parents[4]
    study = load_study_yaml(repo / 'studies' / 'cifar_grid')
    patches = expand_patches(study)
    assert len(patches) == 8
    pairs = [
        (p['model']['name'], p['algorithm']['mode'], p['seed'], p['data']['config']['train_size'])
        for p in patches
    ]
    assert pairs[0] == ('linear', 'train', 0, 1024)
    assert patches[0]['origin'] == 'domestic'
    assert 'origin' not in patches[0]['data']
    assert pairs[1] == ('linear', 'eval', 0, 1024)
    assert {name for name, _, _, _ in pairs} == {'linear', 'mlp', 'cnn', 'resnet18'}
    assert sum(1 for p in patches if p.get('tags') == ['baseline']) == 1
    assert patches[0]['description'] == 'model=linear mode=train seed=0'

