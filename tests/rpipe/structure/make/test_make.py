from pathlib import Path

from rpipe.structure.make import expand_patches, gpu_ids, plan_jobs, render_bash, run_succeeded, write_launch_scripts
from rpipe.structure.artifact import artifact_layout


def _touch_config(study: Path, run_id: str) -> Path:
    layout = artifact_layout(study, run_id)
    layout.config_path.write_text('id: {}\n'.format(run_id), encoding='utf-8')
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

    assert gpu_ids(2, 3) == ['2', '3', '4']
    assert gpu_ids(0, 0) == []


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
    assert 'rpipe launch' in written['ps1'].read_text(encoding='utf-8')
