"""CPU-only completed-group audit; never launches or resumes a formal Run.

Outputs are reporting artifacts, outside the frozen training sources. A pending
group is recorded as incomplete, and no active checkpoint is opened. Optional
CPU replay requires all four *existing* formal eval results to have succeeded.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import re
import subprocess
import sys

# Required import order on this Windows host; no CUDA calls are made.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[2]
STUDY = REPO / 'studies/main_historical'
REF = '4ccb28d0496110253e9f8e3f3df658853f07996b'
BASELINE = '8bccbac321d4c3ac1ea9892a5e774c114e0298c6'
STEPS = list(range(200, 80001, 200))
torch.set_num_threads(2)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b, atol=1e-9):
    return (type(a) in (int, float) and type(b) in (int, float)
            and math.isfinite(a) and math.isfinite(b) and abs(a-b) <= atol)


def git_blob(name):
    body = subprocess.check_output(['git', 'show', f'{REF}:{name}'], cwd=REPO)
    oid = subprocess.check_output(['git', 'rev-parse', f'{REF}:{name}'], cwd=REPO, text=True).strip()
    return body, {'commit': REF, 'path': name, 'git_blob': oid,
                  'sha256': hashlib.sha256(body).hexdigest()}


def source_audit():
    m = read_json(STUDY/'docs/SOURCE_MANIFEST.json')
    output = {}
    for key in ('files', 'plan_files'):
        different = []
        for name, expected in m[key].items():
            path = REPO/name
            actual = sha(path) if path.is_file() else None
            if actual != expected:
                different.append({'path': name, 'expected_sha256': expected, 'actual_sha256': actual})
        output[key] = {'files_checked': len(m[key]), 'differences': different}
    raw = read_json(STUDY/'docs/DATA_MANIFEST.json')
    different = []
    for entry in raw['files']:
        path = STUDY/'shared/data'/entry['path']
        actual = sha(path) if path.is_file() else None
        if actual != entry['sha256']:
            different.append({'path': entry['path'], 'expected_sha256': entry['sha256'], 'actual_sha256': actual})
    output['raw_data'] = {'files_checked': len(raw['files']), 'differences': different}
    output['passed'] = all(not output[key]['differences'] for key in ('files', 'plan_files', 'raw_data'))
    return output


def scalar_rows(run_dir):
    path = run_dir/'assets/tracker/scalars.jsonl'
    rows, unfinished, errors = [], False, []
    if not path.exists():
        return rows, unfinished, ['missing scalars.jsonl']
    lines = path.read_bytes().splitlines(keepends=True)
    for number, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError('not an object')
            rows.append(value)
        except (ValueError, UnicodeDecodeError) as exc:
            if number == len(lines) and not line.endswith(b'\n'):
                unfinished = True
            else:
                errors.append(f'scalar line {number}: {exc}')
    return rows, unfinished, errors


def history(state, split, metric):
    return state.get('splits', {}).get(split, {}).get(metric, {}).get('history', [])


def measurements(rows, split, metric):
    return [r for r in rows if r.get('split') == split and r.get('name') == metric]


def status(entry):
    run_dir = STUDY/'runs'/entry['run_dir']
    result = run_dir/'result.json'
    if result.exists():
        return read_json(result).get('status', 'unknown')
    log = run_dir/'assets/logs/run.log'
    return 'started' if log.exists() and '[flow] start' in log.read_text(encoding='utf-8') else 'pending'


def whole_snapshot(index):
    counts = {mode: Counter() for mode in ('train', 'eval')}
    finished = {mode: [] for mode in ('train', 'eval')}
    for exp in index['experiments']:
        mode = exp['factors']['algorithm.mode']
        states = [status(e) for e in exp['runs']]
        counts[mode].update(states)
        if len(states) == 4 and states == ['succeeded']*4:
            finished[mode].append((exp['factors']['data.name'], exp['factors']['model.name']))
    return {'counts': {k: dict(v) for k, v in counts.items()},
            'completed_training_groups': len(finished['train']),
            'completed_train_and_eval_groups': len(set(finished['train']) & set(finished['eval']))}


def optimizer_audit(payload, data, model):
    state = payload.get('model', {})
    opt = payload.get('optimizer', {})
    groups = opt.get('param_groups', [])
    tensors = opt.get('state', {})
    ids = [pid for group in groups for pid in group.get('params', [])]
    params = list(state.values())
    expected = {'momentum': .9, 'weight_decay': .0005, 'nesterov': True,
                'dampening': 0, 'maximize': False, 'initial_lr': .01}
    # Historical Linear has only its two parameters, so every tensor maps to a
    # momentum buffer. Other models can contain buffers; avoid claiming mapping.
    linear_shapes = ([10, 784], [10]) if data == 'MNIST' else ([10, 3072], [10])
    fields = {
        'single_sgd_parameter_group': len(groups) == 1,
        'historical_sgd_config_exact': len(groups) == 1 and all(groups[0].get(k) == v for k, v in expected.items()),
        'model_tensors_finite': bool(state) and all(isinstance(t, torch.Tensor) and (not t.is_floating_point() or bool(torch.isfinite(t).all())) for t in params),
        'momentum_for_each_optimizer_parameter': bool(ids) and set(ids) == set(tensors) and all(set(tensors[pid]) == {'momentum_buffer'} for pid in ids),
        'momentum_buffers_finite': bool(tensors) and all(isinstance(s.get('momentum_buffer'), torch.Tensor) and bool(torch.isfinite(s['momentum_buffer']).all()) for s in tensors.values()),
    }
    if model == 'linear':
        fields['linear_model_parameter_shapes'] = list(state) == ['linear.weight', 'linear.bias'] and [list(t.shape) for t in params] == list(linear_shapes)
        fields['linear_momentum_shapes_match_parameters'] = len(ids) == len(params) == 2 and all(tensors[pid]['momentum_buffer'].shape == tensor.shape for pid, tensor in zip(ids, params))
    scheduler = payload.get('scheduler', {})
    step = payload.get('step')
    expected_lr = .01 * (1 + math.cos(math.pi*step/80000))/2 if type(step) is int else None
    fields['cosine_horizon_and_step'] = (scheduler.get('T_max') == 80000 and scheduler.get('eta_min') == 0
        and scheduler.get('base_lrs') == [.01] and scheduler.get('last_epoch') == step
        and scheduler.get('_step_count') == step+1)
    fields['cosine_lr_matches_step'] = len(groups) == 1 and close(groups[0].get('lr'), expected_lr, 1e-10) and len(scheduler.get('_last_lr', [])) == 1 and close(scheduler['_last_lr'][0], expected_lr, 1e-10)
    return {'checks': fields, 'passed': all(fields.values()), 'scheduler': scheduler,
            'optimizer_groups': groups, 'momentum_parameter_count': len(tensors),
            'expected_cosine_lr': expected_lr,
            'boundary': 'Finite/configuration/shape checks verify saved SGD state; they do not reconstruct all 80000 updates or unrecorded RNG.'}


def train_audit(entry, data, model):
    d = STUDY/'runs'/entry['run_dir']
    result = read_json(d/'result.json')
    config = yaml.safe_load((d/'config.yaml').read_text(encoding='utf-8'))
    log = (d/'assets/logs/run.log').read_text(encoding='utf-8')
    rows, unfinished, parse_errors = scalar_rows(d)
    tracker = read_json(d/'assets/tracker/tracker_state.json')
    latest_path, best_path = [d/'assets/checkpoints'/f'{s}.pt' for s in ('latest', 'best')]
    latest = torch.load(latest_path, map_location='cpu', weights_only=False)
    best = torch.load(best_path, map_location='cpu', weights_only=False)
    observed = {(split, metric): measurements(rows, split, metric) for split in ('train', 'test') for metric in ('Loss', 'Accuracy')}
    acc = [r['mean'] for r in observed['test', 'Accuracy']]
    loss = [r['mean'] for r in observed['test', 'Loss']]
    steps = [r.get('optimizer_step') for r in observed['test', 'Accuracy']]
    starts = [r for r in rows if r.get('event') == 'start']
    errors = [line for line in log.splitlines() if re.search(r'\bERROR\b|Traceback|\[retry\]|\[error\]', line)]
    flow = [line for line in log.splitlines() if '[flow] start' in line]
    ctrl = result.get('control', {})
    ds = result.get('structure', {}).get('data', {})
    alg = ctrl.get('algorithm', {})
    max_acc = max(acc) if acc else None
    max_steps = [x for x, y in zip(steps, acc) if close(y, max_acc)]
    best_step = best.get('step')
    prefix = best_step//200 if type(best_step) is int and best_step%200 == 0 else -1
    sample_counts = [t.get('step', -1)-a.get('step', 0) for a,t in zip(observed['train', 'Accuracy'],observed['test', 'Accuracy'])]
    checks = {
        'result_succeeded': result.get('status') == 'succeeded',
        'seed_and_factors_match_index': ctrl.get('seed') == entry['seed'] and ctrl.get('data', {}).get('name') == data and ctrl.get('model', {}).get('name') == model and alg.get('mode') == 'train',
        'config_control_matches_result': config == ctrl,
        'unique_flow_start': len(flow) == 1,
        'unique_fresh_tracker_start': len(starts) == 1 and starts[0].get('keep_until') == 0,
        'no_resume_events': '[resume]' not in log,
        'no_error_retry_records': not errors,
        'all_scalar_lines_committed_and_valid': not unfinished and not parse_errors,
        'full_eval_configuration': alg.get('eval_num_steps') == -1 and alg.get('eval_period') == 200 and ds.get('train_size') == (60000 if data == 'MNIST' else 50000) and ds.get('test_size') == 10000 and ds.get('batch_size') == ds.get('test_batch_size') == 250,
        'historical_source_metadata': ctrl.get('historical_ref') == ds.get('historical_commit') == REF and ds.get('sampler_steps') == 80000,
        'all_test_samples_10000_reconstructed': sample_counts == [40]*400 and all(n*250 == 10000 for n in sample_counts),
        'latest_global_step_80000': latest.get('step') == 80000 and latest.get('tracker', {}).get('progress', {}).get('step') == 80000,
        'latest_counter_96000': latest.get('tracker', {}).get('step') == tracker.get('step') == 96000,
        'latest_checkpoint_tracker_equals_saved_tracker': latest.get('tracker') == tracker,
        'best_metric_accuracy_global_max': best.get('best_metric') == 'Accuracy' and close(best.get('best_value'), max_acc) and close(best.get('best_accuracy'), max_acc) and close(best.get('test_accuracy'), max_acc),
        'best_step_is_own_global_max': best_step in max_steps,
        'best_counter_and_progress_match_step': prefix > 0 and best.get('tracker', {}).get('step') == prefix*240 and best.get('tracker', {}).get('progress', {}).get('step') == best_step,
        'result_summary_matches_final_and_best': bool(acc) and close(result.get('metrics', {}).get('accuracy'), acc[-1]) and close(result.get('metrics', {}).get('best_accuracy'), max_acc) and close(result.get('metrics', {}).get('test_loss'), loss[-1]),
    }
    for (split, metric), sequence in observed.items():
        tag = f'{split}_{metric.lower()}'
        ys = [r.get('mean') for r in sequence]
        expected_counts = [(j+1)*240-(40 if split == 'train' else 0) for j in range(400)]
        checks[f'{tag}_400_optimizer_steps'] = [r.get('optimizer_step') for r in sequence] == STEPS
        checks[f'{tag}_internal_counters'] = [r.get('step') for r in sequence] == expected_counts
        checks[f'{tag}_tracker_history_exact'] = history(tracker, split, metric) == ys
        checks[f'{tag}_latest_history_exact'] = history(latest.get('tracker', {}), split, metric) == ys
        checks[f'{tag}_best_prefix_exact'] = prefix > 0 and history(best.get('tracker', {}), split, metric) == ys[:prefix]
    optimizers = {stem: optimizer_audit(payload, data, model) for stem, payload in [('latest', latest), ('best', best)]}
    checks['latest_optimizer_and_scheduler_integrity'] = optimizers['latest']['passed']
    checks['best_optimizer_and_scheduler_integrity'] = optimizers['best']['passed']
    checks['final_lr_zero'] = latest['scheduler'].get('_last_lr') == [0.] and [g.get('lr') for g in latest['optimizer'].get('param_groups', [])] == [0.]
    files = [d/'result.json',d/'config.yaml',d/'assets/logs/run.log',d/'assets/tracker/scalars.jsonl',d/'assets/tracker/tracker_state.json',latest_path,best_path]
    return {'id': entry['id'], 'seed': entry['seed'], 'status': result.get('status'),
        'checks': checks, 'audit_passed': all(checks.values()),
        'failed_checks': [k for k,v in checks.items() if not v], 'parse_errors': parse_errors,
        'flow_start_record': flow[0] if len(flow) == 1 else flow,
        'flow_start_count': len(flow), 'tracker_start_count': len(starts),
        'resume_event_count': log.count('[resume]'), 'error_retry_records': errors,
        'test_points': len(acc), 'optimizer_steps': steps, 'test_accuracy_pct': acc, 'test_loss': loss,
        'test_samples_reconstructed_per_point': [n*250 for n in sample_counts],
        'test_sample_count_method': 'Explicit train/test tracker counters differ by40 batches; fixed full test10000/batch250 metadata. Formal worker did not export per-sample byte observations.',
        'latest': {'global_step': latest.get('step'), 'tracker_counter': latest.get('tracker', {}).get('step'), **optimizers['latest']},
        'best': {'global_step': best_step, 'accuracy_pct': best.get('best_accuracy'), 'loss_at_best': loss[prefix-1] if 0 < prefix <= len(loss) else None,
                 'global_max_steps': max_steps, 'path': str(best_path), **optimizers['best']},
        'endpoint_accuracy_pct': acc[-1] if acc else None,
        'endpoint_correct_samples': round(acc[-1]*100) if acc else None,
        'elapsed_seconds': result.get('metrics', {}).get('elapsed_seconds'),
        'file_sha256': {str(p.relative_to(REPO)).replace('\\','/'): sha(p) for p in files}}


def curve_diagnostics(runs, data, model, reference):
    matrix = np.array([r['test_accuracy_pct'] for r in runs], dtype=np.float64)
    mean, std = matrix.mean(axis=0), matrix.std(axis=0, ddof=0)
    contract = reference['acceptance_contract']
    target = reference['datasets'][data]['models'][model]
    refs = {p['slot']: p['accuracy_pct_estimate'] for p in target['sparse_points']}
    anchors = [{'slot': slot, 'optimizer_step': STEPS[slot], 'observed_accuracy_pct': float(mean[slot]),
                'population_std_pp': float(std[slot]), 'reference_image_estimate_pct': refs[slot],
                'signed_error_pp': float(mean[slot]-refs[slot]), 'absolute_error_pp': abs(float(mean[slot]-refs[slot]))}
               for slot in contract['curve_comparison_slots']]
    errs = [a['absolute_error_pp'] for a in anchors]
    metrics = {
        'endpoint': (float(mean[-1]), target['endpoint_accuracy_pct_estimate'], contract['endpoint_abs_difference_pp'][data]),
        'late_50_mean': (float(mean[350:400].mean()), target['late_50_mean_accuracy_pct_estimate'], contract['late_50_mean_abs_difference_pp'][data]),
        'anchor_mae': (float(np.mean(errs)), 0., contract['anchor_mean_absolute_error_pp'][data]),
        'anchor_max': (float(np.max(errs)), 0., contract['anchor_max_absolute_error_pp'][data]),
        'late_50_temporal_std': (float(mean[350:400].std(ddof=0)), 0., contract['late_50_temporal_population_std_max_pp'][data]),
    }
    diagnostics = {k: {'observed': v, 'reference_image_estimate': ref if k in ('endpoint','late_50_mean') else None,
                      'observed_error_or_std_pp': abs(v-ref), 'threshold_pp': lim, 'inside_predeclared_limit': bool(abs(v-ref) <= lim)}
                   for k,(v,ref,lim) in metrics.items()}
    curve = [{'slot': j, 'optimizer_step': step, 'seed_count': 4, 'mean_accuracy_pct': float(mean[j]),
              'population_std_pp': float(std[j])} for j,step in enumerate(STEPS)]
    cross = {'available': False, 'boundary': 'COMPARISON is an independently timed partial snapshot; missing later points do not invalidate this completed-group audit.'}
    cp = STUDY/'docs/COMPARISON.json'
    if cp.exists():
        comparison = read_json(cp)
        group = next((x for x in comparison.get('experiments', []) if x['data'] == data and x['model'] == model), None)
        if group:
            common = [(j, step//200-1) for j,step in enumerate(group['optimizer_steps']) if step in STEPS and group['seed_count_by_step'][j] == 4]
            cross.update(available=True, recorded_at_utc=comparison.get('recorded_at_utc'), comparable_four_seed_points=len(common),
                         complete_400_points=group['optimizer_steps'] == STEPS and group['seed_count_by_step'] == [4]*400,
                         maximum_mean_difference_pp=max((abs(group['mean_accuracy_pct'][j]-mean[k]) for j,k in common),default=None),
                         maximum_population_std_difference_pp=max((abs(group['population_std_accuracy_pp'][j]-std[k]) for j,k in common),default=None),
                         comparison_sha256=sha(cp))
    return {'four_seed_same_step_curve': curve, 'anchors': anchors, 'single_group_diagnostics': diagnostics,
            'all_five_available_group_diagnostics_pass': all(v['inside_predeclared_limit'] for v in diagnostics.values()),
            'comparison_json_crosscheck': cross}


def original_linear_class():
    raw, source = git_blob('src/model/linear.py')
    parsed = ast.parse(raw.decode('utf-8'))
    cls = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == 'Linear')
    module = ast.Module(body=[cls], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {'torch': torch, 'nn': nn, 'math': math}
    exec(compile(module, f'{REF}:src/model/linear.py', 'exec'), namespace)
    return namespace['Linear'], source


def test_data(data):
    raw, source = git_blob('src/dataset/dataset.py')
    assignment = next(n for n in ast.parse(raw.decode('utf-8')).body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'data_stats' for t in n.targets))
    mean_values, std_values = ast.literal_eval(assignment.value)[data]
    base = STUDY/'shared/data'/data/'raw'
    if data == 'MNIST':
        import struct
        image_path, label_path = base/'t10k-images-idx3-ubyte', base/'t10k-labels-idx1-ubyte'
        im, lab = image_path.read_bytes(), label_path.read_bytes()
        magic, count, h, w = struct.unpack('>IIII', im[:16])
        lmagic, lcount = struct.unpack('>II',lab[:8])
        if (magic,count,h,w,lmagic,lcount) != (2051,10000,28,28,2049,10000):
            raise ValueError('unexpected MNIST IDX test header')
        pixels = np.frombuffer(im,dtype=np.uint8,offset=16).copy().reshape(10000,1,28,28)
        labels = np.frombuffer(lab,dtype=np.uint8,offset=8).copy().astype(np.int64)
        source_files = [image_path,label_path]
    else:
        test_path = base/'cifar-10-batches-py/test_batch'
        with test_path.open('rb') as handle:
            obj = pickle.load(handle,encoding='latin1')
        pixels = np.array(obj['data'],dtype=np.uint8).reshape(-1,3,32,32)
        labels = np.array(obj['labels'],dtype=np.int64)
        if pixels.shape != (10000,3,32,32) or labels.shape != (10000,):
            raise ValueError('unexpected CIFAR10 test_batch shape')
        source_files = [test_path]
    # Matches PIL ToTensor followed by in-place torchvision Normalize; no test augmentation.
    x = torch.from_numpy(pixels).to(dtype=torch.float32).div(255)
    mean = torch.tensor(mean_values,dtype=x.dtype).view(1,-1,1,1)
    std = torch.tensor(std_values,dtype=x.dtype).view(1,-1,1,1)
    x.sub_(mean).div_(std)
    y = torch.from_numpy(labels)
    return x,y,{'data':data,'samples':10000,'batch_size':250,'batches':40,
                'normalization_mean':list(mean_values),'normalization_std':list(std_values),
                'normalization_source':source,'normalization_operation':'float32 /255, subtract mean, divide std; no test augmentation',
                'raw_source_files': {str(p.relative_to(REPO)).replace('\\','/'):sha(p) for p in source_files},
                'normalized_pixels_sha256':hashlib.sha256(x.numpy().tobytes()).hexdigest(),
                'labels_sha256':hashlib.sha256(y.numpy().tobytes()).hexdigest()}


def cpu_replay(best, data, x, y, reference_accuracy, reference_loss):
    cls, source = original_linear_class()
    metric_raw, metric_source = git_blob('src/metric/metric.py')
    accuracy_function = next(n for n in ast.parse(metric_raw.decode('utf-8')).body if isinstance(n, ast.FunctionDef) and n.name == 'Accuracy')
    metric_module = ast.Module(body=[accuracy_function],type_ignores=[])
    ast.fix_missing_locations(metric_module)
    metric_namespace = {'torch':torch}
    exec(compile(metric_module,f'{REF}:src/metric/metric.py','exec'),metric_namespace)
    shape = [1,28,28] if data == 'MNIST' else [3,32,32]
    model = cls(shape,10).cpu()
    load = model.load_state_dict(best['model'],strict=True)
    exact = not load.missing_keys and not load.unexpected_keys and all(torch.equal(value,best['model'][name]) for name,value in model.state_dict().items())
    model.eval()
    correct, loss_sum, accuracy_sum = 0,0.,0.
    with torch.inference_mode():
        for start in range(0,10000,250):
            logits = model.f(x[start:start+250])
            labels = y[start:start+250]
            count = int((logits.argmax(1) == labels).sum())
            correct += count
            loss_sum += float(F.cross_entropy(logits,labels).item())*250
            # Preserve the actual archived float32 multiplication order; a
            # mathematically equal divide-then-multiply changes rounded values.
            accuracy_sum += metric_namespace['Accuracy'](logits,labels)*250
    loss, accuracy = loss_sum/10000,accuracy_sum/10000
    difference = abs(loss-reference_loss)
    checks = {'strict_weights_equal_checkpoint':exact,'all10000_samples':len(y)==10000,
              'correct_samples_equal_formal_eval':correct==round(reference_accuracy*100),
              'accuracy_equal_formal_eval':close(accuracy,reference_accuracy),
              'loss_difference_within_1e_6':difference<=1e-6}
    return {'device':'cpu','source':source,'metric_source':metric_source,'checks':checks,'passed':all(checks.values()),
            'correct_samples':correct,'samples':10000,'batches':40,'accuracy_pct':accuracy,
            'loss':loss,'formal_eval_loss':reference_loss,'loss_absolute_difference':difference,
            'loss_tolerance':1e-6,'formal_flow_run_launched':False,'gpu_operations':False}


def eval_audit(entry, train, data, model, replay_inputs=None):
    d = STUDY/'runs'/entry['run_dir']
    result = read_json(d/'result.json')
    tracker = read_json(d/'assets/tracker/tracker_state.json')
    log = (d/'assets/logs/run.log').read_text(encoding='utf-8')
    rows, unfinished, errors = scalar_rows(d)
    starts = [r for r in rows if r.get('event') == 'start']
    acc,loss = measurements(rows,'test','Accuracy'),measurements(rows,'test','Loss')
    paths = re.findall(r'\[split\] test.*?resume_path=(.+?)\s+step=(\d+)',log)
    resumes = re.findall(r'\[resume\].*target=(.+?)\s+epoch=.*?\s+step=(\d+)',log)
    best_path = Path(train['best']['path'])
    best = torch.load(best_path,map_location='cpu',weights_only=False)
    metrics = result.get('metrics',{})
    actual = acc[0]['mean'] if len(acc)==1 else None
    value_loss = loss[0]['mean'] if len(loss)==1 else None
    ds = result.get('structure',{}).get('data',{})
    ctrl = result.get('control',{})
    best_hash = train['file_sha256'][str(best_path.relative_to(REPO)).replace('\\','/')]
    checks = {
        'result_succeeded': result.get('status')=='succeeded',
        'seed_and_factors_match_index':ctrl.get('seed')==entry['seed'] and ctrl.get('data',{}).get('name')==data and ctrl.get('model',{}).get('name')==model and ctrl.get('algorithm',{}).get('mode')=='eval',
        'config_control_matches_result':yaml.safe_load((d/'config.yaml').read_text(encoding='utf-8'))==ctrl,
        'unique_flow_start':log.count('[flow] start')==1,
        'unique_flow_succeeded':log.count('[flow] succeeded')==1,
        'unique_fresh_tracker_start':len(starts)==1 and starts[0].get('keep_until')==0,
        'one_resume_at_own_best_step':len(resumes)==1 and int(resumes[0][1])==best.get('step'),
        'actual_resume_path_is_sibling_best':len(paths)==1 and Path(paths[0][0]).resolve()==best_path.resolve() and int(paths[0][1])==best.get('step'),
        'best_checkpoint_unchanged':sha(best_path)==best_hash,
        'all_scalar_lines_committed_and_valid':not unfinished and not errors,
        'single_accuracy_and_loss_observation':len(acc)==len(loss)==1,
        'accuracy_equals_own_global_best':close(actual,train['best']['accuracy_pct']),
        'loss_equals_own_best_history':close(value_loss,train['best']['loss_at_best']),
        'recorded_optimizer_step_is_best':len(acc)==len(loss)==1 and acc[0].get('optimizer_step')==loss[0].get('optimizer_step')==best.get('step'),
        'test10000_40batch_metadata':ds.get('test_size')==10000 and ds.get('test_batch_size')==250 and len(acc)==len(loss)==1 and acc[0].get('step')==loss[0].get('step')==tracker.get('step')==40,
        'full_eval_configuration':ctrl.get('algorithm',{}).get('eval_num_steps')==-1,
        'tracker_and_scalar_accuracy_equal':len(acc)==1 and history(tracker,'test','Accuracy')==[actual],
        'tracker_and_scalar_loss_equal':len(loss)==1 and history(tracker,'test','Loss')==[value_loss],
        'formal_result_matches_scalars':close(metrics.get('eval_accuracy'),actual) and close(metrics.get('test_accuracy'),actual) and close(metrics.get('accuracy'),actual) and close(metrics.get('test_loss'),value_loss),
        'no_error_retry_records':not re.search(r'\bERROR\b|Traceback|\[retry\]|\[error\]',log),
    }
    files = [d/'result.json',d/'config.yaml',d/'assets/logs/run.log',d/'assets/tracker/scalars.jsonl',d/'assets/tracker/tracker_state.json']
    output = {'id':entry['id'],'seed':entry['seed'],'train_id':train['id'],'status':result.get('status'),
              'checks':checks,'audit_passed':all(checks.values()),'failed_checks':[k for k,v in checks.items() if not v],
              'actual_resume_path':paths[0][0] if len(paths)==1 else None,'logged_initial_resume_target':resumes[0][0] if len(resumes)==1 else None,
              'optimizer_step':acc[0].get('optimizer_step') if len(acc)==1 else None,
              'accuracy_pct':actual,'loss':value_loss,'correct_samples':round(actual*100) if actual is not None else None,
              'best_checkpoint_sha256':best_hash,'file_sha256':{str(p.relative_to(REPO)).replace('\\','/'):sha(p) for p in files}}
    if replay_inputs is not None and all(checks.values()):
        x,y = replay_inputs
        output['cpu_full_test_replay'] = cpu_replay(best,data,x,y,actual,value_loss)
        output['audit_passed'] &= output['cpu_full_test_replay']['passed']
    return output


def eval_execution_audit(data, model, entries):
    stem = f'EARLY_EVAL_{data}_{model.upper()}_EXECUTION'
    path = STUDY/'docs'/f'{stem}.json'
    if not path.exists():
        return {'available':False,'boundary':'No group-specific execution record present; formal Run result/log checks remain separate.'}
    record = read_json(path)
    checks = {'four_expected_workers':len(record.get('runs',[]))==4 and [r['id'] for r in record['runs']]==[r['id'] for r in entries],
              'four_child_exit0_succeeded':len(record.get('runs',[]))==4 and all(r.get('exit_code')==0 and r.get('status')=='succeeded' for r in record['runs'])}
    recovery = record.get('reporting_recovery')
    provenance = {str(path.relative_to(REPO)).replace('\\','/'):sha(path)}
    if recovery:
        source_manifest = STUDY/'docs/SOURCE_MANIFEST.json'
        helper = STUDY/'docs'/f'{("CIFAR" if data=="CIFAR10" else data)}_{model.upper()}_EVAL_EXECUTION_HELPER.py'
        failed = STUDY/'docs'/f'EARLY_EVAL_{data}_{model.upper()}_FAILED_CONTROLLER.json'
        uncommitted = STUDY/'docs'/f'EARLY_EVAL_{data}_{model.upper()}_UNCOMMITTED_RECORD.json'
        checks.update(
            controller_failure_preserved=record.get('controller_status')=='failed' and record.get('controller_exit_code')==1,
            no_workers_restarted=recovery.get('no_workers_restarted') is True,
            actual_execution_helper_hash_matches=helper.exists() and sha(helper)==recovery.get('actual_execution_helper_sha256'),
            failed_controller_record_hash_matches=failed.exists() and sha(failed)==recovery.get('original_record_sha256'),
            uncommitted_record_hash_matches=uncommitted.exists() and sha(uncommitted)==recovery.get('uncommitted_record_sha256'),
            recovery_bound_to_frozen_source_manifest=sha(source_manifest)==recovery.get('source_manifest_sha256'),
        )
        for p in (helper,failed,uncommitted):
            if p.exists():provenance[str(p.relative_to(REPO)).replace('\\','/')]=sha(p)
    return {'available':True,'record_path':str(path),'status':record.get('status'),
            'controller_status':record.get('controller_status'),'controller_exit_code':record.get('controller_exit_code'),
            'checks':checks,'passed':all(checks.values()),'reporting_recovery':recovery,
            'worker_start_utc':record.get('started_at_utc'),'worker_end_utc':record.get('finished_at_utc'),
            'evidence_sha256':provenance}


def report_text(report):
    data,model = report['data'],report['model']
    lines = [f'# {data} / {model} 四 seed 独立 CPU 审查', '', '## 一、摘要', '',
             f"审查时间：{report['recorded_at_cst']}。固定 dev `{BASELINE}`，历史来源 `{REF}`。只读取正式 Run 的实际结果、完整 JSONL、tracker 和 CPU 重载的 canonical checkpoint；未启动、恢复或重启任何正式 Run，未使用 GPU。", '',
             f"训练四 seed complete={report['training_group_complete']}，完整性审查 passed={report['run_integrity_audit_passed']}；正式独立 eval complete={report['formal_eval_group_complete']}。整项历史 Study 的最终门未应用，`historical_reproduction_passed=null`。", '',
             f"全矩阵快照：`{json.dumps(report['whole_study_status_snapshot'],ensure_ascii=False)}`。", '',
             '## 二、训练终态与来源', '',
             '| **seed / Run** | **状态** | **test 点数** | **latest step** | **终点 Accuracy (%) / 正确样本** | **自身 global Accuracy-best step / (%)** | **完整性检查** |',
             '|---|---|---:|---:|---|---|---|']
    for r in report.get('runs',[]):
        lines.append(f"| {r['seed']} / {r['id']} | {r['status']} | {r['test_points']} | {r['latest']['global_step']} | {r['endpoint_accuracy_pct']:.8f} / {r['endpoint_correct_samples']} | {r['best']['global_step']} / {r['best']['accuracy_pct']:.8f} | {sum(r['checks'].values())}/{len(r['checks'])}，失败项 {r['failed_checks']} |")
    lines += ['', '全部四条需要正式 succeeded，各一次 Flow start、一次 fresh tracker start，无 resume/error/retry；400点显式 optimizer_step覆盖200–80000，train/test内部counter按200/40batch连续。每次test10000来自固定batch250、完整数据元信息与40batch差复算，正式worker未记录逐样本输入字节。latest global_step80000、tracker counter96000，所有train/test Loss/Accuracy历史与原始JSONL一致；best只取自身完整Accuracy曲线最大值，并核对对应历史前缀、scheduler和SGD momentum。保存的配置/shape/有限数检查不等同重新执行全部80000次更新。', '',
              f"来源审查：`{json.dumps(report['source_integrity'],ensure_ascii=False)}`。机器证据保存各result/config/log/scalars/tracker/latest/best的SHA-256，未读取分件镜像作为checkpoint权威。", '', '## 三、固定曲线门的单组合诊断', '']
    if model != 'linear':
        lines += ['本非线性组合的 SGD 检查覆盖保存配置、momentum 字段和有限数；未应用 Linear 参数形状与 momentum 的逐一映射检查，也未声称完成额外 CPU 推理。', '']
    if report.get('single_group_diagnostics'):
        ds = report['single_group_diagnostics']
        lines += [f"同一步四seed mean终点{ds['endpoint']['observed']:.8f}%，末50点mean{ds['late_50_mean']['observed']:.8f}%；按population std (ddof=0)聚合全部400点，未用best值替代终点。参考值为PNG估读，不是历史原始指标。", '',
                  '| **诊断** | **实测误差 / 时间 std (百分点)** | **预先门限 (百分点)** | **门内** |', '|---|---:|---:|---|']
        for name,value in ds.items():
            lines.append(f"| {name} | {value['observed_error_or_std_pp']:.8f} | ≤{value['threshold_pp']} | {value['inside_predeclared_limit']} |")
        lines += ['', '| **槽 / optimizer_step** | **mean Accuracy (%)** | **population std (百分点)** | **PNG (%)** | **有符号差 (百分点)** |', '|---|---:|---:|---:|---:|']
        for a in report['anchors']:
            lines.append(f"| {a['slot']} / {a['optimizer_step']} | {a['observed_accuracy_pct']:.8f} | {a['population_std_pp']:.8f} | {a['reference_image_estimate_pct']:.2f} | {a['signed_error_pp']:+.8f} |")
        lines += ['', '上述五项仅为该四seed完整组的诊断；如门外按实测记录，不调整门限。四模型排序及其余组合完整性尚待，不能声明整体goal完成。']
    else:
        lines.append('完整且一致的四seed训练尚不可用，未计算完整曲线门。')
    lines += ['', '## 四、正式独立 eval 与 CPU 回放', '']
    execution = report.get('formal_eval_execution_audit',{})
    if execution.get('available'):
        lines += [f"执行记录状态为 `{execution['status']}`，四条child.wait退出0及正式succeeded已独立核对；执行来源审查 passed={execution['passed']}。",'']
        if execution.get('reporting_recovery'):
            lines += ['原eval调度器在第四条child.wait已返回0后，保存执行JSON的 `os.replace` 遭遇WinError5，调度器实际exit1，`controller_status=failed`保留。主线程只恢复执行记录，未重跑worker；原failed-controller记录、原未提交记录字节和实际执行helper源码均独立归档并核对SHA-256。后续修改的临时helper SHA不混入本轮结果。这是记录写入失败，不能称为调度器exit0，也没有证据据此归因为GPU worker失败。', '',
                      '[恢复后的执行记录](EARLY_EVAL_CIFAR10_LINEAR_EXECUTION.json)、[原控制器记录](EARLY_EVAL_CIFAR10_LINEAR_FAILED_CONTROLLER.json)、[原未提交记录](EARLY_EVAL_CIFAR10_LINEAR_UNCOMMITTED_RECORD.json)、[实际执行 helper](CIFAR_LINEAR_EVAL_EXECUTION_HELPER.py)。', '']
    if report.get('formal_evals'):
        lines += ['| **seed / eval Run** | **own-best step** | **Accuracy (%) / 正确样本** | **完整性检查** | **CPU 回放** |', '|---|---:|---|---|---|']
        for r in report['formal_evals']:
            replay = r.get('cpu_full_test_replay')
            text = f"passed={replay['passed']}，Loss差{replay['loss_absolute_difference']:.12g}" if replay else '未执行'
            lines.append(f"| {r['seed']} / {r['id']} | {r['optimizer_step']} | {r['accuracy_pct']:.8f} / {r['correct_samples']} | {sum(r['checks'].values())}/{len(r['checks'])}，失败项 {r['failed_checks']} | {text} |")
    else:
        lines.append('本次未审查尚未齐全的四条正式 eval，也未启动 GPU eval。待主线程完成原计划四条 eval 后，可追加自身 best 的实际加载来源和结果核对。')
    if report.get('cpu_full_test_replay_performed'):
        lines += ['', '本次额外 CPU 完整 test 回放直接使用官方原始 test 文件、归档 Linear / Accuracy 定义及归档 test 归一化；严格加载和逐 tensor 检查 own-best，不增加正式 GPU Run。正式 eval 未导出在线模型终态哈希，权重来源证据为实际 resume_path、未变化的 canonical own-best、冻结加载器与独立 CPU 回放。', '']
    else:
        lines += ['', '本次没有执行额外 CPU 推理或模型权重回放；当前 --replay 仅支持 Linear。正式 eval 未导出在线模型终态哈希，其来源核对以实际 resume_path、未变化的 canonical own-best、冻结加载器和正式结果/日志为依据；CPU 重载保存状态不等同独立模型推理。', '']
    lines += ['## 五、证据与边界', '', '[预定曲线门](TARGET.md)、[原图估读](REFERENCE_CURVES.json)、[源码/展开配置 manifest](SOURCE_MANIFEST.json)、[数据 manifest](DATA_MANIFEST.json)。本报告的同名JSON包含完整400点、七锚点、每项检查、实际终态和哈希。', '',
              f"本次审计程序为 `{report['audit_script_path']}`，实际 SHA-256 为 `{report['audit_script_sha256']}`。原始 runs/checkpoint 不会随 Git clone 提供。冻结训练源与配置未改变。正式长训练未保存完整 RNG/迭代位置，本审查以无 resume 连续日志和正式终态为证据，不声称重新证明每一次未记录的 RNG 状态。", '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',choices=('MNIST','CIFAR10'),default='CIFAR10')
    parser.add_argument('--model',choices=('linear','mlp','cnn','resnet18'),default='linear')
    parser.add_argument('--output',type=Path,default=REPO/'.tmp/historical-group-audit')
    parser.add_argument('--with-eval',action='store_true')
    parser.add_argument('--replay',action='store_true',help='Additional CPU own-best test replay; needs formally succeeded evals and linear model.')
    args = parser.parse_args()
    if args.replay and (not args.with_eval or args.model != 'linear'):
        parser.error('--replay requires --with-eval and model=linear')
    index = read_json(STUDY/'index.json')
    groups = {}
    for e in index['experiments']:
        f = e['factors']
        if (f['data.name'],f['model.name']) == (args.data,args.model):
            groups[f['algorithm.mode']] = sorted(e['runs'],key=lambda r:r['seed'])
    if any([r['seed'] for r in groups.get(m,[])] != [0,1,2,3] for m in ('train','eval')):
        raise ValueError('index must supply exactly seeds0..3 for train and eval')
    now = datetime.now(timezone.utc)
    snapshot = whole_snapshot(index)
    report = {'schema':'rpipe.historical_completed_group_independent_cpu_audit.v1',
        'recorded_at_utc':now.isoformat(),'recorded_at_cst':now.astimezone().isoformat(),
        'data':args.data,'model':args.model,'source_baseline':BASELINE,'historical_commit':REF,
        'audit_script_path':str(Path(__file__)),'audit_script_sha256':sha(Path(__file__)),
        'scope':'Completed-group CPU reload/source/history audit; no training, process actions, GPU operations, or formal eval launches.',
        'training_group_complete':all(status(r)=='succeeded' for r in groups['train']),
        'run_integrity_audit_passed':False,'formal_eval_group_complete':False,
        'whole_study_final_gates_applied':False,'historical_reproduction_passed':None,
        'whole_study_status_snapshot':snapshot,'formal_flow_run_launched':False,'gpu_operations':False,
        'source_integrity':source_audit(),'runs':[], 'pending_train_runs':[], 'errors':[],
        'evidence_sha256':{str(p.relative_to(REPO)).replace('\\','/'):sha(p) for p in [STUDY/'docs/SOURCE_MANIFEST.json',STUDY/'docs/DATA_MANIFEST.json',STUDY/'docs/REFERENCE_CURVES.json',STUDY/'index.json']}}
    if report['training_group_complete']:
        for entry in groups['train']:
            try:
                report['runs'].append(train_audit(entry,args.data,args.model))
            except Exception as exc:
                report['errors'].append({'run':entry['id'],'error':f'{type(exc).__name__}: {exc}'})
        integrity = len(report['runs'])==4 and all(r['audit_passed'] for r in report['runs']) and report['source_integrity']['passed'] and not report['errors']
        report['run_integrity_audit_passed'] = integrity
        if integrity:
            report.update(curve_diagnostics(report['runs'],args.data,args.model,read_json(STUDY/'docs/REFERENCE_CURVES.json')))
        if args.with_eval:
            report['formal_eval_execution_audit'] = eval_execution_audit(args.data,args.model,groups['eval'])
            eval_complete = all(status(r)=='succeeded' for r in groups['eval'])
            report['pending_eval_runs'] = [{'id':r['id'],'seed':r['seed'],'status':status(r)} for r in groups['eval'] if status(r)!='succeeded']
            if integrity and eval_complete:
                inputs = None
                if args.replay:
                    x,y,info = test_data(args.data)
                    inputs = x,y
                    report['cpu_replay_data_provenance'] = info
                report['formal_evals'] = [eval_audit(e,t,args.data,args.model,inputs) for e,t in zip(groups['eval'],report['runs'])]
                report['formal_eval_group_complete'] = all(e['audit_passed'] for e in report['formal_evals'])
                if report['formal_eval_execution_audit'].get('available'):
                    report['formal_eval_group_complete'] &= report['formal_eval_execution_audit']['passed']
                report['cpu_full_test_replay_performed'] = args.replay
    else:
        report['pending_train_runs'] = [{'id':r['id'],'seed':r['seed'],'status':status(r)} for r in groups['train'] if status(r)!='succeeded']
    report['source_integrity_after'] = source_audit()
    if not report['source_integrity_after']['passed']:
        report['run_integrity_audit_passed'] = False
        report['formal_eval_group_complete'] = False
        report['errors'].append({'error':'Frozen source/config/data changed during audit'})
    output = args.output.resolve()
    # Formal reports are not created from merely nearing80000 or pending runs.
    if not report['training_group_complete'] and output == (STUDY/'docs').resolve():
        print(json.dumps({'status':'incomplete','pending_train_runs':report['pending_train_runs']},ensure_ascii=False))
        return 2
    output.mkdir(parents=True,exist_ok=True)
    stem = ('CIFAR' if args.data=='CIFAR10' else args.data)+'_'+args.model.upper()+'_RESULT'
    (output/f'{stem}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (output/f'{stem}.md').write_text(report_text(report),encoding='utf-8')
    print(json.dumps({'status':'audited' if report['run_integrity_audit_passed'] else 'incomplete_or_invalid',
                     'training_group_complete':report['training_group_complete'],
                     'run_integrity_audit_passed':report['run_integrity_audit_passed'],
                     'formal_eval_group_complete':report['formal_eval_group_complete'],
                     'errors':report['errors'], 'failed_checks':{r['id']:r['failed_checks'] for r in report['runs']},
                     'group_diagnostics':report.get('single_group_diagnostics'),
                     'outputs':[str(output/f'{stem}.json'),str(output/f'{stem}.md')]},ensure_ascii=False))
    if not report['run_integrity_audit_passed'] or (args.with_eval and not report['formal_eval_group_complete']):
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
