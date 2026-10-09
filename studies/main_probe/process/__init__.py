"""Aggregate only the paired Runs in the current index, never stale workspaces."""
import json
from rpipe.structure.artifact.index import load_index
from rpipe.structure.artifact._atomic import atomic_write_text


def run(ctx):
    if ctx.scope != 'study':
        return
    expected = {(data, model) for data in ('MNIST', 'CIFAR10')
                for model in ('linear', 'mlp', 'cnn', 'resnet18')}
    rows, missing, pairs, sources = [], [], [], set()
    index = load_index(ctx.study_dir)
    for experiment in index['experiments']:
        for run in experiment['runs']:
            root = ctx.study_dir / 'runs' / run['run_dir']
            result_path = root / 'result.json'
            report_path = root / 'assets' / 'probe' / 'COMPARISON.json'
            if not result_path.is_file() or not report_path.is_file():
                missing.append(run['id'])
                continue
            result = json.loads(result_path.read_text(encoding='utf-8'))
            report = json.loads(report_path.read_text(encoding='utf-8'))
            if result.get('status') != 'succeeded' or len(report.get('runs', [])) != 1:
                missing.append(run['id'])
                continue
            row = report['runs'][0]
            pair = (row['data'], row['model'])
            factors = experiment['factors']
            if pair != (factors['data.name'], factors['model.name']) or run['seed'] != 0:
                raise ValueError('probe evidence does not match the current index')
            pairs.append(pair)
            sources.add((report['main_commit'], report['dev_commit'], report['source_manifest_sha256'], report['device'], report.get('torch')))
            rows.append({'run': run['id'], 'data': pair[0], 'model': pair[1],
                         'passed': report.get('selected_passed') is True and report.get('selected_complete') is True
                         and row.get('passed') is True and report.get('source_unchanged') is True
                         and report.get('archive_after', {}).get('unchanged') is True
                         and report.get('main_commit') == '98648f3a5c7db7dccf3ca806410d5b6fdee9484c',
                         'evidence': str(report_path.relative_to(ctx.study_dir))})
    complete = not missing and len(pairs) == 8 and set(pairs) == expected and len(sources) == 1
    passed = complete and all(row['passed'] for row in rows)
    body = {'complete': complete, 'passed': passed, 'runs': rows, 'missing_runs': missing}
    atomic_write_text(ctx.study_dir / 'PROBE_COMPARISON.json', json.dumps(body, indent=2) + '\n')
    ctx.state['probe'] = body
    if not passed:
        raise RuntimeError('probe Study gate incomplete or failed; see PROBE_COMPARISON.json')
