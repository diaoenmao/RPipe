from pathlib import Path
import os,sys,json,pickle,math,re
import numpy
scratch=Path(__file__).resolve().parent; repo=scratch.parents[1];ref=scratch/'reference/src'
sys.path[:0]=[str(scratch/'deps'),str(ref)];os.chdir(ref)
import torch,yaml
torch.set_num_threads(2)
study=repo/'studies/main_reproduction'
index=json.loads((study/'index.json').read_text(encoding='utf-8'))
lookup={}
for exp in index['experiments']:
    f=exp['factors']; key=(f['data.name'],f['model.name'],f['algorithm.mode'])
    assert len(exp['runs'])==1 and exp['runs'][0]['seed']==0
    lookup[key]=exp['runs'][0]['id']
assert len(lookup)==16

def source(folder):
    body={}
    for name in ['cfg','model','optimizer','scheduler','logger']:
        path=folder/name
        if name in ['scheduler','logger']:
            with path.open('rb') as handle:body[name]=pickle.load(handle)
        else:body[name]=torch.load(path,map_location='cpu',weights_only=False)
    return body

def weights(a,b):
    a={k.removeprefix('model.'):v for k,v in a.items()}
    b={k.removeprefix('net.'):v for k,v in b.items()}
    assert a.keys()==b.keys(),(a.keys(),b.keys())
    delta=max((float((a[k]-b[k]).abs().max()) for k in a),default=0.)
    passed=all(torch.allclose(a[k],b[k],atol=1e-6,rtol=1e-5) if a[k].is_floating_point() else torch.equal(a[k],b[k]) for k in a)
    return delta,passed

rows=[]
for data in ['MNIST','CIFAR10']:
    for model in ['linear','mlp','cnn','resnet18']:
        train_id=lookup[(data,model,'train')];eval_id=lookup[(data,model,'eval')]
        root=study/'runs'/train_id; eval_root=study/'runs'/eval_id
        result=json.loads((root/'result.json').read_text(encoding='utf-8'))
        independent=json.loads((eval_root/'result.json').read_text(encoding='utf-8'))
        assert result['status']==independent['status']=='succeeded'
        config=yaml.safe_load((root/'config.yaml').read_text(encoding='utf-8'))
        assert config['algorithm']['num_steps']==60 and config['algorithm']['eval_period']==30 and config['algorithm']['eval_num_steps']==-1
        assert result['structure']['data']['test_size']==10000
        records=[]
        for step in [30,60]:
            old=source(scratch/'original-evidence'/f'0_{data}_{model}'/f'step{step}')
            new=torch.load(root/'assets/checkpoints'/f'step_{step:06}.pt',map_location='cpu',weights_only=False)
            assert old['cfg']['step']==new['step']==step
            assert old['logger']['counter']['test/Loss']==10000
            assert new['tracker']['progress']['step']==step
            metric_old={k:float(old['logger']['mean'][k]) for k in ['train/Loss','train/Accuracy','test/Loss','test/Accuracy']}
            metric_new={k:float(new['tracker']['last_segment'][k.split('/')[0]][k.split('/')[1]]) for k in metric_old}
            delta={k:metric_new[k]-metric_old[k] for k in metric_old}
            maximum,param_ok=weights(old['model'],new['model'])
            scheduler_ok=old['scheduler']==new['scheduler']
            # A float32 percentage is not an exact integer count; recover the count from 10000 test images.
            accuracy_count_equal=round(metric_old['test/Accuracy']*100)==round(metric_new['test/Accuracy']*100)
            passed=abs(delta['train/Loss'])<=1e-6 and abs(delta['test/Loss'])<=1e-6 and accuracy_count_equal and param_ok and scheduler_ok
            records.append({'step':step,'original':metric_old,'current':metric_new,'delta':delta,'parameter_max_delta':maximum,'parameters_pass':param_ok,'scheduler_equal':scheduler_ok,'test_correct_count_equal':accuracy_count_equal,'passed':passed})
        best_old=source(ref/'output/exp'/f'0_{data}_{model}'/'best')
        best_new=torch.load(root/'assets/checkpoints/best.pt',map_location='cpu',weights_only=False)
        expected_best=30 if records[0]['original']['test/Loss']<=records[1]['original']['test/Loss'] else 60
        assert best_old['cfg']['step']==expected_best
        best_same=best_old['cfg']['step']==best_new['step']
        # Verify each artifact's chosen weights are its own expected snapshot, independent of cross-implementation parity.
        assert best_new['step']==min(records,key=lambda r:r['current']['test/Loss'])['step']
        chosen_new=torch.load(root/'assets/checkpoints'/f"step_{best_new['step']:06}.pt",map_location='cpu',weights_only=False)
        assert all(torch.equal(best_new['model'][k],chosen_new['model'][k]) for k in best_new['model'])
        reference_result=torch.load(ref/'output/result'/f'0_{data}_{model}',map_location='cpu',weights_only=False)
        old_eval={k:float(reference_result['logger']['test']['mean'][f'test/{k}']) for k in ['Loss','Accuracy']}
        assert reference_result['logger']['test']['counter']['test/Loss']==10000
        new_eval={'Loss':independent['metrics']['test_loss'],'Accuracy':independent['metrics']['test_accuracy']}
        eval_delta={k:new_eval[k]-old_eval[k] for k in old_eval}
        new_parent=next(r['current'] for r in records if r['step']==best_new['step'])
        old_parent=next(r['original'] for r in records if r['step']==best_old['cfg']['step'])
        current_eval_matches_best=abs(new_eval['Loss']-new_parent['test/Loss'])<=1e-6 and round(new_eval['Accuracy']*100)==round(new_parent['test/Accuracy']*100)
        original_eval_matches_best=abs(old_eval['Loss']-old_parent['test/Loss'])<=1e-6 and round(old_eval['Accuracy']*100)==round(old_parent['test/Accuracy']*100)
        eval_pass=abs(eval_delta['Loss'])<=1e-6 and round(new_eval['Accuracy']*100)==round(old_eval['Accuracy']*100)
        log=(eval_root/'assets/logs/run.log').read_text(encoding='utf-8')
        assert train_id in log and '[resume]' in log and f"step={best_new['step']}" in log
        for directory in [root,eval_root]:
            text=(directory/'assets/logs/run.log').read_text(encoding='utf-8')
            assert '[error]' not in text and text.count('[flow] start')==1
            if directory==root:assert '[resume]' not in text
        train_scalars=[json.loads(line) for line in (root/'assets/tracker/scalars.jsonl').read_text(encoding='utf-8').splitlines()]
        test_steps=[r['optimizer_step'] for r in train_scalars if r.get('split')=='test' and r.get('name')=='Loss']
        assert test_steps==[30,60]
        row={'data':data,'model':model,'train_id':train_id,'eval_id':eval_id,'steps':records,'best_original':best_old['cfg']['step'],'best_current':best_new['step'],'best_step_equal':best_same,'eval_original':old_eval,'eval_current':new_eval,'eval_delta':eval_delta,'eval_cross_pass':eval_pass,'eval_matches_own_best':{'original':original_eval_matches_best,'current':current_eval_matches_best},'passed':all(r['passed'] for r in records) and best_same and eval_pass and current_eval_matches_best and original_eval_matches_best}
        rows.append(row)
        print(data,model,'PASS' if row['passed'] else 'FAIL',[(r['step'],r['parameter_max_delta'],r['delta']['test/Loss'],r['delta']['test/Accuracy']) for r in records],flush=True)
body={'recipe':'main 98648f3 base, seed0, 60-step/eval30, same current environment','loss_atol':1e-6,'parameter_atol':1e-6,'parameter_rtol':1e-5,'accuracy_gate':'equal correct counts over 10000 samples','passed':sum(r['passed'] for r in rows),'total':len(rows),'rows':rows}
(scratch/'full-comparison.json').write_text(json.dumps(body,indent=2),encoding='utf-8')
print(f"Full numerical parity: {body['passed']}/{body['total']}",flush=True)
