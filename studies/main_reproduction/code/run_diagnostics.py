from pathlib import Path
import subprocess,os,sys,time,json
scratch=Path(__file__).resolve().parent;repo=scratch.parents[1]
records=[]
def run(command,label,env):
    print('start',label,flush=True)
    start=time.perf_counter();log=scratch/f'{label}.log'
    with log.open('w',encoding='utf-8') as handle:result=subprocess.run(command,cwd=repo,env=env,stdout=handle,stderr=subprocess.STDOUT)
    records.append({'label':label,'command':command,'exit_code':result.returncode,'elapsed_seconds':time.perf_counter()-start,'log':str(log)})
    (scratch/'diagnostic-execution.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print('done',label,result.returncode,round(records[-1]['elapsed_seconds'],3),flush=True)
    if result.returncode:
        print(log.read_text(encoding='utf-8')[-4000:],flush=True);sys.exit(result.returncode)
base=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',RPIPE_PROBE_DETERMINISTIC='0',RPIPE_REFERENCE_WORKDIR=str(scratch/'repeat'),RPIPE_REFERENCE_EVIDENCE=str(scratch/'repeat-evidence'))
for tag in ['MNIST_resnet18','MNIST_cnn','CIFAR10_resnet18']:
    run([sys.executable,str(scratch/'original_case.py'),'train_model.py','--control_name',tag],f'repeat-train-{tag}',base)
det=dict(base,CUBLAS_WORKSPACE_CONFIG=':4096:8',RPIPE_PROBE_DETERMINISTIC='1',RPIPE_REFERENCE_WORKDIR=str(scratch/'deterministic-reference'),RPIPE_REFERENCE_EVIDENCE=str(scratch/'deterministic-evidence'))
for mode,script in [('train','train_model.py'),('eval','test_model.py')]:
    for data in ['MNIST','CIFAR10']:
        for model in ['linear','mlp','cnn','resnet18']:
            tag=f'{data}_{model}'
            run([sys.executable,str(scratch/'original_case.py'),script,'--control_name',tag],f'deterministic-{mode}-{tag}',det)
current=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
for action in ['make','launch']:
    run([sys.executable,'-m','rpipe',action,str(scratch/'deterministic-study'),'--round','1','--num-gpus','1','--init-gpu','0','--console','shared'],f'deterministic-current-{action}',current)
print('All diagnostic commands completed',flush=True)
