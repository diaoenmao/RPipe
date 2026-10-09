from pathlib import Path
import subprocess,time,json,sys,os
scratch=Path(__file__).resolve().parent
records=[]
env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
for mode,script in [('train','train_model.py'),('eval','test_model.py')]:
    for data in ['MNIST','CIFAR10']:
        for model in ['linear','mlp','cnn','resnet18']:
            tag=f'{data}_{model}'
            log=scratch/f'original-{mode}-{tag}.log'
            command=[sys.executable,str(scratch/'original_case.py'),script,'--control_name',tag]
            print('start',mode,tag,flush=True)
            start=time.perf_counter()
            with log.open('w',encoding='utf-8') as handle:
                result=subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,env=env)
            records.append({'mode':mode,'data':data,'model':model,'exit_code':result.returncode,'elapsed_seconds':time.perf_counter()-start,'log':str(log),'command':command})
            (scratch/'original-execution.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
            print('done',mode,tag,result.returncode,round(records[-1]['elapsed_seconds'],3),flush=True)
            if result.returncode:
                print(log.read_text(encoding='utf-8')[-4000:],flush=True)
                sys.exit(result.returncode)
print('Original base: 16/16 commands completed',flush=True)
