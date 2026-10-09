from pathlib import Path
import json,shutil,yaml,subprocess,sys,os,time,hashlib
repo=Path.cwd();s=repo/'.tmp/main-reproduction-20261003';study=s/'deterministic-after-B017-study'
shutil.copytree(s/'deterministic-study/shared/data',study/'shared/data')
body=yaml.safe_load((s/'deterministic-study/study.yaml').read_text(encoding='utf-8'))
body['study']='main_reproduction_after_B017';body['fixed']['version']='main-98648f3-deterministic-after-B017';body['experiment']['name']=body['study']
(study/'study.yaml').write_text(yaml.safe_dump(body,sort_keys=False),encoding='utf-8')
shutil.copy2(s/'deterministic-study/experiment_config.yaml',study/'experiment_config.yaml')
manifest={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (repo/'src/rpipe').rglob('*.py')}
(s/'source-after-B017-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
for action in ['make','launch']:
    start=time.perf_counter();command=[sys.executable,'-m','rpipe',action,str(study),'--round','1','--num-gpus','1','--init-gpu','0','--console','shared']
    with (s/f'after-B017-{action}.log').open('w',encoding='utf-8') as handle:
        result=subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8'))
    print(action,result.returncode,round(time.perf_counter()-start,3),flush=True)
    if result.returncode:sys.exit(result.returncode)
code=(s/'compare_deterministic.py').read_text(encoding='utf-8-sig').replace("study=scratch/'deterministic-study'","study=scratch/'deterministic-after-B017-study'").replace("'deterministic-comparison.json'","'deterministic-comparison-after-B017.json'")
(s/'compare_after_B017.py').write_text(code,encoding='utf-8')
