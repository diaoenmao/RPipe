from pathlib import Path
import subprocess,sys,os,time,json
scratch=Path(__file__).resolve().parent
repo=scratch.parents[1]
command=[sys.executable,'-m','rpipe','launch','studies/main_reproduction','--round','1','--num-gpus','1','--init-gpu','0','--console','shared']
start=time.perf_counter()
with (scratch/'current-launch.log').open('w',encoding='utf-8') as handle:
    result=subprocess.run(command,cwd=repo,stdout=handle,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1'))
body={'command':command,'exit_code':result.returncode,'elapsed_seconds':time.perf_counter()-start,'log':str(scratch/'current-launch.log')}
(scratch/'current-execution.json').write_text(json.dumps(body,indent=2),encoding='utf-8')
print(json.dumps(body),flush=True)
sys.exit(result.returncode)
