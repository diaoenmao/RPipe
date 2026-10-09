from pathlib import Path
import os,sys,runpy,shutil
import numpy
scratch=Path(__file__).resolve().parent
ref=scratch/'reference/src'
sys.path[:0]=[str(scratch/'deps'),str(ref)]
work=Path(os.environ.get('RPIPE_REFERENCE_WORKDIR',str(ref)))
os.chdir(work)
evidence=Path(os.environ.get('RPIPE_REFERENCE_EVIDENCE',str(scratch/'original-evidence')))
import torch
torch.set_num_threads(2)
script=sys.argv[1]
sys.argv=[script]+sys.argv[2:]
if script=='train_model.py':
    import module,model
    from config import cfg
    check_original=module.check
    make_original=model.make_model
    def save_initial(config):
        result=make_original(config)
        folder=evidence/cfg['tag']
        folder.mkdir(parents=True,exist_ok=True)
        torch.save(result.state_dict(),folder/'initial.pt')
        torch.save(torch.get_rng_state(),folder/'rng-after-init.pt')
        return result
    def archive_checkpoint(result,path,*args,**kwargs):
        check_original(result,path,*args,**kwargs)
        target=evidence/cfg['tag']/f"step{cfg['step']}"
        shutil.copytree(path,target)
    model.make_model=save_initial
    module.check=archive_checkpoint
if os.environ.get('RPIPE_PROBE_DETERMINISTIC') == '1':
    import importlib
    entry=importlib.import_module(Path(script).stem)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    entry.main()
else:
    runpy.run_path(str(ref/script),run_name='__main__')
