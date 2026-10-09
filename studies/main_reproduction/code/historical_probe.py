from pathlib import Path
import os,sys,json,ast,types
import numpy
repo=Path(__file__).resolve().parents[2];scratch=Path(__file__).resolve().parent;old=scratch/'historical-4ccb28d/src'
sys.path.insert(0,str(old));os.chdir(old)
import torch
from config import cfg
from module import process_control
from model import make_model
from rpipe.structure.model.factory import ModelFactory
from rpipe.structure.model.config import ModelConfig
from rpipe.structure.algorithm.eval_hook import should_early_stop
torch.set_num_threads(2)
rows=[]
for data,shape in [('MNIST',(1,28,28)),('CIFAR10',(3,32,32))]:
    for name in ['linear','mlp','cnn','resnet18']:
        cfg['control']={'data_name':data,'model_name':name};process_control()
        torch.manual_seed(0);a=make_model(name)
        torch.manual_seed(0);b=ModelFactory.build(ModelConfig(name=name),scratch/'historical-preflight',data_meta={'data_size':shape,'target_size':10}).module
        state_old={k.replace('linear.','output_proj.'):v for k,v in a.state_dict().items()};state_new=b.state_dict()
        exact_keys=state_old.keys()==state_new.keys()
        initial_equal=exact_keys and all(torch.equal(state_old[k],state_new[k]) for k in state_old)
        x=torch.rand((2,*shape),generator=torch.Generator().manual_seed(17));target=torch.tensor([0,1])
        with torch.no_grad():
            pred_old=a({'data':x,'target':target})['target'];pred_new=b(x)
        row={'data':data,'model':name,'mapped_state_keys_equal':exact_keys,'initial_parameters_buffers_equal':initial_equal,'old_batchnorm_layers':sum(isinstance(m,torch.nn.BatchNorm2d) for m in a.modules()),'current_batchnorm_layers':sum(isinstance(m,torch.nn.BatchNorm2d) for m in b.modules()),'logits_max_delta':float((pred_old-pred_new).abs().max())}
        rows.append(row);print(json.dumps(row),flush=True)
# Execute each archived compare method directly, without editing historical source.
def method(path):
    tree=ast.parse(path.read_text(encoding='utf-8'));cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Metric');fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='compare');ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns);return ns['compare']
comparisons={}
for label,path in [('historical',old/'metric/metric.py'),('current_main',scratch/'reference/src/metric/metric.py')]:
    fn=method(path);instance=types.SimpleNamespace(best=-float('inf'),best_direction='up');chosen=None;steps=[]
    for index,value in enumerate([95.0,90.0,92.0],1):
        improved=fn(instance,value,True)
        if improved:chosen=index
        steps.append({'accuracy':value,'improved':improved,'comparison_baseline_after':instance.best,'chosen_checkpoint':chosen})
    comparisons[label]=steps
best=None;chosen=None;steps=[]
for index,value in enumerate([95.0,90.0,92.0],1):
    previous=best;_,best,_=should_early_stop(value=value,best=best,stall=0,patience=None,min_delta=0.0,mode='max')
    improved=previous is None or value>previous
    if improved:chosen=index
    steps.append({'accuracy':value,'improved':improved,'comparison_baseline_after':best,'chosen_checkpoint':chosen})
comparisons['rpipe']=steps
assert comparisons['historical'][-1]['chosen_checkpoint']==3
assert comparisons['current_main'][-1]['chosen_checkpoint']==comparisons['rpipe'][-1]['chosen_checkpoint']==1
body={'historical_commit':'4ccb28d0496110253e9f8e3f3df658853f07996b','scope':'CPU model initialization/forward probe on fixed synthetic inputs; no historical long training or image reproduction','models':rows,'best_selection_probe':comparisons}
(scratch/'historical-audit.json').write_text(json.dumps(body,indent=2),encoding='utf-8')
print('Historical model and checkpoint-selection audit completed',flush=True)
