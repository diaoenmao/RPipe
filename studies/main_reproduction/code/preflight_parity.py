from pathlib import Path
import os,sys,json,hashlib
import numpy
scratch=Path(__file__).resolve().parent; repo=scratch.parents[1]; ref=scratch/'reference/src'
sys.path[:0]=[str(scratch/'deps'),str(ref)];os.chdir(ref)
import torch,yaml
from config import cfg
from module import process_control
from dataset import make_dataset,make_data_loader,process_dataset
from rpipe.structure.data.factory import DataFactory
from rpipe.structure.data.config import DataConfig
from rpipe.structure.model.factory import ModelFactory
from rpipe.structure.model.config import ModelConfig
from rpipe.structure.system.runtime import apply_runtime
from rpipe.structure.system.config import SystemConfig
torch.set_num_threads(2)
rows=[]
for data,folder in [('MNIST','mnist'),('CIFAR10','cifar10')]:
    current_data=DataFactory.build(DataConfig.from_mapping({'name':data,'source':'torch','config':{'batch_size':250,'test_batch_ratio':4,'pin_memory':True,'num_workers':0,'augment':True}}),repo/'studies/main_reproduction/shared/data',seed=0,origin='foreign')
    cfg.update(seed=0,tag=f'0_{data}_linear',control={'data_name':data,'model_name':'linear'})
    process_control(); source_data=process_dataset(make_dataset(data,verbose=False))
    cfg['step']=0
    old_loader=make_data_loader(source_data,cfg[cfg['tag']]['optimizer']['batch_size'],cfg['num_steps'],cfg['step'],cfg['step_period'],cfg['pin_memory'],cfg['num_workers'],cfg['collate_mode'],cfg['seed'])
    current_data.rebind_train_steps(step=0,num_steps=60,step_period=1)
    a=list(old_loader['train'].sampler);b=list(current_data._loaders['train'].sampler)
    assert a==b and len(a)==15000
    sampler_hash=hashlib.sha256(numpy.array(a,dtype=numpy.int64).tobytes()).hexdigest()
    for model in ['linear','mlp','cnn','resnet18']:
        apply_runtime(0,SystemConfig.from_mapping({'deterministic':False,'cudnn_benchmark':True}))
        built=ModelFactory.build(ModelConfig(name=model,source='custom_torch'),scratch/'preflight/models',data_meta=current_data.meta)
        expected=torch.load(scratch/'original-evidence'/f'0_{data}_{model}'/'initial.pt',map_location='cpu',weights_only=False)
        old={k.removeprefix('model.'):v for k,v in expected.items()}
        new={k.removeprefix('net.'):v for k,v in built.module.state_dict().items()}
        assert old.keys()==new.keys()
        assert all(torch.equal(old[k],new[k]) for k in old),(data,model,'init')
        rng=torch.load(scratch/'original-evidence'/f'0_{data}_{model}'/'rng-after-init.pt',weights_only=False)
        assert torch.equal(torch.get_rng_state(),rng),(data,model,'rng')
        rows.append({'data':data,'model':model,'initial_parameters_equal':True,'rng_after_init_equal':True,'all_15000_sample_indices_equal':True,'sampler_sha256':sampler_hash})
(scratch/'preflight-parity.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
print('8/8 full-data initialization, RNG and 15000-sample-order preflight cases passed',flush=True)
