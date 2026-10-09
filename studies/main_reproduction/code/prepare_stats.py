from pathlib import Path
import os, sys, runpy, json
import numpy as np
repo=Path(__file__).resolve().parents[2]
scratch=Path(__file__).resolve().parent
ref=scratch/'reference/src'
sys.path[:0]=[str(scratch/'deps'),str(ref)]
os.chdir(ref)
import torch, yaml
torch.set_num_threads(2)
runpy.run_path('make_dataset.py',run_name='__main__')
from module import make_stats
from dataset import make_dataset
from rpipe.structure.data.factory import DataFactory
from rpipe.structure.data.config import DataConfig
rows=[]
for name,folder in [('MNIST','mnist'),('CIFAR10','cifar10')]:
    stats=make_stats(name)
    old=make_dataset(name,verbose=False)
    metadata=yaml.safe_load((repo/f'studies/main_base/shared/data/{folder}/stats.yaml').read_text(encoding='utf-8'))
    metadata.update(mean=stats.mean.tolist(),std=stats.std.tolist(),stats_source='main 98648f3 module.Stats, full train, sequential batch250')
    metadata['splits']['train']['pixel'].update(mean=stats.mean.tolist(),std=stats.std.tolist())
    output=repo/f'studies/main_reproduction/shared/data/{folder}/stats.yaml'
    output.write_text(yaml.safe_dump(metadata,sort_keys=False),encoding='utf-8')
    new=DataFactory.build(DataConfig.from_mapping({'name':name,'source':'torch','config':{'batch_size':250,'test_batch_ratio':4,'augment':True}}),repo/'studies/main_reproduction/shared/data',seed=0,origin='foreign')
    train=new._train_set; test=new._loaders['test'].dataset
    for split,ds in [('train',train),('test',test)]:
        actual=ds.data.numpy() if isinstance(ds.data,torch.Tensor) else ds.data
        assert np.array_equal(old[split].data,actual), (name,split,'pixels')
        assert np.array_equal(old[split].target,np.asarray(ds.targets)), (name,split,'labels')
    assert new.meta['mean']==tuple(stats.mean.tolist()) and new.meta['std']==tuple(stats.std.tolist())
    rows.append({'data':name,'mean':stats.mean.tolist(),'std':stats.std.tolist(),'train':len(train),'test':len(test),'all_pixels_labels_equal':True,'stats_pixel_count':stats.n_samples})
    print(json.dumps(rows[-1]),flush=True)
(scratch/'data-parity.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
print('Full data and original-statistics parity passed',flush=True)
