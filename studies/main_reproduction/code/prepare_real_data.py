from pathlib import Path
import hashlib,json,shutil,subprocess,sys
root=Path.cwd(); scratch=root/'.tmp/main-reproduction-20261003'; ref=scratch/'reference/src'; study=root/'studies/main_reproduction'
old={str(p.relative_to(root)):[p.stat().st_size,p.stat().st_mtime_ns] for p in (root/'studies').rglob('*') if p.is_file() and study not in p.parents}
(scratch/'prior-study-files.json').write_text(json.dumps(old,indent=2),encoding='utf-8')
rows=[]
def copy_verified(src,dst):
    if not dst.exists(): shutil.copytree(src,dst)
    for p in src.rglob('*'):
        if p.is_file():
            q=dst/p.relative_to(src)
            digest=hashlib.sha256(p.read_bytes()).hexdigest()
            assert digest==hashlib.sha256(q.read_bytes()).hexdigest()
            rows.append({'source':str(p.relative_to(root)),'destination':str(q.relative_to(root)),'sha256':digest,'size':p.stat().st_size})
cache=root/'studies/main_base/shared/data'
copy_verified(cache/'mnist/MNIST/raw',ref/'data/MNIST/raw')
copy_verified(root/'studies/support_model_smoke/shared/data/cifar10/cifar-10-batches-py',ref/'data/CIFAR10/raw/cifar-10-batches-py')
copy_verified(cache/'mnist/MNIST/raw',study/'shared/data/mnist/MNIST/raw')
copy_verified(root/'studies/support_model_smoke/shared/data/cifar10/cifar-10-batches-py',study/'shared/data/cifar10/cifar-10-batches-py')
(scratch/'data-copy-manifest.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
snapshot=scratch/'current-source'
shutil.copytree(root/'src/rpipe',snapshot/'src/rpipe',ignore=shutil.ignore_patterns('__pycache__'))
for f in ['pyproject.toml', 'studies/main_reproduction/study.yaml','studies/main_reproduction/experiment_config.yaml']:
    d=snapshot/f; d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/f,d)
files={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.rglob('*') if p.is_file()}
(scratch/'source-manifest.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
print('Copied and hashed',len(rows),'raw files; protected',len(old),'prior Study files; snapshotted',len(files),'source files')
