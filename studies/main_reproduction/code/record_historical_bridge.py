import hashlib
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[2]
scratch = Path(__file__).resolve().parent
work = scratch / 'historical-bridge-7ec0aaaec0e1'
docs = root / 'studies/main_reproduction/docs'
body = json.loads((work / 'comparison.json').read_text(encoding='utf-8'))
assert len(body['cases']) == 8 and all(row['passed'] for row in body['cases'])
assert hashlib.sha256((scratch / 'historical_bridge.py').read_bytes()).hexdigest() == body['script_sha256']
raw = json.loads((work / 'data-copy-manifest.json').read_text(encoding='utf-8'))
assert all(hashlib.sha256((work / path).read_bytes()).hexdigest() == digest for path, digest in raw.items())
source = json.loads((docs / 'SOURCE_AFTER_B017_MANIFEST.json').read_text(encoding='utf-8'))
changed = [path for path, digest in source.items()
           if hashlib.sha256((root / path).read_bytes()).hexdigest() != digest]
assert not changed, changed
prior = json.loads((scratch / 'prior-study-files.json').read_text(encoding='utf-8'))
changed_prior = []
for name, before in prior.items():
    stat = (root / name).stat()
    if [stat.st_size, stat.st_mtime_ns] != before:
        changed_prior.append(name)
assert not changed_prior, changed_prior
archives = {}
for archive, extracted in [('historical-4ccb28d.zip', 'historical-4ccb28d'),
                           ('main-98648f3.zip', 'reference')]:
    with zipfile.ZipFile(scratch / archive) as z:
        names = [name for name in z.namelist() if not name.endswith('/')]
        assert all((scratch / extracted / name).read_bytes() == z.read(name) for name in names)
        archives[archive] = len(names)
body['verification'] = {'source_files_unchanged': len(source), 'prior_study_files_unchanged': len(prior),
    'archived_files_unchanged': archives, 'raw_files_sha256_rechecked': len(raw),
    'coverage_note': 'Prior readable Study files only; inaccessible main_base CIFAR10 directory was not accessed.'}
(docs / 'HISTORICAL_BRIDGE.json').write_text(json.dumps(body, indent=2), encoding='utf-8')
(docs / 'HISTORICAL_BRIDGE_DATA_MANIFEST.json').write_text(json.dumps(raw, indent=2), encoding='utf-8')
print(json.dumps(body['verification']))
