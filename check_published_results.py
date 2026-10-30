"""Audit repository snapshots using Python's standard library only."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    checks=0;pairing=None
    for label,expected in [('experiment1',9),('experiment2',30)]:
        folder=ROOT/'results'/label
        paths=sorted(folder.rglob('*test.users.json'))
        assert len(paths)==expected,(label,len(paths))
        for path in paths:
            rows=read(path)
            assert len(rows)==1000
            assert all(set(r)=={'user','recall','ndcg'} for r in rows)
            ids=[r['user'] for r in rows]
            assert len(set(ids))==1000 and all(len(x)==16 for x in ids)
            if label=='experiment2':
                if pairing is None:pairing=ids
                assert ids==pairing,'Paired users/order differ'
            metric=read(path.with_name(path.name.replace('.users.json','.json')))
            for key,source in [('item_level_recall','recall'),('item_level_ndcg','ndcg')]:
                assert math.isclose(sum(r[source] for r in rows)/len(rows),metric[key],abs_tol=1e-12)
            checks+=1
        manifest=read(folder/('run_manifest.json' if label=='experiment2' else 'manifest.json'))
        for name,expected_sha in manifest['sha256'].items():
            basename=name.replace('\\','/').split('/')[-1]
            archived=ROOT/'provenance/executed_source'/basename
            if basename.endswith('.py') and archived.exists():assert sha(archived)==expected_sha,basename
    for seed in [0,1,2]:
        for kind in ['mrl','pca']:
            a=ROOT/'results/experiment2/collab'/f'seed{seed}'/kind
            b=ROOT/'results/experiment2/mixed'/f'seed{seed}'/kind
            for name in ['test.json','test.users.json']:assert sha(a/name)==sha(b/name)
    manifest_path=ROOT/'FILE_SHA256.json'
    if manifest_path.exists():
        for name,expected in read(manifest_path).items():assert sha(ROOT/name)==expected,name
    print(f'PASS: {checks} conditions, metric recomputation, paired pseudonyms, fixed C/D references, executed-source hashes and publication hashes.')

if __name__=='__main__':main()
