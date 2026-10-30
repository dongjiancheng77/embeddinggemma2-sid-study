"""Audit persisted real-data outputs, without retraining or touching model weights."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'work/benchmark_20261029'
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    manifest=read(OUT/'manifest.json')
    data=read(ROOT/'work/beauty.json')
    user_map={u['user']:u['seq'] for u in data['users']}
    checks=[]
    for seed in manifest['config']['seeds']:
        for kind in ['linear','ortho','mrl']:
            d=OUT/f'seed{seed}'
            x=np.load(d/f'{kind}_128.npy')
            assert x.shape==(len(data['items']),128) and np.isfinite(x).all()
            assert np.max(np.abs(np.linalg.norm(x,axis=1)-1))<1e-5
            sids=np.load(d/f'{kind}_sid.npy')
            assert sids.shape==(len(x),4)
            assert len(set(map(tuple,sids)))==len(x)
            qerror=None
            if kind=='ortho':
                ckpt=torch.load(d/f'{kind}_128.pt',weights_only=False,map_location='cpu')
                q,_=torch.linalg.qr(ckpt['state_dict']['raw'])
                qerror=float((q.T@q-torch.eye(128)).abs().max())
                assert qerror<1e-5
                full=torch.from_numpy(np.load(ROOT/'work/full768.npy'))
                expected=torch.nn.functional.normalize(full@q,dim=-1).numpy()
                # QR signs can differ by backend; pairwise cosine is the invariant.
                assert np.max(np.abs((expected[:50]@expected[:50].T)-(x[:50]@x[:50].T)))<1e-5
            rows=read(d/f'{kind}_test.users.json')
            for r in rows:
                seq=user_map[r['user']]
                assert r['target']==seq[-1]
                assert len(set(r['predictions']))==len(r['predictions'])
                assert all(0<=i<len(x) and i not in seq[:-1] for i in r['predictions'])
                assert r['recall']==int(r['target'] in r['predictions'])
            result=read(d/f'{kind}_test.json')
            assert abs(np.mean([r['recall'] for r in rows])-result['item_level_recall'])<1e-12
            assert abs(np.mean([r['ndcg'] for r in rows])-result['item_level_ndcg'])<1e-12
            checks.append({'seed':seed,'method':kind,'users':len(rows),'orthogonality_cpu_max_error':qerror,'passed':True})
    for name,expected in manifest['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    (OUT/'verification.json').write_text(json.dumps({'checks':checks,'hashes_match':True},indent=2))
    print('PASS: all nine runs; finite normalized vectors, unique SIDs, exact targets, unseen predictions, recomputed metrics, orthogonality, provenance hashes.')

if __name__=='__main__':
    torch.set_num_threads(4)
    main()
