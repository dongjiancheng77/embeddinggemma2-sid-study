"""Audit all persisted methods/settings and shared fixed references."""
import json
import hashlib
import math
import sys
import numpy as np
import torch
from experiment2_data import ROOT,OUT,read,sha

def main():
    torch.set_num_threads(4)
    manifest=read(OUT/'run_manifest.json')
    data=read(ROOT/'work/beauty.json')
    histories={u['user']:u['seq'] for u in data['users']}
    checks=[]
    partial='--partial' in sys.argv
    for setting in ['collab','mixed']:
        for seed in [0,1,2]:
            for kind in ['linear','ortho','mrl','pca','lora']:
                path=OUT/setting/f'seed{seed}'/kind
                if partial and not (path/'complete.json').exists():continue
                assert (path/'complete.json').exists(),str(path)
                x=np.load(path/'items128.npy');sids=np.load(path/'sid.npy')
                assert x.shape==(len(data['items']),128) and np.isfinite(x).all()
                assert np.abs(np.linalg.norm(x,axis=-1)-1).max()<1e-5
                assert len(set(map(tuple,sids)))==len(x)
                stats=read(path/'stage1.json')
                if kind=='ortho':assert stats['orthogonality_max_error']<1e-5
                if kind=='lora':
                    assert stats['gradient_audit']['nonzero_trainable_gradient_tensors']>0
                    assert stats['gradient_audit']['unexpected_frozen_gradients']==0
                    assert (path/'adapter/adapter_model.safetensors').stat().st_size>0
                if kind=='pca':
                    pca=np.load(path/'pca.npz')
                    assert np.max(np.abs(pca['components']@pca['components'].T-np.eye(128)))<1e-5
                rows=read(path/'test.users.json')
                for row in rows:
                    seq=histories[row['user']]
                    assert row['target']==seq[-1]
                    pred=row['predictions']
                    assert len(set(pred))==len(pred) and all(0<=i<len(x) and i not in seq[:-1] for i in pred)
                    assert row['recall']==int(row['target'] in pred)
                    expected=1/math.log2(pred.index(row['target'])+2) if row['target'] in pred else 0
                    assert abs(row['ndcg']-expected)<1e-12
                result=read(path/'test.json')
                assert len(rows)==1000 and len({r['user'] for r in rows})==1000
                assert abs(np.mean([r['recall'] for r in rows])-result['item_level_recall'])<1e-12
                assert abs(np.mean([r['ndcg'] for r in rows])-result['item_level_ndcg'])<1e-12
                if setting=='mixed' and kind in ['mrl','pca']:
                    other=OUT/'collab'/f'seed{seed}'/kind
                    for name in ['items128.npy','sid.npy','gen.pt','test.users.json']:assert sha(path/name)==sha(other/name)
                checks.append({'setting':setting,'seed':seed,'kind':kind,'passed':True})
    for name,expected in manifest['sha256'].items():assert sha(ROOT/name)==expected,name
    d=read(OUT/'dataset.json')
    seen=[set() for _ in range(2)]
    for i,key in enumerate(['text_train','text_validation']):
        for r in d['examples'][key]:seen[i].update(d['texts'][r[k]] for k in ['q','p','n'])
    assert not(seen[0]&seen[1])
    report={'conditions':checks,'normal_text_train_validation_overlap':0,'provenance_hashes_match':True,
            'lora_gradient_cache_verification':read(OUT/'lora_gradient_verification.json')}
    assert partial or len(checks)==30
    (OUT/('verification_partial.json' if partial else 'verification.json')).write_text(json.dumps(report,indent=2))
    print(f'PASS: {len(checks)} conditions, normalization, SID uniqueness, history filtering, paired users, adapter gradients, fixed-reference equality, source hashes, text holdout separation.')

if __name__=='__main__':main()
