"""Reproducible three-seed laptop benchmark, with stage logs and provenance."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--eval-users',type=int,default=1000)
    args=p.parse_args()
    os.chdir(ROOT)
    out=ROOT/'work'/'benchmark_20261029'
    out.mkdir(parents=True,exist_ok=True)
    config={'seeds':args.seeds,'eval_users':args.eval_users,'eval_user_seed':42,
            'head_epochs':3,'rq_warmup_epochs':10,'rq_epochs':15,'gen_epochs':5,'max_windows':60000,
            'beam':30,'n_codes':256,'device':'cuda','hist_len':10,
            'note':'C=pretrained MRL prefix; backbone frozen in all methods. B=QR matrix, not full SpecMRL.'}
    import torch
    manifest={'config':config,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
              'cuda':torch.version.cuda,'python':sys.version,
              'model_cache_metadata':json.loads((ROOT/'work/full768.metadata.json').read_text(encoding='utf-8')),
              'data_audit':json.loads((ROOT/'work/beauty.json').read_text(encoding='utf-8'))['audit'],
              'packages':{n:importlib.metadata.version(n) for n in ['numpy','scikit-learn','transformers','sentence-transformers','torchvision']},
              'sha256':{str(f.relative_to(ROOT)):sha(f) for f in [ROOT/'mini_tiger.py',ROOT/'work/beauty.json',ROOT/'work/full768.npy',ROOT/'data/reviews_Beauty_5.json.gz',ROOT/'data/meta_Beauty.json.gz']}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    timings=[]
    for seed in args.seeds:
        run_dir=out/f'seed{seed}'
        run_dir.mkdir(exist_ok=True)
        for kind in ['mrl','linear','ortho']:
            head=run_dir/f'{kind}_128.npy'
            sid=run_dir/f'{kind}_sid.npy'
            gen=run_dir/f'{kind}_gen.pt'
            result=run_dir/f'{kind}_test.json'
            common=['--device','cuda']
            jobs=[('head',head,['head','--kind',kind,'--data','work/beauty.json','--embeddings','work/full768.npy','--out',head,'--epochs',3,'--seed',seed]+common),
                  ('rq',sid,['rqvae','--embeddings',head,'--out',sid,'--epochs',15,'--seed',seed]+common),
                  ('gen',gen,['train-gen','--data','work/beauty.json','--sids',sid,'--out',gen,'--epochs',5,'--seed',seed]+common),
                  ('eval',result,['eval','--data','work/beauty.json','--sids',sid,'--checkpoint',gen,'--out',result,'--eval-users',args.eval_users,'--seed',42]+common)]
            for stage,target,argv in jobs:
                if target.exists():
                    print('RESUME skip',target.name,flush=True)
                    continue
                log=run_dir/f'{kind}_{stage}.log'
                cmd=[sys.executable,'-u','mini_tiger.py']+list(map(str,argv))
                print(f'START seed={seed} {kind} {stage}',flush=True)
                start=time.perf_counter()
                with log.open('w',encoding='utf-8') as f:
                    subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
                elapsed=time.perf_counter()-start
                timings.append({'seed':seed,'kind':kind,'stage':stage,'seconds':elapsed})
                with (out/'timings.jsonl').open('a') as f:
                    f.write(json.dumps(timings[-1])+'\n')
                print(f'DONE seed={seed} {kind} {stage}: {elapsed:.1f}s',flush=True)
    print('ALL COMPLETED',flush=True)

if __name__=='__main__':
    main()
