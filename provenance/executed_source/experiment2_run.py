"""Five methods x two supervision settings x three paired seeds on one GPU."""
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from experiment2_data import ROOT,OUT,read,sha

def main():
    os.chdir(ROOT)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    manifest_path=OUT/'run_manifest.json'
    if not manifest_path.exists():
        import torch
        protocol=read(OUT/'dataset.json')['protocol']
        protocol.update(negatives='31 other in-batch positives + all 32 sampled negatives shared across the batch; positive duplicates and seen items masked.',
                        lora_gradient_cache_microbatch=32,downstream={'rq_warmup_epochs':10,'rq_epochs':15,'gen_epochs':5,'gen_max_windows':60000,'gen_hist_len':10,'beam':30,'eval_users':1000,'eval_seed':42})
        manifest={'seeds':[0,1,2],'settings':['collab','mixed'],'methods':['mrl','pca','linear','ortho','lora'],
                  'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'protocol':protocol,
                  'packages':{n:importlib.metadata.version(n) for n in ['torch','torchvision','transformers','sentence-transformers','peft','pyarrow','scikit-learn']},
                  'sha256':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'mini_tiger.py',ROOT/'experiment2_train.py',ROOT/'experiment2_data.py',OUT/'dataset.json',OUT/'base768.npy',ROOT/'work/beauty.json']},
                  'data_manifest':read(OUT/'data_manifest.json')}
        manifest['data_manifest']['protocol']=protocol
        manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    else:
        for name,expected in read(manifest_path)['sha256'].items():
            assert sha(ROOT/name)==expected,f'Input/code changed since run registration: {name}'
    for seed in [0,1,2]:
        for setting in ['collab','mixed']:
            for kind in ['mrl','pca','linear','ortho','lora']:
                run=OUT/setting/f'seed{seed}'/kind
                run.mkdir(parents=True,exist_ok=True)
                if (run/'complete.json').exists():continue
                if setting=='mixed' and kind in ['mrl','pca']:
                    source=OUT/'collab'/f'seed{seed}'/kind
                    assert (source/'complete.json').exists()
                    for p in source.iterdir():
                        if p.is_file() and p.name!='complete.json':shutil.copy2(p,run/p.name)
                    (run/'complete.json').write_text(json.dumps({'reused_fixed_reference':str(source)}))
                    print('REFERENCE REUSE',setting,seed,kind,flush=True)
                    continue
                jobs=[('stage1','stage1.json',['experiment2_train.py','--kind',kind,'--setting',setting,'--seed',seed]),
                      ('rq','sid.stats.json',['mini_tiger.py','rqvae','--embeddings',run/'items128.npy','--out',run/'sid.npy','--device','cuda','--seed',seed]),
                      ('gen','gen.pt',['mini_tiger.py','train-gen','--data','work/beauty.json','--sids',run/'sid.npy','--out',run/'gen.pt','--device','cuda','--seed',seed]),
                      ('eval','test.json',['mini_tiger.py','eval','--data','work/beauty.json','--sids',run/'sid.npy','--checkpoint',run/'gen.pt','--out',run/'test.json','--device','cuda','--seed',42])]
                for stage,target,argv in jobs:
                    if (run/target).exists():continue
                    print('START',seed,setting,kind,stage,flush=True)
                    start=time.perf_counter()
                    with (run/f'{stage}.log').open('w',encoding='utf-8') as f:
                        subprocess.run([sys.executable,'-u']+list(map(str,argv)),stdout=f,stderr=subprocess.STDOUT,check=True)
                    record={'seed':seed,'setting':setting,'kind':kind,'stage':stage,'seconds':time.perf_counter()-start}
                    with (OUT/'timings.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                    print('DONE',seed,setting,kind,stage,round(record['seconds'],1),flush=True)
                (run/'complete.json').write_text(json.dumps({'complete':True}))
    print('ALL 30 CONDITIONS COMPLETE',flush=True)

if __name__=='__main__':main()
