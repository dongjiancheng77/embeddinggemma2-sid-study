"""Full chronological GR training with ordinary NTP and validation selection."""
import argparse
import functools
import json
import math
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from common import OUT,read,write,guard,save_checkpoint
from prepare import Events
from gr_model import BasicGR,collate,make_trie,retrieve_batch
from batching import grouped_order
from optimization import RECIPE,optimizer_for,set_lr
from quality import probe,frequency_loss
from common import sha

@torch.inference_mode()
def evaluate(model,events,codes,device,monitor,path,train_counts,train_paths):
    trie,leaves=make_trie(codes);total=0;hits=0;groups={'cold':[0,0],'zsp':[0,0],'hot':[0,0]}
    model.eval()
    with path.open('w',encoding='utf-8',buffering=1) as log:
        for j in range(0,len(events),2):
            rows=[events[k] for k in range(j,min(j+2,len(events)))]
            with torch.autocast('cuda',dtype=torch.bfloat16):preds=retrieve_batch(model,[r['history'] for r in rows],codes,trie,leaves,beam=100,device=device)
            for index,(row,pred) in enumerate(zip(rows,preds)):
                hit=row['target'] in pred;total+=1;hits+=hit
                cold=train_counts[row['target']]==0;zsp=tuple(codes[row['target']]) not in train_paths
                for label,condition in [('cold',cold),('zsp',zsp),('hot',not cold)]:
                    if condition:groups[label][0]+=int(hit);groups[label][1]+=1
                log.write(json.dumps({'event_index':j+index,'user':row['user'],'target':row['target'],'hit100':hit,'cold':bool(cold),'zsp':zsp})+'\n')
            if j%100==0:write(path.with_suffix('.progress.json'),{'done':total,'total':len(events),'hits':hits})
            monitor.pulse(.002)
    return {'recall@100':hits/total,'events':total,'hits':hits,
        **{key:{'hits':v[0],'events':v[1],'recall@100':v[0]/v[1] if v[1] else None} for key,v in groups.items()},
        'hot_definition':'at least one training interaction; paper does not specify a higher sufficient-support threshold'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--kind',required=True)
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--patience',type=int,default=2)
    p.add_argument('--batch',type=int,default=32);p.add_argument('--lr',type=float,default=.0003)
    p.add_argument('--train-only',action='store_true');p.add_argument('--eval-only',action='store_true');p.add_argument('--preflight-only',action='store_true');p.add_argument('--preflight-steps',type=int,default=3000)
    args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
    root=OUT/args.dataset;meta=read(root/'manifest.json');assert meta['data_verified']
    codes=np.load(root/args.kind/'sid.npy');assert codes.shape==(meta['counts']['items'],3)
    path=root/args.kind/'gr';path.mkdir(parents=True,exist_ok=True)
    if (path/'complete.json').exists():raise RuntimeError('Completed experiment is protected')
    if (path/'configuration.json').exists() and read(path/'configuration.json').get('recipe')!=RECIPE:
        raise RuntimeError('Old experiment/configuration is protected. Use a new repair workspace, not old checkpoints.')
    train=Events(root,'train');valid=Events(root,'valid');test=Events(root,'test')
    counts=np.bincount(train.events[:train.end,1],minlength=len(codes))
    supported=set(map(tuple,codes[np.flatnonzero(counts)]))
    model=BasicGR(input_scale=math.sqrt(512));optim=None
    steps_per_epoch=math.ceil(len(train)/args.batch);total_steps=args.epochs*steps_per_epoch
    probe_indices=np.random.default_rng(42).choice(len(valid),size=min(1024,len(valid)),replace=False)
    marginal=frequency_loss(valid,codes,counts,probe_indices);quality_failures=0
    sid_hash=sha(root/args.kind/'sid.npy')
    write(path/('evaluation_configuration.json' if args.eval_only else 'configuration.json'),{'arguments':vars(args),'parameters':sum(p.numel() for p in model.parameters()),
        'architecture_confirmed_by_pdf':read(__import__('pathlib').Path(__file__).parent/'protocol.json')['paper_confirmed'],
        'recipe':RECIPE,'sid_sha256':sid_hash,'input_embedding_scale':math.sqrt(512),
        'optimizer':'AdamW betas=(.9,.98), no weight decay on normalization or tied embeddings',
        'schedule':'1000 update linear warmup, cosine to 0.1x LR; same effective batch32','batching':'seeded local length grouping, window2048, each event once per epoch',
        'training_hyperparameters':'Repaired independent recipe, not author-provided; all A-E share it',
        'train_events':len(train),'validation_events':len(valid),'test_events':len(test),'selection':'validation mean NTP loss; no test selection',
        'embedding_loss':'TRAIN-only unordered i2i upstream; GR is chronological NTP, no SPROUT losses'})
    monitor=guard(path,layer_sleep=.00075);start=time.monotonic();best=float('inf');bad=0
    try:
        torch.cuda.set_per_process_memory_fraction(.55);monitor.attach(model);model.to('cuda')
        optim=optimizer_for(model,args.lr)
        epoch_start=0;resume_step=0;checkpoint=path/'latest.pt'
        if checkpoint.exists():
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if saved.get('recipe')!=RECIPE or saved.get('sid_sha256')!=sid_hash:raise RuntimeError('Checkpoint recipe/SID mismatch')
            model.load_state_dict(saved['model']);optim.load_state_dict(saved['optimizer'])
            epoch_start=saved['epoch'];resume_step=saved['next_step'];best=saved['best'];bad=saved['bad']
            torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
            print('RESUME',epoch_start,resume_step,flush=True)
        last_epoch=epoch_start
        for epoch in range(epoch_start,args.epochs if bad<args.patience and not args.eval_only else epoch_start):
            last_epoch=epoch+1
            order=grouped_order(train,epoch,args.batch)
            step_start=resume_step if epoch==epoch_start else 0
            loader=DataLoader(train,batch_size=args.batch,sampler=order[step_start*args.batch:],
                collate_fn=functools.partial(collate,codes=codes),num_workers=0,generator=torch.Generator().manual_seed(10000+epoch))
            def checkpoint_at(next_epoch,next_step):
                save_checkpoint(checkpoint,{'model':model.state_dict(),'optimizer':optim.state_dict(),
                    'recipe':RECIPE,'sid_sha256':sid_hash,
                    'epoch':next_epoch,'next_step':next_step,'best':best,'bad':bad,
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()})
            model.train();losses=[]
            for step,batch in enumerate(loader,start=step_start):
                monitor.wait();batch=[x.to('cuda') for x in batch];optim.zero_grad(set_to_none=True)
                global_step=epoch*steps_per_epoch+step
                learning_rate=set_lr(optim,args.lr,global_step,total_steps)
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.loss(*batch)
                assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optim.step();losses.append(float(loss.detach()))
                if step%100==0:write(path/'progress.json',{'state':'training','epoch':epoch+1,'step':step+1,'total_steps':steps_per_epoch,'loss':losses[-1],'lr':learning_rate});print(args.dataset,args.kind,epoch+1,step+1,losses[-1],learning_rate,flush=True)
                if (step+1)%1000==0:checkpoint_at(epoch,step+1)
                if (args.preflight_only and (global_step+1)%1000==0) or (not args.preflight_only and (global_step+1)%5000==0):
                    audit=probe(model,valid,codes,monitor,probe_indices,marginal)
                    audit.update(global_step=global_step+1,recipe=RECIPE,peak_gpu_utilization=monitor.peak)
                    write(path/f'probe_{global_step+1}.json',audit);print('QUALITY_PROBE',audit,flush=True)
                    if not args.preflight_only:
                        quality_failures=0 if audit['passed'] else quality_failures+1
                        if quality_failures>=2:
                            checkpoint_at(epoch,step+1);write(OUT/'QUALITY_STOP.json',audit)
                            raise RuntimeError('Repeated validation quality gate failure; stopping instead of continuing blind training')
                if args.preflight_only and global_step+1>=args.preflight_steps:
                    checkpoint_at(epoch,step+1)
                    audit=probe(model,valid,codes,monitor,probe_indices,marginal)
                    audit.update(global_step=global_step+1,recipe=RECIPE,peak_gpu_utilization=monitor.peak)
                    write(path/'preflight.json',audit)
                    if not audit['passed']:raise RuntimeError('Repair preflight quality check failed')
                    write(path/'preflight_passed.json',audit);write(path/'progress.json',{'state':'preflight_passed','steps':global_step+1})
                    return
                monitor.pulse(.002)
            checkpoint_at(epoch,math.ceil(len(train)/args.batch))
            write(path/'progress.json',{'state':'validation','epoch':epoch+1,'training_events':len(train)})
            model.eval();vloss=0.;number=0
            with torch.inference_mode():
                for batch in DataLoader(valid,batch_size=args.batch,collate_fn=functools.partial(collate,codes=codes)):
                    batch=[x.to('cuda') for x in batch]
                    with torch.autocast('cuda',dtype=torch.bfloat16):value=model.loss(*batch)
                    vloss+=float(value)*len(batch[0]);number+=len(batch[0]);monitor.pulse(.002)
            vloss/=number
            write(path/f'epoch{epoch+1}.json',{'train_loss':float(np.mean(losses)) if losses else None,'validation_loss':vloss})
            if vloss<best:best=vloss;bad=0;torch.save(model.state_dict(),path/'best.pt')
            else:bad+=1
            checkpoint_at(epoch+1,0)
            if bad>=args.patience:break
        if args.train_only:
            write(path/'training_complete.json',{'epochs':last_epoch,'best_validation_loss':best,'seconds':time.monotonic()-start})
            write(path/'progress.json',{'state':'trained_pending_test','epochs':last_epoch});return
        model.load_state_dict(torch.load(path/'best.pt',map_location='cuda',weights_only=True))
        write(path/'progress.json',{'state':'test','events':len(test),'beam':100})
        metrics=evaluate(model,test,codes,'cuda',monitor,path/'test_events.jsonl',counts,supported)
        write(path/'metrics.json',metrics);write(path/'complete.json',{'seconds':time.monotonic()-start,'epochs':last_epoch,'metrics':metrics,
            'best_validation_loss':best,'peak_gpu_utilization':monitor.peak,'peak_torch_bytes':torch.cuda.max_memory_allocated()})
        write(path/'progress.json',{'state':'complete','epochs':last_epoch,'metrics':metrics})
    finally:monitor.close()

if __name__=='__main__':main()
