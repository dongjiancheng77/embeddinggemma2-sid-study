"""Common query/document contrastive objective for cached heads and real token-forward LoRA."""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from mini_tiger import Head,set_seed
from experiment2_data import ROOT,OUT,MODEL,read

class TokenLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        from peft import LoraConfig,get_peft_model
        self.encoder=SentenceTransformer(MODEL,device='cuda',local_files_only=True,
            config_kwargs={'vision_config':None,'audio_config':None},model_kwargs={'torch_dtype':torch.bfloat16})
        self.encoder.max_seq_length=192
        base=self.encoder[0].model
        base.requires_grad_(False)
        self.encoder[0].model=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=0,
            target_modules=['q_proj','v_proj'],bias='none'))
        # Gradient cache below recomputes one microbatch at a time, without
        # retaining the whole contrastive batch's Transformer activations.

    def forward(self,texts):
        chunks=[]
        for start in range(0,len(texts),32):
            features=self.encoder.preprocess(texts[start:start+32])
            features={k:v.to('cuda') if isinstance(v,torch.Tensor) else v for k,v in features.items()}
            chunks.append(F.normalize(self.encoder(features)['sentence_embedding'].float(),dim=-1))
        return torch.cat(chunks)

def batch_loss(z,rows,task):
    b=len(rows)
    q,p,n=z[:b],z[b:2*b],z[2*b:]
    cand=torch.cat([p,n])
    logits=q@cand.T/.1
    cand_ids=torch.tensor([r['p'] for r in rows]+[r['n'] for r in rows],device=z.device)
    pos_ids=torch.tensor([r['p'] for r in rows],device=z.device)
    invalid=cand_ids[None,:].eq(pos_ids[:,None])
    invalid[torch.arange(b,device=z.device),torch.arange(b,device=z.device)]=False
    if task=='rec':
        for i,r in enumerate(rows):
            invalid[i]|=torch.isin(cand_ids,torch.tensor(r['history'],device=z.device))
    logits=logits.masked_fill(invalid,float('-inf'))
    return F.cross_entropy(logits,torch.arange(b,device=z.device))

@torch.no_grad()
def diagnostics(z,d):
    z=torch.from_numpy(z).to('cuda')
    result={}
    for task in ['rec','text']:
        rows=d['examples'][task+'_validation']
        if task=='rec':candidates=list(range(d['n_items']))
        else:candidates=sorted({r['p'] for r in rows}|{r['n'] for r in rows})
        candidate=torch.tensor(candidates,device='cuda')
        hit=0.;ndcg=0.
        for start in range(0,len(rows),64):
            batch=rows[start:start+64]
            scores=z[[r['q'] for r in batch]]@z[candidate].T
            if task=='rec':
                for i,r in enumerate(batch):scores[i,torch.isin(candidate,torch.tensor(r['history'],device='cuda'))]=float('-inf')
            pred=candidate[scores.topk(10,dim=-1).indices]
            mask=pred.eq(torch.tensor([r['p'] for r in batch],device='cuda')[:,None])
            anyhit=mask.any(1)
            hit+=anyhit.sum().item()
            ndcg+=(1/torch.log2(mask.float().argmax(1)[anyhit]+2)).sum().item()
        result[task+'_validation_recall10']=hit/len(rows)
        result[task+'_validation_ndcg10']=ndcg/len(rows)
        result[task+'_validation_candidates']=len(candidates)
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--kind',choices=['linear','ortho','mrl','pca','lora'],required=True)
    p.add_argument('--setting',choices=['collab','mixed'],required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--profile',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(4);set_seed(args.seed)
    d=read(OUT/'dataset.json')
    path=OUT/args.setting/f'seed{args.seed}'/args.kind
    path.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    losses=[];gradient_audit={}
    if args.kind in ['mrl','pca']:
        full=np.load(OUT/'base768.npy')
        if args.kind=='mrl':z=full[:,:128]
        else:
            from sklearn.decomposition import PCA
            pca=PCA(n_components=128,svd_solver='full',whiten=False)
            pca.fit(full[:d['n_items']])
            z=pca.transform(full)
            np.savez(path/'pca.npz',mean=pca.mean_,components=pca.components_,explained_variance_ratio=pca.explained_variance_ratio_)
        z=z/np.linalg.norm(z,axis=1,keepdims=True).clip(1e-12)
        parameter_count=0
    else:
        if args.kind=='lora':
            model=TokenLoRA()
            base=None
        else:
            model=Head(args.kind).to('cuda')
            base=torch.from_numpy(np.load(OUT/'base768.npy')).to('cuda')
        set_seed(args.seed)
        trainable=[p for p in model.parameters() if p.requires_grad]
        parameter_count=sum(p.numel() for p in trainable)
        optim=torch.optim.AdamW(trainable,lr=1e-4 if args.kind=='lora' else 3e-4,weight_decay=1e-4)
        tasks=['rec','text'] if args.setting=='mixed' else ['rec']
        rng=random.Random(args.seed)
        batches={}
        for task in tasks:
            examples=list(d['examples'][task+'_train']);rng.shuffle(examples)
            batches[task]=[examples[j:j+32] for j in range(0,len(examples),32)]
        stream=[(task,batches[task][j]) for j in range(len(batches['rec'])) for task in tasks]
        for step,(task,rows) in enumerate(stream):
            model.train()
            indices=[r['q'] for r in rows]+[r['p'] for r in rows]+[r['n'] for r in rows]
            optim.zero_grad()
            if args.kind=='lora':
                texts=[d['texts'][ix] for ix in indices]
                with torch.no_grad():z0=F.normalize(model(texts)[:,:128],dim=-1)
                leaf=z0.detach().requires_grad_(True)
                loss=batch_loss(leaf,rows,task)
                assert torch.isfinite(loss),float(loss)
                loss.backward()
                grad=leaf.grad.detach()
                for j in range(0,len(texts),32):
                    replay=F.normalize(model(texts[j:j+32])[:,:128],dim=-1)
                    torch.autograd.backward(replay,grad[j:j+32])
            else:
                z0=model(base[indices])
                loss=batch_loss(z0,rows,task)
                assert torch.isfinite(loss),float(loss)
                loss.backward()
            if step==0:
                gradient_audit={'nonzero_trainable_gradient_tensors':sum(p.grad is not None and p.grad.abs().sum().item()>0 for p in trainable),
                                'trainable_parameter_tensors':len(trainable),
                                'unexpected_frozen_gradients':sum(p.grad is not None for p in model.parameters() if not p.requires_grad)}
                assert gradient_audit['nonzero_trainable_gradient_tensors']>0 and gradient_audit['unexpected_frozen_gradients']==0
            torch.nn.utils.clip_grad_norm_(trainable,1)
            optim.step()
            losses.append({'step':step+1,'task':task,'loss':float(loss.item())})
            if (step+1)%16==0 or args.profile:
                print(f'{args.kind} {args.setting} {step+1}/{len(stream)} loss={loss.item():.4f} seconds={time.perf_counter()-start:.1f} peakGB={torch.cuda.max_memory_allocated()/1e9:.2f}',flush=True)
            if args.profile and step==3:
                (path/'profile.json').write_text(json.dumps({'seconds':time.perf_counter()-start,'peak_bytes':torch.cuda.max_memory_allocated(),'gradient_audit':gradient_audit,'trainable_parameters':parameter_count},indent=2))
                return
        torch.cuda.synchronize()
        train_seconds=time.perf_counter()-start
        if args.kind=='lora':model.encoder[0].model.save_pretrained(path/'adapter')
        else:torch.save(model.state_dict(),path/'head.pt')
        model.eval();out=[]
        export_ids=sorted(set(range(d['n_items']))|{r[k] for task in ['rec_validation','text_validation'] for r in d['examples'][task] for k in ['q','p','n']}) if args.kind=='lora' else list(range(len(d['texts'])))
        with torch.no_grad():
            for j in range(0,len(export_ids),128 if args.kind!='lora' else 32):
                if args.kind=='lora':
                    value=F.normalize(model([d['texts'][ix] for ix in export_ids[j:j+32]])[:,:128],dim=-1)
                else:value=model(base[export_ids[j:j+128]])
                out.append(value.float().cpu().numpy())
                if args.kind=='lora' and j%4096==0:print('EXPORT',j,'/',len(export_ids),flush=True)
        z=np.zeros((len(d['texts']),128),dtype='float32')
        z[export_ids]=np.concatenate(out)
    assert z.shape==(len(d['texts']),128) and np.isfinite(z).all()
    np.save(path/'items128.npy',z[:d['n_items']].astype('float32'))
    stats={'kind':args.kind,'setting':args.setting,'seed':args.seed,'parameters':parameter_count,
           'train_seconds':locals().get('train_seconds',0),'total_seconds':time.perf_counter()-start,
           'peak_bytes':torch.cuda.max_memory_allocated(),'gradient_audit':gradient_audit,
           **diagnostics(z,d)}
    if args.kind=='ortho':
        q,_=torch.linalg.qr(model.raw.float());stats['orthogonality_max_error']=float((q.T@q-torch.eye(128,device='cuda')).abs().max().item())
    (path/'stage1.json').write_text(json.dumps(stats,indent=2))
    (path/'training.json').write_text(json.dumps(losses,indent=2))
    print('FINISHED',json.dumps(stats),flush=True)

if __name__=='__main__':main()
