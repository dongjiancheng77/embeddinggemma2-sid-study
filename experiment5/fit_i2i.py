"""A/B/E train on unordered i2i; C/D remain fixed references. No GR inputs."""
import argparse
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from common import OUT,read,write,guard
from i2i import Cooccurrence

MODEL='C:/Users/Lenovo/.cache/huggingface/hub/models--google--embeddinggemma-2/snapshots/f9972ac22e696a105c482202646979e33e4b5a1e'

class Head(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        if kind=='linear':
            self.layer=nn.Linear(768,128)
            with torch.no_grad():self.layer.weight.zero_();self.layer.weight[:,:128]=torch.eye(128);self.layer.bias.zero_()
        else:
            m=torch.zeros(768,128);m[:128]=torch.eye(128)
            self.raw=nn.Parameter(m+torch.randn_like(m)*.0001)
    def forward(self,x):
        z=self.layer(x) if self.kind=='linear' else x@torch.linalg.qr(self.raw,mode='reduced')[0]
        return F.normalize(z,dim=-1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--kind',choices=['linear','ortho','mrl','pca','lora'],required=True)
    p.add_argument('--variant',choices=['all_cooccurrence','swing_top1_top5','swing_mutual_top1'],default='all_cooccurrence');p.add_argument('--updates',type=int,default=2000);args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(0);rng=np.random.default_rng(0)
    data=OUT/args.dataset;meta=read(data/'manifest.json');assert meta['data_verified']
    path=data/(args.kind if args.kind in ['mrl','pca'] else args.variant+'/'+args.kind);path.mkdir(parents=True,exist_ok=True)
    if (path/'embedding_stage1.json').exists():raise RuntimeError('Completed result exists')
    full=np.load(data/'base768.npy',mmap_mode='r');assert full.shape==(meta['counts']['items'],768)
    start=time.monotonic();losses=[];monitor=None
    if args.kind in ['mrl','pca']:
        if args.kind=='mrl':z=np.array(full[:,:128])
        else:
            from sklearn.decomposition import PCA
            pca=PCA(128,svd_solver='covariance_eigh',whiten=False).fit(full)
            z=pca.transform(full);np.savez(path/'pca.npz',mean=pca.mean_,components=pca.components_)
        z=z/np.linalg.norm(z,axis=1,keepdims=True).clip(1e-12);parameters=0
    else:
        from pairs import SwingPairs
        graph=Cooccurrence.from_prepared(data) if args.variant=='all_cooccurrence' else SwingPairs(data,args.variant,len(full))
        if args.kind=='lora':
            from sentence_transformers import SentenceTransformer
            from peft import LoraConfig,get_peft_model
            encoder=SentenceTransformer(MODEL,device='cpu',local_files_only=True,
                config_kwargs={'vision_config':None,'audio_config':None},model_kwargs={'torch_dtype':torch.bfloat16})
            encoder.max_seq_length=192;encoder[0].model.requires_grad_(False)
            encoder[0].model=get_peft_model(encoder[0].model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0,target_modules=['q_proj','v_proj'],bias='none'))
            model=encoder;catalog=read(data/'catalog.json')
            monitor=guard(path,layer_sleep=.003);torch.cuda.set_per_process_memory_fraction(.55);monitor.attach(model);model.to('cuda')
            def encode(ids):
                features=encoder.preprocess([catalog[i]['text'] for i in ids])
                features={k:v.to('cuda') if isinstance(v,torch.Tensor) else v for k,v in features.items()}
                return F.normalize(encoder(features)['sentence_embedding'].float()[:,:128],dim=-1)
        else:model=Head(args.kind);base=torch.from_numpy(np.array(full))
        params=[q for q in model.parameters() if q.requires_grad];parameters=sum(q.numel() for q in params)
        optim=torch.optim.AdamW(params,lr=1e-4 if args.kind=='lora' else 3e-4,weight_decay=1e-4)
        try:
            for step in range(args.updates):
                
                if (OUT/'STOP').exists():raise RuntimeError('User requested stop')
                ids=graph.sample(32,rng).flatten();optim.zero_grad(set_to_none=True)
                if args.kind=='lora':
                    # Exact gradient-cache chain rule, effective 32 unordered
                    # pairs / 64 endpoints, microbatch4 real token forwards.
                    with torch.no_grad():
                        chunks=[]
                        for j in range(0,len(ids),4):chunks.append(encode(ids[j:j+4]));monitor.pulse(.01)
                    leaf=torch.cat(chunks).detach().requires_grad_();loss=graph.loss(leaf,ids);loss.backward();g=leaf.grad.detach()
                    for j in range(0,len(ids),4):torch.autograd.backward(encode(ids[j:j+4]),g[j:j+4]);monitor.pulse(.01)
                else:loss=graph.loss(model(base[ids]),ids);loss.backward()
                assert torch.isfinite(loss);torch.nn.utils.clip_grad_norm_(params,1);optim.step();losses.append(float(loss.detach()))
                if step==0:
                    audit={'nonzero_trainable_gradients':sum(q.grad is not None and bool(q.grad.abs().sum()>0) for q in params),
                           'frozen_gradients':sum(q.grad is not None for q in model.parameters() if not q.requires_grad)}
                    assert audit['nonzero_trainable_gradients']>0 and audit['frozen_gradients']==0;write(path/'gradient_audit.json',audit)
                if (step+1)%20==0:write(path/'progress.json',{'step':step+1,'total':args.updates,'loss':losses[-1]});print(args.kind,step+1,losses[-1],flush=True)
                if args.kind=='lora' and (step+1)%100==0:encoder[0].model.save_pretrained(path/'adapter',save_embedding_layers=False)
            model.eval()
            with torch.inference_mode():
                if args.kind=='lora':
                    encoder[0].model.save_pretrained(path/'adapter',save_embedding_layers=False);parts=[]
                    for j in range(0,len(full),4):parts.append(encode(range(j,min(j+4,len(full)))).cpu().numpy());monitor.pulse(.01)
                    z=np.concatenate(parts)
                else:z=model(base).numpy();torch.save(model.state_dict(),path/'head.pt')
            if args.kind=='ortho':
                q=torch.linalg.qr(model.raw.detach(),mode='reduced')[0];assert (q.T@q-torch.eye(128)).abs().max()<1e-5
        finally:
            if monitor:monitor.close()
    assert z.shape==(len(full),128) and np.isfinite(z).all();np.save(path/'items128.npy',z.astype('float32'))
    write(path/'embedding_stage1.json',{'kind':args.kind,'variant':args.variant,'positive_rule':('shared TRAIN user count' if args.variant=='all_cooccurrence' else 'shared-user counts restricted to eligible Swing edges; other batch identities negative unless also eligible'),'parameters':parameters,'updates':len(losses),'losses':losses,
        'seconds':time.monotonic()-start,'supervision':('TRAIN-only uniform-user unordered pairs, shared-user-weighted multi-positive loss' if args.variant=='all_cooccurrence' else 'TRAIN-only uniform eligible unordered Swing edges, symmetric multi-positive contrastive loss'),
        'effective_pairs_per_batch':32,'temperature':.1,'history_text_inputs':False,'next_item_labels':False})

if __name__=='__main__':main()
