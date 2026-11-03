"""Synthetic compatibility probe only; not a paper dataset experiment."""
import time
import argparse
import numpy as np
import torch
from common import OUT,guard,write
from gr_model import BasicGR,collate,make_trie,retrieve_batch

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,default=32);p.add_argument('--steps',type=int,default=32);args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(0);rng=np.random.default_rng(7)
    model=BasicGR(input_scale=512**.5);path=OUT/f'compatibility_profile_b{args.batch}';monitor=guard(path,layer_sleep=.0015)
    try:
        torch.cuda.set_per_process_memory_fraction(.55);monitor.attach(model);model.to('cuda')
        codes=rng.integers(0,256,size=(1000,3));trie,leaves=make_trie(codes)
        events=[{'history':rng.integers(0,1000,size=100).tolist(),'target':int(rng.integers(1000))} for _ in range(args.batch)]
        batch=collate(events,codes,device='cuda');optim=torch.optim.AdamW(model.parameters(),lr=.001)
        losses=[];start=time.monotonic()
        for step in range(args.steps):
            optim.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.loss(*batch)
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optim.step()
            losses.append(float(loss.detach()));monitor.pulse(.005)
        model.eval()
        with torch.autocast('cuda',dtype=torch.bfloat16):ranked=retrieve_batch(model,[events[0]['history'],events[1]['history']],codes,trie,leaves,beam=100,device='cuda')
        assert all(len(r)==100 and len(set(r))==100 for r in ranked)
        write(path/'complete.json',{'profile_only':True,'synthetic_item_ids':True,'paper_data_training':False,
            'parameters':sum(p.numel() for p in model.parameters()),'training_steps':args.steps,'batch':args.batch,'history_items':100,
            'losses':losses,'beam':100,'unique_retrieved_items_per_query':len(ranked[0]),'seconds':time.monotonic()-start,
            'peak_gpu_utilization':monitor.peak,'peak_torch_GiB':torch.cuda.max_memory_allocated()/1024**3})
        print('PASSED compatibility profile; no scientific recommendation metrics',flush=True)
    finally:monitor.close()

if __name__=='__main__':main()
