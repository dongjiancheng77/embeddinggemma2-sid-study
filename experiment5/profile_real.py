"""Guarded speed-only probe on real histories, synthetic SID labels; no metrics."""
import time
import numpy as np
import torch
from common import OUT,guard,write
from prepare import Events
from batching import grouped_order
from gr_model import BasicGR,collate
from optimization import optimizer_for

def main():
    torch.set_num_threads(4);torch.manual_seed(0);root=OUT/'Musical_Instruments';events=Events(root,'train')
    n=events.manifest['counts']['items'];codes=np.random.default_rng(0).integers(0,256,(n,3))
    profiles={}
    for name,order,layer_sleep,step_sleep in [
        ('old_random',np.random.default_rng(0).permutation(len(events)),.002,.02),
        ('new_grouped',grouped_order(events,0,32),.00075,.002)]:
        monitor=guard(root/('speed_'+name),layer_sleep=layer_sleep);model=BasicGR(input_scale=512**.5)
        try:
            torch.cuda.set_per_process_memory_fraction(.55);monitor.attach(model);model.to('cuda');optim=optimizer_for(model)
            batches=np.random.default_rng(12).choice(len(events)//32,68,replace=False)
            start=None
            for j,b in enumerate(batches):
                if j==4:torch.cuda.synchronize();start=time.monotonic()
                ids=order[b*32:(b+1)*32];batch=collate([events[int(i)] for i in ids],codes,device='cuda')
                optim.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.loss(*batch)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optim.step();monitor.pulse(step_sleep)
            torch.cuda.synchronize();elapsed=time.monotonic()-start
            profiles[name]={'steps':64,'seconds':elapsed,'events_per_second':64*32/elapsed,
                'train_epoch_minutes_estimate':len(events)/(64*32/elapsed)/60,'peak_gpu':monitor.peak}
        finally:
            monitor.close();del model,optim;torch.cuda.empty_cache()
    profiles['speedup']=profiles['new_grouped']['events_per_second']/profiles['old_random']['events_per_second']
    profiles['speed_only']=True;profiles['synthetic_sid_labels']=True;profiles['includes_validation_or_test']=False
    write(root/'real_history_speed.json',profiles);print(profiles,flush=True)

if __name__=='__main__':main()
