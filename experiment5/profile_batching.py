"""Padding operation-count proxy, not a measured GPU speedup."""
import numpy as np
from common import OUT,write
from prepare import Events
from batching import grouped_order

def main():
    root=OUT/'Musical_Instruments';events=Events(root,'train');batch=32
    position=np.asarray(events.positions[:events.end]);users=events.events[:events.end,0]
    length=3*np.minimum(position-events.offsets[users],100)+4
    old=np.random.default_rng(0).permutation(len(events));new=grouped_order(events,0,batch)
    def cost(order):
        chunks=np.array_split(np.asarray(order),np.arange(batch,len(order),batch))
        return {'padded_tokens':sum(len(c)*int(length[c].max()) for c in chunks),'attention_proxy':sum(len(c)*int(length[c].max())**2 for c in chunks)}
    a=cost(old);b=cost(new)
    write(root/'batching_profile.json',{'old_random':a,'new_local_grouped':b,'token_reduction':1-b['padded_tokens']/a['padded_tokens'],
        'attention_proxy_reduction':1-b['attention_proxy']/a['attention_proxy'],'all_events_once':sorted(new)==list(range(len(events))),'measured_gpu_speedup':False})

if __name__=='__main__':main()
