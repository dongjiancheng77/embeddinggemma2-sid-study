"""Seeded local length grouping; every event appears once per GR epoch."""
import numpy as np

def grouped_order(events,epoch,batch=32,window=2048):
    rng=np.random.default_rng(1000+epoch);order=rng.permutation(len(events))
    positions=np.asarray(events.positions[events.begin:events.end])
    users=events.events[events.begin:events.end,0]
    lengths=np.minimum(positions-events.offsets[users],events.max_history)
    full=[];tail=[]
    for begin in range(0,len(order),window):
        chunk=order[begin:begin+window]
        chunk=chunk[np.argsort(lengths[chunk],kind='stable')]
        end=len(chunk)//batch*batch
        full.extend(chunk[j:j+batch] for j in range(0,end,batch))
        tail.extend(chunk[end:])
    rng.shuffle(full)
    return np.concatenate(full+[np.array(tail,dtype=np.int64)]).tolist()
