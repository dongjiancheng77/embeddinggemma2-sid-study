"""Validation-only conditioning checks; never read test targets."""
import numpy as np
import torch
import torch.nn.functional as F
from gr_model import collate

def frequency_loss(events,codes,counts,indices):
    frequency=np.stack([np.bincount(codes[:,j],weights=counts,minlength=256) for j in range(3)])
    targets=np.array([codes[events[int(i)]['target']] for i in indices])
    return float(np.mean([-np.log((frequency[j,targets[:,j]]+.1)/(frequency[j].sum()+25.6)) for j in range(3)]))

@torch.inference_mode()
def probe(model,events,codes,monitor,indices,marginal):
    training=model.training;model.eval();losses={key:[] for key in ['history','other_history','wrong_prefix']};first={key:[] for key in losses}
    try:
        for start in range(0,len(indices),32):
            rows=[events[int(i)] for i in indices[start:start+32]]
            for key in losses:
                selected=rows if key!='other_history' else [{**r,'history':rows[(j+1)%len(rows)]['history']} for j,r in enumerate(rows)]
                ids,mask,pos,target=collate(selected,codes,device='cuda')
                if key=='wrong_prefix':
                    for j in range(2):ids[torch.arange(len(rows),device='cuda'),pos[:,j]+1]=torch.roll(target[:,j],1)+256*j
                with torch.autocast('cuda',dtype=torch.bfloat16):h,_=model.hidden(ids,mask)
                ar=torch.arange(len(rows),device='cuda');level_loss=[]
                for level in range(3):
                    value=F.cross_entropy(model.logits(h[ar,pos[:,level]],level),target[:,level],reduction='none')
                    level_loss.append(value);first[key].extend(value.tolist() if level==0 else [])
                losses[key].extend(torch.stack(level_loss).mean(0).tolist());monitor.pulse(.02)
    finally:model.train(training)
    mean={k:float(np.mean(v)) for k,v in losses.items()};first_mean={k:float(np.mean(v)) for k,v in first.items()}
    report={'examples':len(indices),'loss':mean,'first_level_loss':first_mean,'marginal_frequency_loss':marginal,
        'prefix_penalty':mean['wrong_prefix']-mean['history'],
        'history_penalty':mean['other_history']-mean['history'],
        'first_level_history_penalty':first_mean['other_history']-first_mean['history']}
    report['passed']=bool(np.isfinite(mean['history']) and mean['history']<marginal-.05
        and report['prefix_penalty']>.05 and report['history_penalty']>.01)
    return report
