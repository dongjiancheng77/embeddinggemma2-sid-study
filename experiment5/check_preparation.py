"""Small synthetic integration checks, never used as scientific results."""
import csv
import sys
import numpy as np
import torch
from common import OUT,write,read,save_checkpoint
import prepare
from i2i import Cooccurrence
from gr_model import BasicGR,collate
from optimization import optimizer_for

def main():
    path=OUT/'compatibility_data'
    path.mkdir(parents=True,exist_ok=True)
    if (path/'Beauty/manifest.json').exists():raise RuntimeError('Compatibility fixture already exists; inspect rather than overwrite')
    source=path/'source.json';events=path/'input.csv'
    write(source,{'release':'Synthetic compatibility fixture only','source_url':'local:test fixture','paper_data':False})
    with events.open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['user_id','item_id','timestamp'])
        for j in range(200):writer.writerow([f'u{j%10}',f'i{j%30}' if j<180 else 'heldout_only',j//2])
    prepare.OUT=path
    sys.argv=['prepare.py','--dataset','Beauty','--events',str(events),'--source-manifest',str(source),'--allow-stats-mismatch']
    prepare.main();meta=read(path/'Beauty/manifest.json')
    assert meta['data_verified'] and not meta['paper_stats_match']
    assert meta['splits']['train']==[0,160] and meta['splits']['test']==[180,200]
    test=prepare.Events(path/'Beauty','test');all_events=np.load(path/'Beauty/events.npy')
    for j in range(len(test)):
        event=test[j];before=all_events[:180+j];expected=before[before[:,0]==event['user'],1].tolist()
        assert event['history']==expected[-100:]
    graph=Cooccurrence.from_prepared(path/'Beauty');item=read(path/'Beauty/item_map.json')['heldout_only']
    assert graph.matrix[:,item].nnz==0
    torch.set_num_threads(4);torch.manual_seed(10)
    model=BasicGR(width=64,layers=2,heads=4,ff=128,dropout=.1,input_scale=8.)
    optim=optimizer_for(model)
    codes=np.array([[0,1,2],[3,4,5],[6,7,8]])
    batch=collate([{'history':[0],'target':1},{'history':[1],'target':2}],codes)
    def step(model,optim):
        optim.zero_grad();model.loss(*batch).backward();optim.step()
    step(model,optim)
    save_checkpoint(path/'checkpoint.pt',{'model':model.state_dict(),'optimizer':optim.state_dict(),'rng':torch.get_rng_state()})
    step(model,optim);expected={k:v.detach().clone() for k,v in model.state_dict().items()}
    restored=BasicGR(width=64,layers=2,heads=4,ff=128,dropout=.1,input_scale=8.)
    optimizer=optimizer_for(restored)
    checkpoint=torch.load(path/'checkpoint.pt',weights_only=False);restored.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer']);torch.set_rng_state(checkpoint['rng']);step(restored,optimizer)
    assert all(torch.allclose(v,expected[k],atol=1e-7,rtol=0) for k,v in restored.state_dict().items())
    write(path/'checks.json',{'synthetic_fixture':True,'full_prepare_and_history_index':'passed',
        'stats_deviation_preserved':'passed','i2i_heldout_exclusion':'passed','optimizer_and_dropout_rng_resume':'passed'})
    print('PASS: canonical CSV -> global split -> histories -> train-only i2i; optimizer/RNG resume')

if __name__=='__main__':main()
