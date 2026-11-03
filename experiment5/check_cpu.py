"""Causality, cached decoding equivalence, pair-order invariance and losses."""
import numpy as np
import math
import torch
from common import OUT,write
from gr_model import BasicGR,collate,make_trie,retrieve,retrieve_batch,item_prompt
from i2i import Cooccurrence
from prepare import temporal_boundaries
from optimization import optimizer_for,lr_at

def main():
    torch.set_num_threads(4);torch.manual_seed(7)
    full=BasicGR();count=sum(p.numel() for p in full.parameters())
    assert count==27666944,count
    model=BasicGR(width=64,layers=2,heads=4,ff=128,dropout=0,input_scale=8.);model.eval()
    optimizer=optimizer_for(model)
    exempt={id(p) for group in optimizer.param_groups if group['weight_decay']==0 for p in group['params']}
    assert all(id(p) in exempt for name,p in model.named_parameters() if 'norm' in name or name=='embedding.weight')
    assert len({id(p) for group in optimizer.param_groups for p in group['params']})==len(list(model.parameters()))
    assert lr_at(0,10000)==.001 and lr_at(999,10000)==1 and abs(lr_at(10000,10000)-.1)<1e-8
    codes=np.array([[0,1,2],[0,1,3],[4,5,6],[7,8,9]])
    inputs,mask,positions,targets=collate([{'history':[0,2],'target':1},{'history':[],'target':3}],codes)
    loss=model.loss(inputs,mask,positions,targets);loss.backward()
    assert torch.isfinite(loss) and model.layers[0].qkv.weight.grad.norm()>0
    prompt=torch.tensor([item_prompt([0,2],codes)])
    with torch.no_grad():
        hidden,cache=model.hidden(prompt,torch.ones_like(prompt),use_cache=True)
        new=torch.tensor([[5]]);combined=torch.cat([prompt,new],1);key_mask=torch.ones_like(combined)
        cached,_=model.hidden(new,key_mask,past=cache,use_cache=True)
        uncached,_=model.hidden(combined,key_mask)
        assert torch.allclose(cached[:,-1],uncached[:,-1],atol=1e-5)
        altered=combined.clone();altered[:,-1]=6
        changed,_=model.hidden(altered,key_mask)
        assert torch.allclose(uncached[:,:-1],changed[:,:-1],atol=1e-5),'Future token leaked into history'
        short=torch.tensor([item_prompt([],codes)])
        solo,_=model.hidden(short,torch.ones_like(short))
        from gr_model import PAD
        padded=torch.cat([torch.full((1,5),PAD),short],1)
        pmask=torch.cat([torch.zeros(1,5,dtype=torch.long),torch.ones_like(short)],1)
        ph,_=model.hidden(padded,pmask)
        assert torch.allclose(solo[:,-1],ph[:,-1],atol=1e-5),'Padding changed prediction'
        trie,leaves=make_trie(codes);ranked=retrieve(model,[0],codes,trie,leaves,beam=100)
        assert set(ranked)==set(range(4)) and len(ranked)==4
        batch_ranked=retrieve_batch(model,[[0],[]],codes,trie,leaves)
        assert batch_ranked==[ranked,retrieve(model,[],codes,trie,leaves)],'Batched beam changed item ranking'
    a=Cooccurrence([[0,1,2],[2,0],[3,4]],6)
    b=Cooccurrence([[2,1,0],[0,2],[4,3]],6)
    ids=np.array([0,1,2,2,3,4]);assert np.array_equal(a.weights(ids),b.weights(ids))
    assert (a.matrix.T@a.matrix)[0,2]==2 and (a.matrix.T@a.matrix)[0,3]==0
    assert a.matrix[:,5].nnz==0,'Held-out-only item entered training graph'
    z=torch.randn(6,128,requires_grad=True);contrastive=a.loss(z,ids);contrastive.backward()
    assert torch.isfinite(contrastive) and z.grad.norm()>0
    samples=a.sample(1000,np.random.default_rng(2));assert any(set(pair)=={0,2} for pair in samples)
    assert np.array_equal(samples,b.sample(1000,np.random.default_rng(2)))
    split=temporal_boundaries(np.repeat(np.arange(100),2));assert split['train']==[0,160] and split['valid']==[160,180]
    report={'paper_sized_parameters':count,'causal_attention':'passed','KV_cache_equivalence':'passed',
        'left_padding_invariance':'passed','beam_validity':'passed','unordered_i2i_invariance':'passed',
        'nonadjacent_pairs_included':'passed','multi_positive_gradient':'passed','heldout_only_item_excluded':'passed',
        'global_time_split':'passed','profile_only':True}
    write(OUT/'cpu_checks.json',report);print(report)

if __name__=='__main__':main()
