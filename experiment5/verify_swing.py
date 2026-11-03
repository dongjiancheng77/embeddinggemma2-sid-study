"""Independently replay sampled Swing scores from common-user sets."""
import itertools
import numpy as np
from scipy import sparse
from common import OUT,read,write
from i2i import Cooccurrence
from pairs import SwingPairs
from swing import VARIANTS

def main():
    root=OUT/'Musical_Instruments';graph=Cooccurrence.from_prepared(root);m=graph.matrix;c=m.tocsc()
    scores=sparse.load_npz(root/'swing/scores.npz');coo=scores.tocoo()
    positions=np.random.default_rng(3).choice(len(coo.data),32,replace=False);worst=0
    for index in positions:
        a,b=int(coo.row[index]),int(coo.col[index])
        users=np.intersect1d(c.indices[c.indptr[a]:c.indptr[a+1]],c.indices[c.indptr[b]:c.indptr[b+1]])
        reference=0.
        for u,v in itertools.combinations(users,2):
            iu=m.indices[m.indptr[u]:m.indptr[u+1]];iv=m.indices[m.indptr[v]:m.indptr[v+1]]
            reference+=1/np.sqrt(len(iu)*len(iv))/(1+len(np.intersect1d(iu,iv)))
        error=abs(reference-scores[a,b]);worst=max(worst,error);assert error<1e-10
    top=np.load(root/'swing/top5.npy');stats={}
    for v in VARIANTS[1:]:
        g=SwingPairs(root,v,m.shape[1]);rng=np.random.default_rng(0)
        for a,b in g.pairs:
            if v=='swing_mutual_top1':assert top[a,0]==b and top[b,0]==a
            else:assert (top[a,0]==b and a in top[b]) or (top[b,0]==a and b in top[a])
        ids=g.sample(32,rng).flatten();w=g.weights(ids)
        assert (w.sum(1)>0).all() and (w[np.equal.outer(ids,ids)]==0).all()
        stats[v]={'edges':len(g.pairs),'positive_weight_min':float(g.positive.data.min()),'rank_check':'passed'}
    write(root/'swing/verified.json',{'sampled_scores':32,'max_absolute_error':worst,'variants':stats,'only_train':True})
    print('PASS: independent exact score replay, all reciprocal ranks, shared-user weights, batch positives')

if __name__=='__main__':main()
