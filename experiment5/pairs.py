"""Filtered symmetric multi-positive contrastive supervision."""
import numpy as np
from scipy import sparse
from i2i import Cooccurrence

class SwingPairs(Cooccurrence):
    def __init__(self,root,variant,n_items):
        self.pairs=np.load(root/'swing'/(variant+'.npy'))
        a,b=self.pairs.T
        counts=np.ones(len(a),dtype=np.float32)
        if (root/'manifest.json').exists():
            base=Cooccurrence.from_prepared(root)._columns
            for begin in range(0,len(a),1024):
                counts[begin:begin+1024]=np.asarray(base[:,a[begin:begin+1024]].multiply(base[:,b[begin:begin+1024]]).sum(axis=0)).ravel()
            assert counts.min()>=2
        self.positive=sparse.coo_matrix((np.r_[counts,counts],(np.r_[a,b],np.r_[b,a])),shape=(n_items,n_items)).tocsr()
    def sample(self,batch,rng):
        return self.pairs[rng.integers(len(self.pairs),size=batch)]
    def weights(self,ids):
        w=self.positive[ids][:,ids].toarray().astype(np.float32)
        w[np.equal.outer(ids,ids)]=0
        return w
