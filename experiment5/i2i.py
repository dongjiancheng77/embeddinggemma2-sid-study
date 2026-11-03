"""Order-independent user/item co-occurrence, represented without O(n^2) storage."""
import numpy as np
from scipy import sparse
import torch
import torch.nn.functional as F

class Cooccurrence:
    @classmethod
    def from_prepared(cls,path):
        from pathlib import Path
        import json
        path=Path(path);meta=json.loads((path/'manifest.json').read_text(encoding='utf-8'))
        assert meta['data_verified']
        begin,end=meta['splits']['train'];assert begin==0
        events=np.load(path/'events.npy',mmap_mode='r')[begin:end]
        obj=cls.__new__(cls)
        obj.matrix=sparse.coo_matrix((np.ones(len(events),dtype=np.float32),(events[:,0],events[:,1])),
            shape=(meta['counts']['users'],meta['counts']['items'])).tocsr()
        obj.matrix.data[:]=1;obj.matrix.sort_indices();obj.users=None;obj.n_items=meta['counts']['items']
        obj.eligible=np.flatnonzero(np.diff(obj.matrix.indptr)>=2);obj._columns=obj.matrix.tocsc()
        if not len(obj.eligible):raise ValueError('No training users with two items')
        return obj
    def __init__(self,train_user_items,n_items):
        # Only caller-supplied TRAIN interactions are accepted. No timestamp,
        # held-out target, natural-language history, or sequence position input.
        self.users=[np.array(sorted(set(items)),dtype=np.int32) for items in train_user_items]
        indptr=np.cumsum([0]+[len(x) for x in self.users],dtype=np.int64)
        indices=np.concatenate(self.users) if self.users else np.array([],dtype=np.int32)
        self.matrix=sparse.csr_matrix((np.ones(len(indices),dtype=np.float32),indices,indptr),shape=(len(self.users),n_items))
        self.eligible=np.array([i for i,x in enumerate(self.users) if len(x)>=2],dtype=np.int32)
        self.n_items=n_items
        self._columns=self.matrix.tocsc()
        if not len(self.eligible):raise ValueError('No training users with at least two distinct items')
    def sample(self,batch,rng):
        users=rng.choice(self.eligible,size=batch,replace=True)
        # Uniform users, then uniform unordered pairs within each complete
        # training item set. Every pair is eligible, including distant items.
        pairs=[rng.choice(self.matrix.indices[self.matrix.indptr[u]:self.matrix.indptr[u+1]],size=2,replace=False) for u in users]
        return np.array(pairs,dtype=np.int64)
    def weights(self,ids):
        selected=self._columns[:,ids]
        weights=(selected.T@selected).toarray().astype(np.float32)
        np.fill_diagonal(weights,0)
        # Duplicate item occurrences within a batch are not positive or negative
        # comparisons with themselves; all duplicate-identity logits are masked.
        weights[np.equal.outer(ids,ids)]=0
        return weights
    def loss(self,z,ids,temperature=.1):
        weights=torch.tensor(self.weights(np.asarray(ids)),device=z.device)
        assert bool((weights.sum(1)>0).all())
        targets=weights/weights.sum(1,keepdim=True)
        duplicate=torch.tensor(np.equal.outer(ids,ids),device=z.device)
        logits=F.normalize(z.float(),dim=-1)@F.normalize(z.float(),dim=-1).T/temperature
        logp=F.log_softmax(logits.masked_fill(duplicate,-1e9),-1)
        # Multi-positive shared-user counts; known co-occurring batch items
        # receive positive mass instead of being treated as false negatives.
        return -(targets*logp).sum(1).mean()
