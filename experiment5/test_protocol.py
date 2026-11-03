"""CPU checks of Swing values/filtering, loss and epoch coverage."""
import tempfile
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from swing import rank_and_filter
from pairs import SwingPairs
from i2i import Cooccurrence
from batching import grouped_order

def main():
    # 0->1 is top1; 1->2 is top1; 0 is still in 1's top5.
    scores=sparse.csr_matrix(np.array([[0,2,1],[2,0,3],[1,3,0]],dtype=float))
    top,edges=rank_and_filter(scores)
    assert edges[0].tolist()==[[0,1],[1,2]]
    assert edges[1].tolist()==[[1,2]]
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/'swing').mkdir();np.save(root/'swing/x.npy',edges[0])
        graph=SwingPairs(root,'x',4);ids=np.array([0,1,1,2]);z=torch.randn(4,128,requires_grad=True)
        w=graph.weights(ids);assert w[0,3]==0 and w[1,2]==0 and w[0,1]==1
        loss=graph.loss(z,ids);loss.backward();assert torch.isfinite(loss) and z.grad.abs().sum()>0
    base=Cooccurrence([[0,1,2],[0,1]],3)
    assert base.weights(np.arange(3)).tolist()==[[0,2,1],[2,0,1],[1,1,0]]
    class Dummy:
        begin=0;end=101;max_history=100
        positions=np.arange(101);offsets=np.array([0]);events=np.zeros((101,3),dtype=int)
        def __len__(self):return 101
    order=grouped_order(Dummy(),0,32,64);assert sorted(order)==list(range(101))
    assert order==grouped_order(Dummy(),0,32,64)
    print('PASS: ranking, reciprocal filters, duplicate masking, weighted loss gradients, epoch coverage')

if __name__=='__main__':main()
