"""Exact, uncapped TRAIN-only Swing. No test interaction enters the graph."""
import itertools
import time
import numpy as np
from scipy import sparse
from common import OUT, write, sha
from i2i import Cooccurrence

VARIANTS=['all_cooccurrence','swing_top1_top5','swing_mutual_top1']

def rank_and_filter(scores):
    n=scores.shape[0]; top=np.full((n,5),-1,dtype=np.int64)
    for i in range(n):
        a,b=scores.indptr[i:i+2]; ids=scores.indices[a:b]; values=scores.data[a:b]
        order=np.lexsort((ids,-values))[:5];top[i,:len(order)]=ids[order]
    edges=[set(),set()]
    for i in range(n):
        j=int(top[i,0])
        if j<0:continue
        pair=(min(i,j),max(i,j))
        if i in top[j]:edges[0].add(pair)
        if top[j,0]==i:edges[1].add(pair)
    return top,[np.array(sorted(x),dtype=np.int64).reshape(-1,2) for x in edges]

def main():
    root=OUT/'Musical_Instruments';dest=root/'swing';dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():return
    graph=Cooccurrence.from_prepared(root);m=graph.matrix;n=m.shape[1]
    lengths=np.diff(m.indptr);weights=1/np.sqrt(np.maximum(lengths,1));scores={};start=time.monotonic();structures=0
    # A block of user-user intersections avoids materializing a full U x U matrix.
    # Each distinct user pair contributes once to each unordered common-item pair.
    for begin in range(0,m.shape[0],128):
        if (OUT/'STOP').exists():raise RuntimeError('User stop')
        shared=(m[begin:begin+128]@m.T).tocoo()
        keep=(shared.data>=2)&(shared.col>shared.row+begin)
        for u,v,count in zip(shared.row[keep]+begin,shared.col[keep],shared.data[keep]):
            common=np.intersect1d(m.indices[m.indptr[u]:m.indptr[u+1]],m.indices[m.indptr[v]:m.indptr[v+1]],assume_unique=True)
            assert len(common)==int(count)
            contribution=float(weights[u]*weights[v]/(1+len(common)))
            for a,b in itertools.combinations(common,2):
                key=int(a)*n+int(b);scores[key]=scores.get(key,0.)+contribution
            structures+=1
        write(dest/'progress.json',{'users_done':min(begin+128,m.shape[0]),'users':m.shape[0],
              'scored_item_pairs':len(scores),'user_pairs':structures,'seconds':time.monotonic()-start})
        if begin%2048==0:print('SWING',begin,m.shape[0],len(scores),flush=True)
    keys=np.fromiter(scores.keys(),dtype=np.int64);values=np.fromiter(scores.values(),dtype=np.float64)
    rows=keys//n;cols=keys%n
    mat=sparse.coo_matrix((np.r_[values,values],(np.r_[rows,cols],np.r_[cols,rows])),shape=(n,n)).tocsr()
    sparse.save_npz(dest/'scores.npz',mat);top,edges=rank_and_filter(mat);np.save(dest/'top5.npy',top)
    stats={}
    for name,pairs in zip(VARIANTS[1:],edges):
        if not len(pairs):raise RuntimeError('No eligible pairs: '+name)
        np.save(dest/(name+'.npy'),pairs)
        stats[name]={'unordered_pairs':len(pairs),'covered_items':len(np.unique(pairs)),
                     'catalog_coverage':len(np.unique(pairs))/n}
    write(dest/'complete.json',{'formula':'sum over u<v in U(i) intersect U(j): 1/sqrt(|Iu| |Iv|) / (1+|Iu intersect Iv|)',
          'alpha':1,'user_weight':'1/sqrt(distinct training items)','exact':True,'caps_or_sampling':False,
          'self_user_pairs':False,'source':'https://arxiv.org/abs/2010.05525',
          'ties':'descending score, ascending canonical item index; zero scores excluded',
          'only_train':True,'events_sha256':sha(root/'events.npy'),'item_pairs':len(scores),
          'filter_semantics':'top1/top5 conditions on directed rankings, then union into unordered positive edges',
          'variants':stats,'seconds':time.monotonic()-start})

if __name__=='__main__':main()
