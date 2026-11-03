"""Independent three-level RQ-VAE; no MiniOneRec imports or fourth SID token."""
import argparse
import collections
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from common import OUT,write,read

class RQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(128,256),nn.GELU(),nn.Linear(256,64),nn.GELU(),nn.Linear(64,32))
        self.decoder=nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Linear(64,256),nn.GELU(),nn.Linear(256,128))
        self.books=nn.ModuleList([nn.Embedding(256,32) for _ in range(3)])
    def forward(self,x):
        z=self.encoder(x);residual=z;total=0;loss=0;codes=[]
        for book in self.books:
            # Double-precision distances prevent cancellation for close centers;
            # the main repair is periodic centroid refresh and slower encoder.
            a=residual.double();b=book.weight.double()
            distances=a.square().sum(1,keepdim=True)+b.square().sum(1)[None,:]-2*a@b.T
            code=distances.argmin(1);q=book(code);codes.append(code)
            loss=loss+F.mse_loss(q,residual.detach())+.25*F.mse_loss(q.detach(),residual)
            total=total+q;residual=residual-q.detach()
        quant=z+(total-z).detach()
        return self.decoder(quant),loss/3,torch.stack(codes,1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--kind',required=True);args=p.parse_args()
    dataset_root=OUT/args.dataset
    path=dataset_root/args.kind
    if (path/'sid.json').exists():raise RuntimeError('Completed SID exists')
    meta=read(dataset_root/'manifest.json');assert meta['data_verified']
    values=np.load(path/'items128.npy')
    assert values.shape==(meta['counts']['items'],128) and np.isfinite(values).all()
    x=torch.from_numpy(values);torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
    model=RQ();optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    for epoch in range(5):
        for ids in torch.randperm(len(x)).split(512):
            optim.zero_grad();loss=F.mse_loss(model.decoder(model.encoder(x[ids])),x[ids]);loss.backward();optim.step()
    optim=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':.0001},
        {'params':model.decoder.parameters(),'lr':.001},{'params':model.books.parameters(),'lr':.001}],weight_decay=.0001)
    def refresh(epoch):
        with torch.no_grad():residual=torch.cat([model.encoder(chunk) for chunk in x.split(2048)]).numpy().astype('float64')
        for j,book in enumerate(model.books):
            km=MiniBatchKMeans(256,batch_size=2048,n_init=3,max_iter=100,random_state=j+epoch*100).fit(residual)
            book.weight.data.copy_(torch.from_numpy(km.cluster_centers_).float())
            residual-=km.cluster_centers_[km.predict(residual)]
            optim.state.pop(book.weight,None)
    refresh(0);losses=[];trajectory=[]
    for epoch in range(20):
        values=[]
        for ids in torch.randperm(len(x)).split(512):
            optim.zero_grad();recon,qloss,_=model(x[ids]);loss=F.mse_loss(recon,x[ids])+qloss
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optim.step();values.append(float(loss.detach()))
        refresh(epoch+1)
        with torch.inference_mode():current=torch.cat([model(chunk)[2] for chunk in x.split(1024)]).numpy()
        usage=[int(len(np.unique(current[:,j]))) for j in range(3)]
        collision=1-len(np.unique(current,axis=0))/len(current)
        trajectory.append({'epoch':epoch+1,'usage':usage,'collision_rate':collision})
        losses.append(float(np.mean(values)));write(path/'quantizer_progress.json',trajectory[-1])
        print('RQ_REPAIRED',epoch+1,losses[-1],usage,collision,flush=True)
    model.eval()
    with torch.inference_mode():codes=torch.cat([model(chunk)[2] for chunk in x.split(512)]).numpy()
    unique=len(np.unique(codes,axis=0))
    usage=[int(len(np.unique(codes[:,j]))) for j in range(3)]
    _,groups=np.unique(codes,axis=0,return_counts=True)
    if min(usage)<240 or 1-unique/len(codes)>.15 or groups.max()>100:
        write(path/'quantizer_quality_failure.json',{'usage':usage,'collision_rate':1-unique/len(codes),'largest_group':int(groups.max())})
        raise RuntimeError('Quantizer quality gate failed; no downstream GR will be trained')
    np.save(path/'sid.npy',codes);torch.save(model.state_dict(),path/'rqvae.pt')
    write(path/'sid.json',{'levels':3,'codebook_sizes':[256]*3,'items':len(codes),'unique_paths':unique,
        'collision_excess':len(codes)-unique,'collision_rate':1-unique/len(codes),
        'collision_policy':'No suffix; same-score item expansion, deterministic item-index tie breaking.',
        'source':'Repaired independent RQVAE: slower encoder, per-epoch KMeans refresh, reset codebook optimizer moments.',
        'recipe':'repair_v2','usage':usage,'largest_collision_group':int(groups.max()),
        'residual_collisions':'Natural three-level collisions are retained and reported, not forced to zero; no fourth token or new scoring module.',
        'epochs':20,'ae_warmup_epochs':5,'losses':losses,'trajectory':trajectory})

if __name__=='__main__':main()
