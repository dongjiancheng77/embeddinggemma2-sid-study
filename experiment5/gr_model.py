"""Paper-sized scratch decoder, ordinary per-level NTP, cached beam=100.

Confirmed by the PDF: 8 layers, 8 heads, width512, FFN2304, ~27.7M.
Not specified by the PDF: pre-LN/bias-free/tied embeddings/sinusoidal positions/
GELU/dropout0.1. These are explicit provisional implementation choices, not
claims about the unavailable author's exact code.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

PAD=768;BOS=769;SEP=770;EOS=771

class GRDecoderLayer(nn.Module):
    def __init__(self,width=512,heads=8,ff=2304,dropout=.1):
        super().__init__();self.heads=heads;self.dim=width//heads;self.dropout=dropout
        self.norm1=nn.LayerNorm(width,bias=False);self.norm2=nn.LayerNorm(width,bias=False)
        self.qkv=nn.Linear(width,3*width,bias=False);self.out=nn.Linear(width,width,bias=False)
        self.ff1=nn.Linear(width,ff,bias=False);self.ff2=nn.Linear(ff,width,bias=False)
    def forward(self,x,allowed,past=None,use_cache=False):
        b,t,d=x.shape;q,k,v=self.qkv(self.norm1(x)).chunk(3,-1)
        q,k,v=[z.reshape(b,t,self.heads,self.dim).transpose(1,2) for z in [q,k,v]]
        if past is not None:k=torch.cat([past[0],k],2);v=torch.cat([past[1],v],2)
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=allowed,dropout_p=self.dropout if self.training else 0)
        x=x+F.dropout(self.out(y.transpose(1,2).reshape(b,t,d)),self.dropout,self.training)
        x=x+F.dropout(self.ff2(F.gelu(self.ff1(self.norm2(x)))),self.dropout,self.training)
        return x,(k,v) if use_cache else None

class BasicGR(nn.Module):
    def __init__(self,width=512,layers=8,heads=8,ff=2304,dropout=.1,max_tokens=512,input_scale=1.):
        super().__init__();self.width=width;self.input_scale=float(input_scale)
        self.embedding=nn.Embedding(772,width,padding_idx=PAD)
        self.layers=nn.ModuleList([GRDecoderLayer(width,heads,ff,dropout) for _ in range(layers)])
        self.norm=nn.LayerNorm(width,bias=False)
        pos=torch.arange(max_tokens)[:,None];freq=torch.exp(torch.arange(0,width,2)*(-math.log(10000.)/width))
        enc=torch.zeros(max_tokens,width);enc[:,0::2]=torch.sin(pos*freq);enc[:,1::2]=torch.cos(pos*freq)
        self.register_buffer('position',enc,persistent=False)
        self.apply(self.initialize)
    @staticmethod
    def initialize(module):
        if isinstance(module,(nn.Linear,nn.Embedding)):nn.init.normal_(module.weight,std=.02)
    def hidden(self,ids,key_mask,past=None,use_cache=False):
        b,t=ids.shape;previous=0 if past is None else past[0][0].shape[2]
        positions=(key_mask.long().cumsum(-1)-1).clamp_min(0)[:,-t:]
        x=self.embedding(ids)*self.input_scale+self.position[positions].to(self.embedding.weight.dtype)
        causal=torch.arange(previous+t,device=ids.device)[None,:]<=previous+torch.arange(t,device=ids.device)[:,None]
        allowed=causal[None,None,:,:]&key_mask[:,None,None,:].bool()
        cache=[]
        for j,layer in enumerate(self.layers):
            x,kv=layer(x,allowed,None if past is None else past[j],use_cache)
            if use_cache:cache.append(kv)
        return self.norm(x),cache if use_cache else None
    def logits(self,hidden,level):
        return F.linear(hidden,self.embedding.weight[level*256:(level+1)*256]).float()
    def loss(self,ids,mask,target_positions,target_codes):
        h,_=self.hidden(ids,mask);row=torch.arange(len(ids),device=ids.device)
        return torch.stack([F.cross_entropy(self.logits(h[row,target_positions[:,level]],level),target_codes[:,level]) for level in range(3)]).mean()

def item_prompt(history,codes,max_history=100):
    tokens=[BOS]
    for item in history[-max_history:]:tokens.extend([int(c)+256*j for j,c in enumerate(codes[item])])
    return tokens+[SEP]

def collate(events,codes,device='cpu',max_history=100):
    prompts=[item_prompt(r['history'],codes,max_history) for r in events]
    targets=[list(map(int,codes[r['target']])) for r in events]
    sequences=[p+[c+256*j for j,c in enumerate(y[:2])] for p,y in zip(prompts,targets)]
    length=max(map(len,sequences));ids=[];masks=[];positions=[]
    for p,s in zip(prompts,sequences):
        padding=length-len(s);ids.append([PAD]*padding+s);masks.append([0]*padding+[1]*len(s))
        positions.append([padding+len(p)-1+j for j in range(3)])
    return [torch.tensor(v,device=device,dtype=torch.long) for v in [ids,masks,positions,targets]]

def make_trie(codes):
    trie={};leaves={}
    for i,code in enumerate(codes):
        code=tuple(map(int,code));leaves.setdefault(code,[]).append(i)
        for depth in range(3):trie.setdefault(code[:depth],set()).add(code[depth])
    return {p:sorted(v) for p,v in trie.items()},leaves

@torch.inference_mode()
def retrieve(model,history,codes,trie,leaves,beam=100,topk=100,device='cpu'):
    model.eval();prompt=item_prompt(history,codes)
    ids=torch.tensor([prompt],device=device);mask=torch.ones_like(ids)
    hidden,cache=model.hidden(ids,mask,use_cache=True)
    state=hidden[:,-1];prefixes=[()];scores=torch.zeros(1,device=device)
    for level in range(3):
        # Standard NTP: normalize over all 256 level tokens, then constrain.
        # Mask-before-normalize would implement MSL, which is not this baseline.
        lp=F.log_softmax(model.logits(state,level),-1)
        constrained=torch.full_like(lp,-torch.inf)
        for row,prefix in enumerate(prefixes):constrained[row,trie[prefix]]=lp[row,trie[prefix]]
        combined=scores[:,None]+constrained;count=min(beam,int(torch.isfinite(combined).sum()))
        scores,flat=combined.flatten().topk(count);parents=flat//256;tokens=flat%256
        prefixes=[prefixes[parent]+(token,) for parent,token in zip(parents.tolist(),tokens.tolist())]
        if level<2:
            cache=[(k.index_select(0,parents),v.index_select(0,parents)) for k,v in cache]
            mask=torch.cat([mask.index_select(0,parents),torch.ones(count,1,device=device,dtype=torch.long)],1)
            hidden,cache=model.hidden((tokens+256*level)[:,None],mask,past=cache,use_cache=True);state=hidden[:,-1]
    # Shared SID paths expand to all corresponding items at the same score.
    # Ties are deterministic by item index, never resolved using target labels.
    ranked=sorted([(float(score),item) for prefix,score in zip(prefixes,scores) for item in leaves[prefix]],key=lambda r:(-r[0],r[1]))
    return [item for _,item in ranked[:topk]]

@torch.inference_mode()
def retrieve_batch(model,histories,codes,trie,leaves,beam=100,topk=100,device='cpu'):
    """Batched KV-cached ordinary-NTP beam search; one host transfer per level."""
    model.eval();prompts=[item_prompt(h,codes) for h in histories];length=max(map(len,prompts))
    ids=torch.tensor([[PAD]*(length-len(p))+p for p in prompts],device=device)
    mask=torch.tensor([[0]*(length-len(p))+[1]*len(p) for p in prompts],device=device)
    hidden,cache=model.hidden(ids,mask,use_cache=True);state=hidden[:,-1]
    prefixes=[() for _ in histories];scores=torch.zeros(len(histories),device=device);sizes=[1]*len(histories)
    for level in range(3):
        lp=F.log_softmax(model.logits(state,level),-1);allowed=torch.full_like(lp,-torch.inf)
        rows=[];columns=[]
        for row,prefix in enumerate(prefixes):
            values=trie[prefix];rows.extend([row]*len(values));columns.extend(values)
        ri=torch.tensor(rows,device=device);ci=torch.tensor(columns,device=device);allowed[ri,ci]=lp[ri,ci]
        next_scores=[];next_parents=[];next_tokens=[];next_sizes=[];begin=0
        for size in sizes:
            values=scores[begin:begin+size,None]+allowed[begin:begin+size]
            count=min(beam,int(torch.isfinite(values).sum()))
            value,index=values.flatten().topk(count)
            next_scores.append(value);next_parents.append(begin+index//256);next_tokens.append(index%256);next_sizes.append(count);begin+=size
        scores=torch.cat(next_scores);parents=torch.cat(next_parents);tokens=torch.cat(next_tokens)
        prefixes=[prefixes[parent]+(token,) for parent,token in zip(parents.tolist(),tokens.tolist())];sizes=next_sizes
        if level<2:
            cache=[(k.index_select(0,parents),v.index_select(0,parents)) for k,v in cache]
            mask=torch.cat([mask.index_select(0,parents),torch.ones(len(parents),1,device=device,dtype=torch.long)],1)
            hidden,cache=model.hidden((tokens+256*level)[:,None],mask,past=cache,use_cache=True);state=hidden[:,-1]
    values=scores.tolist();results=[];begin=0
    for size in sizes:
        items=[(values[j],item) for j in range(begin,begin+size) for item in leaves[prefixes[j]]]
        items.sort(key=lambda row:(-row[0],row[1]));results.append([item for _,item in items[:topk]]);begin+=size
    return results
