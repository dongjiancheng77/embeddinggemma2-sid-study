"""Build fixed recommendation/text contrastive examples and frozen backbone cache."""
import os
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from mini_tiger import sample_train_windows, val_windows

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'work/experiment2'
MODEL=os.environ.get('EMBEDDINGGEMMA2_MODEL', 'google/embeddinggemma-2')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare():
    OUT.mkdir(exist_ok=True,parents=True)
    data=read(ROOT/'work/beauty.json')
    texts=[i['text'] for i in data['items']]
    ids={t:i for i,t in enumerate(texts)}
    def index(t):
        if t not in ids:ids[t]=len(texts);texts.append(t)
        return ids[t]
    def query(hist):
        titles=[data['items'][i]['title'] or 'Beauty product' for i in hist[-10:]]
        return 'task: search result | query: Recommend the next beauty product for this purchase history, oldest to newest: '+' ; '.join(titles)
    rng=random.Random(20261029)
    examples={}
    for split,windows in [('rec_train',sample_train_windows(data['users'],10,8192,20261029)),
                          ('rec_validation',val_windows(data['users'],10))]:
        if split=='rec_validation':
            rng.shuffle(windows);windows=windows[:1000]
        rows=[]
        for hist,pos in windows:
            neg=rng.randrange(len(data['items']))
            while neg==pos or neg in hist:neg=rng.randrange(len(data['items']))
            rows.append({'q':index(query(hist)),'p':pos,'n':neg,'history':hist})
        examples[split]=rows
    for split,name in [('text_train','train'),('text_validation','validation')]:
        rows=[]
        for line in (ROOT/f'data/kalm/{name}.jsonl').read_text(encoding='utf-8').split('\n'):
            if not line:continue
            row=json.loads(line)
            rows.append({'q':index('task: search result | query: '+row['query']),
                         'p':index('title: none | text: '+row['positive']),
                         'n':index('title: none | text: '+row['negative'])})
        examples[split]=rows
    obj={'texts':texts,'examples':examples,'n_items':len(data['items']),
         'protocol':{'stage1_epochs':1,'batch_size':32,'temperature':.1,'head_lr':3e-4,'lora_lr':1e-4,
                     'rec_train_pairs':8192,'text_train_pairs':8192,'mix':'Alternate one rec and one text batch; rec exposure unchanged; mixed has twice the updates.',
                     'negatives':'31 other in-batch positives plus all 32 sampled hard/random negatives shared across the batch; duplicates/seen recommendation items masked.',
                     'lora_rank':8,'lora_alpha':16,'lora_targets':['q_proj','v_proj'],'max_tokens':192,
                     'selection':'Fixed final epoch, no test-dependent selection.',
                     'C':'Frozen original prefix, unchanged between settings.',
                     'D':'Centered unwhitened PCA128 fitted only on all-item frozen content, unchanged between settings.'}}
    (OUT/'dataset.json').write_text(json.dumps(obj,ensure_ascii=False),encoding='utf-8')
    (OUT/'data_manifest.json').write_text(json.dumps({'sha256':sha(OUT/'dataset.json'),
        'beauty_sha256':sha(ROOT/'work/beauty.json'),'kalm':read(ROOT/'data/kalm/manifest.json'),
        'counts':{k:len(v) for k,v in examples.items()},'unique_texts':len(texts),'model':MODEL,
        'specmrl_source':read(ROOT/'sources/spec-mrl-revision.json'),'protocol':obj['protocol']},indent=2),encoding='utf-8')
    print('Prepared',len(texts),{k:len(v) for k,v in examples.items()},flush=True)

def cache():
    from sentence_transformers import SentenceTransformer
    d=read(OUT/'dataset.json')
    model=SentenceTransformer(MODEL,device='cuda',local_files_only=True,
        config_kwargs={'vision_config':None,'audio_config':None},model_kwargs={'torch_dtype':torch.bfloat16})
    model.max_seq_length=192
    n=d['n_items']
    extra=model.encode(d['texts'][n:],batch_size=16,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=True).astype('float32')
    extra/=np.linalg.norm(extra,axis=1,keepdims=True).clip(1e-12)
    full=np.concatenate([np.load(ROOT/'work/full768.npy'),extra])
    assert full.shape==(len(d['texts']),768) and np.isfinite(full).all()
    np.save(OUT/'base768.npy',full)
    print('Cache ready',full.shape,flush=True)

if __name__=='__main__':
    import sys
    torch.set_num_threads(4)
    if sys.argv[-1]=='prepare':prepare()
    else:cache()
