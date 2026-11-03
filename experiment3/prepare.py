"""Official Amazon'23 temporal splits; no general-text training examples."""
import csv
import gzip
import hashlib
import html
import json
from pathlib import Path
import random
import re
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/experiment3'

def read(path):
    deadline=time.monotonic()+3.
    while True:
        try:return json.loads(Path(path).read_text(encoding='utf-8'))
        except PermissionError:
            if time.monotonic()>=deadline:raise
            time.sleep(.02)
def atomic_text(path,content,timeout=3.):
    """Publish a complete file, tolerating brief Windows reader/delete locks."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,
                                     prefix=path.name+'.',suffix='.tmp',delete=False) as f:
        f.write(content);tmp=Path(f.name)
    deadline=time.monotonic()+timeout
    try:
        while True:
            try:tmp.replace(path);return
            except PermissionError:
                if time.monotonic()>=deadline:raise
                time.sleep(.02)
    finally:tmp.unlink(missing_ok=True)

def write(path,obj):
    atomic_text(path,json.dumps(obj,ensure_ascii=False,indent=2))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def clean(value):
    if isinstance(value,list):return ' ; '.join(clean(x) for x in value)
    return re.sub(r'\s+',' ',re.sub('<[^>]*>',' ',html.unescape(str(value or '')))).strip()

def main():
    data={};asins=set();earliest={};stats={}
    for split in ['train','valid','test']:
        with gzip.open(OUT/'raw'/f'{split}.csv.gz','rt',encoding='utf-8',newline='') as f:
            rows=list(csv.DictReader(f))
        for r in rows:
            r['timestamp']=int(r['timestamp']);r['history']=r['history'].split()
            asins.add(r['parent_asin']);asins.update(r['history'])
            key=(r['user_id'],r['parent_asin'])
            earliest[key]=min(earliest.get(key,r['timestamp']),r['timestamp'])
        stats[split]={'rows':len(rows),'min_timestamp':min(r['timestamp'] for r in rows),
                      'max_timestamp':max(r['timestamp'] for r in rows)}
        data[split]=rows
    assert stats['train']['max_timestamp']<stats['valid']['min_timestamp']
    assert stats['valid']['max_timestamp']<stats['test']['min_timestamp']
    # Prove each supplied history refers to an observed event at/before target.
    checked=0
    for rows in data.values():
        for r in rows:
            for a in r['history']:
                assert earliest[(r['user_id'],a)]<=r['timestamp']
                checked+=1
            assert r['parent_asin'] not in r['history'],'Expected de-duplicated data'
    catalog={};scanned=0
    with gzip.open(OUT/'raw/meta.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            m=json.loads(line);scanned+=1;a=m['parent_asin']
            if a not in asins:continue
            title=clean(m.get('title')) or 'Industrial and scientific product'
            text=f"title: {title} | text: store: {clean(m.get('store'))}; categories: {clean(m.get('categories'))}; features: {clean(m.get('features'))}; description: {clean(m.get('description'))}"
            catalog[a]={'asin':a,'title':title,'text':text[:2000],'missing_metadata':False}
    ordered=sorted(asins);indices={a:i for i,a in enumerate(ordered)}
    items=[catalog.get(a,{'asin':a,'title':'Industrial and scientific product',
                         'text':'title: Industrial and scientific product | text: Metadata unavailable.',
                         'missing_metadata':True}) for a in ordered]
    # Deterministic uniform event samples from official splits; same for A--E.
    limits={'train':8192,'valid':512,'test':1024}
    selected={}
    for j,(split,rows) in enumerate(data.items()):
        usable=[r for r in rows if r['history']]
        rng=random.Random(20261030+j);rng.shuffle(usable)
        picked=usable[:limits[split]]
        selected[split]=[{'user':hashlib.sha256(r['user_id'].encode()).hexdigest()[:24],
                          'target':indices[r['parent_asin']],
                          'history':[indices[a] for a in r['history'][-10:]],
                          'timestamp':r['timestamp']} for r in picked]
        stats[split].update(nonempty_history_rows=len(usable),selected_rows=len(picked))
    obj={'items':items,'splits':selected,'seed':0,'history_length':10,
         'stats':stats,'audit':{'checked_history_events':checked,'catalog_size':len(items),
                               'metadata_rows_scanned':scanned,'missing_metadata':sum(i['missing_metadata'] for i in items)},
         'protocol':{'dataset':'Amazon Reviews 2023 Industrial_and_Scientific, official deduplicated 5-core timestamp_w_his',
                     'catalog':'All split item metadata, transductive; no future interaction labels used for training.',
                     'stage1':'Next-item contrastive learning, previous item as query, next item as positive, in-batch plus random negatives. Recommendation train split only.',
                     'supervision':'No KaLM, no general text pairs, no SID-to-title auxiliary task; product metadata is encoder input only.',
                     'gr':'MiniOneRec SidSFTDataset, Qwen2.5-0.5B LoRA + new SID token rows, recommendation-only SFT; no RL.',
                     'scale':'Initial single-seed 8192-event run, not a full-size official MiniOneRec reproduction.',
                     'evaluation':'Fixed 512 validation / 1024 test events, full catalog constrained SID beam search. No sampled negatives.'}}
    write(OUT/'dataset.json',obj)
    write(OUT/'data_audit.json',{'dataset_sha256':sha(OUT/'dataset.json'),'stats':stats,**obj['audit'],'protocol':obj['protocol']})
    print(json.dumps(read(OUT/'data_audit.json'),ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
