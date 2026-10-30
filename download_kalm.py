"""Pin and verify selected official KaLM parquet sources, then sample disjoint pairs."""
import hashlib
import json
from pathlib import Path
import random
import re
import urllib.request

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'data/kalm'
FILES={'quora/train-00000-of-00001.parquet':3072,
       'snli/train-00000-of-00001.parquet':2048,
       'msmarco-passage/train-00000-of-00002.parquet':3072}

def clean(text):
    return re.split(r'\bQuery\s*:\s*',str(text),maxsplit=1,flags=re.I)[-1].strip()

def main():
    import pyarrow.parquet as pq
    DEST.mkdir(exist_ok=True,parents=True)
    if not (DEST/'repository.json').exists():
        revision='e9443ab6f5d4dc29c79cea03834e932428ed6ab1'
        url=f'https://huggingface.co/api/datasets/KaLM-Embedding/KaLM-embedding-finetuning-data/tree/{revision}?recursive=true&limit=1000'
        tree=json.load(urllib.request.urlopen(url,timeout=60))
        (DEST/'repository.json').write_text(json.dumps({'revision':revision,'tree':tree},indent=2))
    meta=json.loads((DEST/'repository.json').read_text())
    lookup={r['path']:r for r in meta['tree']}
    rev=meta['revision']
    manifest={'repository':'KaLM-Embedding/KaLM-embedding-finetuning-data','revision':rev,'seed':20261029,
              'sampling':'Uniform without replacement within selected English source shards; not original full-mixture replication.',
              'files':[], 'normalization':'Strip source Instruct/Query wrappers; apply native EmbeddingGemma2 search/document prompts.',
              'hard_negatives_used':1}
    train=[]; val=[]; seen=set()
    rng=random.Random(20261029)
    for name,count in FILES.items():
        target=DEST/name.replace('/','__')
        record=lookup[name]
        expected=record['lfs']['oid']
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
            url=f'https://huggingface.co/datasets/{manifest["repository"]}/resolve/{rev}/{name}?download=true'
            print('DOWNLOAD',name,record['size'],flush=True)
            with urllib.request.urlopen(url,timeout=120) as response, target.with_suffix('.partial').open('wb') as f:
                while chunk:=response.read(1024*1024):f.write(chunk)
            target.with_suffix('.partial').replace(target)
        actual=hashlib.sha256(target.read_bytes()).hexdigest()
        assert actual==expected,(name,actual,expected)
        table=pq.read_table(target)
        candidates=list(range(len(table)));rng.shuffle(candidates)
        chosen=[]
        for ix in candidates:
            row=table.slice(ix,1).to_pylist()[0]
            if not row.get('pos') or not row.get('neg'):continue
            q=clean(row['query']);pos=clean(rng.choice(row['pos']));neg=clean(rng.choice(row['neg']))
            if not q or not pos or not neg or len({q,pos,neg})<3 or any(x in seen for x in [q,pos,neg]):continue
            seen.update([q,pos,neg])
            chosen.append({'query':q,'positive':pos,'negative':neg,'source':name,'row':ix})
            if len(chosen)==count+256:break
        assert len(chosen)==count+256,(name,len(chosen))
        train.extend(chosen[:count]);val.extend(chosen[count:])
        manifest['files'].append({'path':name,'local':str(target),'rows':len(table),'sha256':actual,'train':count,'validation':256})
        print('VERIFIED',name,'train',count,'validation',256,flush=True)
    rng.shuffle(train);rng.shuffle(val)
    for split,rows in [('train',train),('validation',val)]:
        target=DEST/f'{split}.jsonl'
        target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        manifest[split]={'rows':len(rows),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('DONE',len(train),len(val),flush=True)

if __name__=='__main__':main()
