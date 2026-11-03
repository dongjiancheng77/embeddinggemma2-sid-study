"""Fixed public input recipe for the user-authorized independent reproduction."""
import argparse
import csv
import gzip
import html
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
from common import OUT,write,sha

BASE='https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/'
CATEGORIES={'Musical_Instruments':'Musical_Instruments'}

def download(url,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():return
    partial=path.with_suffix(path.suffix+'.partial')
    for attempt in range(5):
        try:
            start=partial.stat().st_size if partial.exists() else 0
            request=urllib.request.Request(url,headers={'Range':f'bytes={start}-'} if start else {})
            with urllib.request.urlopen(request,timeout=90) as response:
                if response.status!=206:start=0
                total=int(response.headers.get('Content-Length','0'))+start;done=start;tick=time.monotonic()
                with partial.open('ab' if start else 'wb') as f:
                    while True:
                        block=response.read(1024*1024)
                        if not block:break
                        f.write(block);done+=len(block)
                        if time.monotonic()-tick>3:
                            f.flush();write(path.parent/'download_progress.json',{'file':path.name,'bytes':done,'total':total,'url':url});tick=time.monotonic()
                if total and done!=total:raise IOError(f'Truncated response: {done}/{total}')
            partial.replace(path);print('DOWNLOADED',path.name,path.stat().st_size,flush=True);return
        except Exception as exc:
            print('DOWNLOAD_RETRY',attempt+1,repr(exc),flush=True)
            if attempt==4:raise
            time.sleep(2)

def clean(value):
    if isinstance(value,list):return ' ; '.join(clean(x) for x in value)
    return re.sub(r'\s+',' ',re.sub('<[^>]+>',' ',html.unescape(str(value or '')))).strip()

def prepare(dataset):
    path=OUT/'raw'/dataset;path.mkdir(parents=True,exist_ok=True)
    if (path/'complete.json').exists():return
    category=CATEGORIES[dataset]
    events_url=BASE+f'benchmark/5core/rating_only/{category}.csv.gz'
    meta_url=BASE+f'raw/meta_categories/meta_{category}.jsonl.gz'
    download(events_url,path/'official_events.csv.gz')
    events=path/'events.csv.gz';wanted=set();count=0
    if not events.exists():
        with gzip.open(path/'official_events.csv.gz','rt',encoding='utf-8',newline='') as f,gzip.open(events.with_suffix('.gz.partial'),'wt',encoding='utf-8',newline='') as out:
            reader=csv.DictReader(f);writer=csv.writer(out);writer.writerow(['user_id','item_id','timestamp'])
            for row in reader:
                writer.writerow([row['user_id'],row['parent_asin'],row['timestamp']]);wanted.add(row['parent_asin']);count+=1
                if count%200000==0:write(path/'normalize_progress.json',{'phase':'events','rows':count,'items':len(wanted)})
        events.with_suffix('.gz.partial').replace(events)
    else:
        with gzip.open(events,'rt',encoding='utf-8',newline='') as f:
            for row in csv.DictReader(f):wanted.add(row['item_id']);count+=1
    write(path/'source.json',{'release':'Amazon Reviews 2023, official deduplicated 5-core IDs before splitting',
        'source_url':events_url,'metadata_url':meta_url,'category':category,'events_sha256':sha(path/'official_events.csv.gz'),
        'normalization':'Rename parent_asin to item_id; no rating filter, no extra k-core or user holdout; preserve all supplied timestamps.',
        'paper_release_unknown':True,'user_authorized_independent_reproduction':True,
        'deviation':'User-selected Musical Instruments 2023 5-core; retain the global timestamp 80/10/10 protocol. This category is a new comparison dataset, not a paper Table1 reproduction.'})
    download(meta_url,path/'metadata.jsonl.gz')
    found=set();scanned=0;missing=0
    with gzip.open(path/'metadata.jsonl.gz','rt',encoding='utf-8') as f,(path/'catalog.jsonl.partial').open('w',encoding='utf-8') as out:
        for line in f:
            row=json.loads(line);scanned+=1;item=row['parent_asin']
            if item in wanted and item not in found:
                title=clean(row.get('title')) or category
                text=f"title: {title} | text: store: {clean(row.get('store'))}; category: {clean(row.get('categories'))}; features: {clean(row.get('features'))}; description: {clean(row.get('description'))}"
                out.write(json.dumps({'item_id':item,'text':text[:2000]},ensure_ascii=False)+'\n');found.add(item)
        for item in sorted(wanted-found):
            out.write(json.dumps({'item_id':item,'text':f'title: {category} product | text: Metadata unavailable.'})+'\n');missing+=1
    (path/'catalog.jsonl.partial').replace(path/'catalog.jsonl')
    write(path/'complete.json',{'dataset':dataset,'category':category,'events':count,'items':len(wanted),'metadata_scanned':scanned,
        'missing_metadata':missing,'missing_policy':'Retain event/catalog items with a category-only placeholder; never silently change the temporal split.',
        'events_url':events_url,'metadata_url':meta_url,'metadata_sha256':sha(path/'metadata.jsonl.gz')})
    print('ASSETS_COMPLETE',dataset,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=list(CATEGORIES)+['all'],default='all');args=p.parse_args()
    for dataset in CATEGORIES if args.dataset=='all' else [args.dataset]:prepare(dataset)

if __name__=='__main__':main()
