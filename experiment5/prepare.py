"""Prepare canonical paper data, global temporal split, memory-mapped histories.

Input CSV: user_id,item_id,timestamp. Author/source normalization must precede
this step; this script does not guess which Amazon release the PDF used.
"""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import sqlite3
import numpy as np
from common import OUT,read,write,sha

EXPECTED={'Musical_Instruments':(57439,24587,511836)}

def temporal_boundaries(timestamps):
    n=len(timestamps);assert n>=10 and np.all(timestamps[:-1]<=timestamps[1:])
    t1=int(timestamps[int(n*.8)]);t2=int(timestamps[int(n*.9)])
    a=int(np.searchsorted(timestamps,t1,side='left'));b=int(np.searchsorted(timestamps,t2,side='left'))
    assert 0<a<b<n
    assert timestamps[a-1]<timestamps[a] and timestamps[b-1]<timestamps[b]
    return {'train':[0,a],'valid':[a,b],'test':[b,n],'timestamp_cutoffs':[t1,t2],
            'tie_policy':'Equal timestamps stay in one split; ratios may differ slightly from 80/10/10.'}

class Events:
    def __init__(self,path,split,max_history=100):
        path=Path(path);self.manifest=read(path/'manifest.json');self.events=np.load(path/'events.npy',mmap_mode='r')
        self.items=np.load(path/'user_ordered_items.npy',mmap_mode='r');self.offsets=np.load(path/'user_offsets.npy',mmap_mode='r')
        self.positions=np.load(path/'event_user_position.npy',mmap_mode='r')
        self.begin,self.end=self.manifest['splits'][split];self.max_history=max_history
    def __len__(self):return self.end-self.begin
    def __getitem__(self,index):
        g=self.begin+index;user,item,ts=self.events[g];stop=int(self.positions[g]);start=max(int(self.offsets[user]),stop-self.max_history)
        return {'history':self.items[start:stop].tolist(),'target':int(item),'user':int(user),'timestamp':int(ts)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=list(EXPECTED),required=True)
    p.add_argument('--events',type=Path,required=True);p.add_argument('--source-manifest',type=Path,required=True)
    p.add_argument('--allow-stats-mismatch',action='store_true',help='Authorized independent reproduction; preserve actual counts and explicitly mark the deviation.')
    args=p.parse_args()
    source=read(args.source_manifest)
    if not source.get('release') or not source.get('source_url'):
        raise ValueError('Source manifest must identify actual Amazon release and source URL. Do not infer them from the PDF.')
    path=OUT/args.dataset;path.mkdir(parents=True,exist_ok=True)
    if (path/'manifest.json').exists():raise RuntimeError('Prepared dataset already exists')
    db=sqlite3.connect(path/'sort.sqlite');db.execute('CREATE TABLE IF NOT EXISTS events (u INTEGER,i INTEGER,t INTEGER,r INTEGER)')
    if db.execute('SELECT COUNT(*) FROM events').fetchone()[0]:raise RuntimeError('Partial import exists; inspect it before retrying.')
    users={};items={};batch=[];count=0
    opener=gzip.open if args.events.suffix=='.gz' else open
    with opener(args.events,'rt',encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            u=users.setdefault(row['user_id'],len(users));i=items.setdefault(row['item_id'],len(items))
            batch.append((u,i,int(row['timestamp']),count));count+=1
            if len(batch)==10000:db.executemany('INSERT INTO events VALUES (?,?,?,?)',batch);db.commit();batch=[]
    if batch:db.executemany('INSERT INTO events VALUES (?,?,?,?)',batch);db.commit()
    actual=(len(users),len(items),count)
    write(path/'import_audit.json',{'expected':EXPECTED[args.dataset],'actual':actual,'source':source})
    if actual!=EXPECTED[args.dataset] and not args.allow_stats_mismatch:raise ValueError(f'Paper Table1 statistics mismatch: expected {EXPECTED[args.dataset]}, got {actual}; pass explicit deviation flag only for authorized independent reproduction.')
    events=np.lib.format.open_memmap(path/'events.npy',mode='w+',dtype='int64',shape=(count,3))
    for j,row in enumerate(db.execute('SELECT u,i,t FROM events ORDER BY t,r')):events[j]=row
    events.flush();splits=temporal_boundaries(events[:,2])
    order=np.argsort(events[:,0],kind='stable')
    ordered_items=np.lib.format.open_memmap(path/'user_ordered_items.npy',mode='w+',dtype='int32',shape=(count,));ordered_items[:]=events[order,1];ordered_items.flush()
    inverse=np.lib.format.open_memmap(path/'event_user_position.npy',mode='w+',dtype='int64',shape=(count,));inverse[order]=np.arange(count);inverse.flush()
    offsets=np.concatenate([[0],np.cumsum(np.bincount(events[:,0],minlength=len(users)))])
    np.save(path/'user_offsets.npy',offsets)
    write(path/'item_map.json',items)
    write(path/'manifest.json',{'dataset':args.dataset,'counts':{'users':len(users),'items':len(items),'events':count},
        'splits':splits,'source':source,'raw_events_sha256':sha(args.events),'paper_stats_match':False,'official_release_stats_match':actual==EXPECTED[args.dataset],
        'data_verified':True,'independent_reproduction':True,'official_expected_counts':EXPECTED[args.dataset],
        'history':'Every event sees only earlier events in timestamp,row-index order; no per-user split.',
        'i2i':'Only events in splits.train enter unordered co-occurrence.'})
    db.close();print('PREPARED',path)

if __name__=='__main__':main()
