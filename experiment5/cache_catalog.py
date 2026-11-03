"""Guarded EmbeddingGemma2 content cache, after canonical data are verified."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from common import OUT,read,write,guard,sha
from fit_i2i import MODEL

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--catalog',type=Path,required=True);args=p.parse_args()
    path=OUT/args.dataset;meta=read(path/'manifest.json');assert meta['data_verified']
    if (path/'embedding_cache/complete.json').exists():raise RuntimeError('Verified completed cache is protected; use a fresh workspace to re-encode.')
    mapping=read(path/'item_map.json');items=[None]*len(mapping)
    with args.catalog.open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            if row['item_id'] in mapping:items[mapping[row['item_id']]]={'text':row['text'],'item_id':row['item_id']}
    if any(row is None or not row['text'].strip() for row in items):raise ValueError('Catalog text missing; do not silently drop items from paper splits')
    write(path/'catalog.json',items)
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    model=SentenceTransformer(MODEL,device='cpu',local_files_only=True,
        config_kwargs={'vision_config':None,'audio_config':None},model_kwargs={'torch_dtype':torch.bfloat16})
    tuning=read(path/'embedding_tuning.json') if (path/'embedding_tuning.json').exists() else {'chosen_batch':4,'layer_sleep':.003,'step_sleep':.02}
    batch=int(tuning['chosen_batch'])
    model.max_seq_length=192;model.eval();monitor=guard(path/'embedding_cache',layer_sleep=tuning['layer_sleep'])
    output=path/'base768.npy';progress=path/'embedding_cache/progress.json';last=read(progress)['done'] if progress.exists() else 0
    z=np.lib.format.open_memmap(output,mode='r+' if output.exists() else 'w+',dtype='float32',shape=(len(items),768))
    try:
        torch.cuda.set_per_process_memory_fraction(.55);monitor.attach(model);model.to('cuda')
        with torch.inference_mode():
            for start in range(last,len(items),batch):
                values=model.encode([r['text'] for r in items[start:start+batch]],batch_size=batch,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype('float32')
                values/=np.linalg.norm(values,axis=1,keepdims=True).clip(1e-12)
                z[start:start+len(values)]=values
                if (start-last)%(batch*16)==0 or start+len(values)==len(items):
                    z.flush();write(progress,{'done':start+len(values),'total':len(items)})
                monitor.pulse(tuning['step_sleep'])
        write(path/'embedding_cache/complete.json',{'model':MODEL,'catalog_sha256':sha(args.catalog),'items':len(items),'dimension':768})
    finally:monitor.close()

if __name__=='__main__':main()
