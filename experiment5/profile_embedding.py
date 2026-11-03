"""Guarded real-text inference batch tuning before the long cache job."""
import time
import numpy as np
import torch
from common import OUT,read,write,guard
from fit_i2i import MODEL

def main():
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4);root=OUT/'Musical_Instruments'
    if not (root/'catalog.json').exists():
        import json
        mapping=read(root/'item_map.json');catalog=[None]*len(mapping)
        with (OUT/'raw/Musical_Instruments/catalog.jsonl').open(encoding='utf-8') as source:
            for line in source:
                row=json.loads(line)
                if row['item_id'] in mapping:catalog[mapping[row['item_id']]]=row
        assert all(catalog);write(root/'catalog.json',catalog)
    catalog=read(root/'catalog.json')
    model=SentenceTransformer(MODEL,device='cpu',local_files_only=True,config_kwargs={'vision_config':None,'audio_config':None},model_kwargs={'torch_dtype':torch.bfloat16})
    model.max_seq_length=192;model.eval();torch.cuda.set_per_process_memory_fraction(.55)
    trials=[];chosen=4
    for batch in [8,16,32]:
        monitor=guard(root/f'embedding_tune_b{batch}',layer_sleep=.0015)
        try:
            monitor.attach(model);model.to('cuda');start=time.monotonic()
            with torch.inference_mode():
                for j in range(24):
                    text=[x['text'] for x in catalog[j*batch:(j+1)*batch]]
                    z=model.encode(text,batch_size=batch,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
                    assert z.shape==(batch,768) and np.isfinite(z).all();monitor.pulse(.01)
            seconds=time.monotonic()-start;trials.append({'batch':batch,'items_per_second':24*batch/seconds,'peak_gpu':monitor.peak})
            if monitor.peak<=78:chosen=batch
            if monitor.peak>=75:break
        finally:monitor.close()
    write(root/'embedding_tuning.json',{'chosen_batch':chosen,'layer_sleep':.0015,'step_sleep':.01,'trials':trials,'purpose':'inference only; LoRA microbatch remains4'})
    print(trials,'chosen',chosen,flush=True)

if __name__=='__main__':main()
