"""Reload saved LoRA adapters on CPU FP32 and compare their exported item vectors."""
import gc
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from experiment2_data import ROOT,OUT,MODEL,read

def main():
    from sentence_transformers import SentenceTransformer
    from peft import PeftModel
    torch.set_num_threads(4)
    texts=[i['text'] for i in read(ROOT/'work/beauty.json')['items'][:4]]
    records=[]
    for path in sorted(OUT.glob('*/seed*/lora')):
        if not (path/'complete.json').exists():continue
        model=SentenceTransformer(MODEL,device='cpu',local_files_only=True,
            config_kwargs={'vision_config':None,'audio_config':None},model_kwargs={'torch_dtype':torch.float32})
        model.max_seq_length=192
        model[0].model=PeftModel.from_pretrained(model[0].model,path/'adapter',is_trainable=False,local_files_only=True)
        value=model.encode(texts,batch_size=4,convert_to_tensor=True,normalize_embeddings=True,show_progress_bar=False)
        value=F.normalize(value[:,:128].float(),dim=-1).cpu().numpy()
        expected=np.load(path/'items128.npy')[:4]
        cosine=np.sum(value*expected,axis=1)
        assert float(cosine.min())>.99,(path,cosine)
        records.append({'run':str(path.relative_to(OUT)),'cpu_fp32_vs_gpu_bf16_min_cosine':float(cosine.min()),
                        'max_abs_difference':float(np.max(np.abs(value-expected))),'passed':True})
        del model;gc.collect()
    partial='--partial' in sys.argv
    assert len(records)>0 and (partial or len(records)==6),len(records)
    (OUT/('adapter_reload_partial.json' if partial else 'adapter_reload_verification.json')).write_text(json.dumps(records,indent=2))
    print(f'PASS: {len(records)} saved adapters reload and match exported GPU embeddings within BF16/FP32 tolerance.')

if __name__=='__main__':main()
