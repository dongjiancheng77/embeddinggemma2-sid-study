"""Record the 192-token cap's effect without changing training inputs."""
import json
import numpy as np
from transformers import AutoTokenizer
from experiment2_data import MODEL,OUT,read

def main():
    d=read(OUT/'dataset.json')
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    lengths=[]
    for start in range(0,len(d['texts']),256):
        lengths.extend(map(len,tokenizer(d['texts'][start:start+256],truncation=False)['input_ids']))
    lengths=np.array(lengths)
    audit={}
    for split,rows in d['examples'].items():
        audit[split]={}
        for role in ['q','p','n']:
            value=lengths[[r[role] for r in rows]]
            audit[split][role]={'mean_raw_tokens':float(value.mean()),'mean_capped_tokens':float(np.minimum(value,192).mean()),
                               'truncated_examples':int((value>192).sum()),'examples':len(value)}
    (OUT/'tokenization_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
