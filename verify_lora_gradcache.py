"""Check actual adapter gradients and gradient-cache chain rule on a small real batch."""
import json
import torch
import torch.nn.functional as F
from experiment2_train import TokenLoRA,batch_loss
from experiment2_data import OUT,read
from mini_tiger import set_seed

torch.set_num_threads(4);set_seed(17)
d=read(OUT/'dataset.json');rows=d['examples']['rec_train'][:4]
indices=[r['q'] for r in rows]+[r['p'] for r in rows]+[r['n'] for r in rows]
texts=[d['texts'][ix] for ix in indices]
model=TokenLoRA().train()
parameters=[p for p in model.parameters() if p.requires_grad]
z=F.normalize(model(texts)[:,:128],dim=-1)
loss=batch_loss(z,rows,'rec');loss.backward()
direct=[p.grad.clone() if p.grad is not None else None for p in parameters]
model.zero_grad()
with torch.no_grad():cached=F.normalize(model(texts)[:,:128],dim=-1)
leaf=cached.detach().requires_grad_(True)
loss2=batch_loss(leaf,rows,'rec');loss2.backward()
replay=F.normalize(model(texts)[:,:128],dim=-1)
torch.autograd.backward(replay,leaf.grad)
delta=max(float((p.grad-g).abs().max()) for p,g in zip(parameters,direct) if g is not None)
scale=max(float(g.abs().max()) for g in direct if g is not None)
assert abs(float(loss)-float(loss2))<1e-6
assert delta/max(scale,1e-12)<.01,(delta,scale)
assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
report={'direct_loss':float(loss),'cached_loss':float(loss2),'gradient_max_abs_difference':delta,
        'gradient_reference_max':scale,'frozen_parameter_gradients':0,'passed':True}
(OUT/'lora_gradient_verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
