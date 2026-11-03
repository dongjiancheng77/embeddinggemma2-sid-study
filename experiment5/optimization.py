"""Explicit stable-optimizer recipe for repaired runs; same NTP objective."""
import math
import torch

RECIPE='musical_swing_v1_length_grouped'

def optimizer_for(model,lr=3e-4):
    decay=[];no_decay=[]
    for name,param in model.named_parameters():
        if not param.requires_grad:continue
        (no_decay if param.ndim<2 or name=='embedding.weight' else decay).append(param)
    return torch.optim.AdamW([{'params':decay,'weight_decay':.01},
                              {'params':no_decay,'weight_decay':0.}],lr=lr,betas=(.9,.98))

def lr_at(step,total_steps,warmup=1000):
    if step<warmup:return (step+1)/warmup
    fraction=min(1.,(step-warmup)/max(1,total_steps-warmup))
    return .1+.9*.5*(1+math.cos(math.pi*fraction))

def set_lr(optimizer,base_lr,step,total_steps,warmup=1000):
    value=base_lr*lr_at(step,total_steps,warmup)
    for group in optimizer.param_groups:group['lr']=value
    return value
