"""Fail-closed NVML monitor plus cooperative layer/step throttling.

This is a monitored soft limit, not a hardware utilization cap. It never changes
driver settings, power limits, clocks, permissions, or other applications.
"""
import json
import os
from pathlib import Path
import threading
import time

class SafetyStop(RuntimeError):pass

def classify(sample):
    if sample['gpu']>=90:return 'stop','gpu_utilization_at_least_90'
    if sample['temperature']>=78:return 'stop','gpu_temperature_at_least_78C'
    if sample['memory_used']>=6.2*1024**3:return 'stop','gpu_memory_above_6.2GiB'
    if sample['gpu']>=85:return 'pause','gpu_utilization_at_least_85'
    if sample['temperature']>=70:return 'pause','gpu_temperature_at_least_70C'
    if sample['memory_used']>=5.8*1024**3:return 'pause','gpu_memory_above_5.8GiB'
    return 'run',None

class GpuGuard:
    def __init__(self,path,layer_sleep=.02,control_root=None):
        if control_root is None:
            from prepare import OUT
            control_root=OUT
        control_root=Path(control_root);control_root.mkdir(parents=True,exist_ok=True)
        self.global_stop=control_root/'STOP';self.safety_stop=control_root/'SAFETY_STOP.json'
        if self.global_stop.exists() or self.safety_stop.exists():
            raise SafetyStop('Persistent stop marker exists; no automatic GPU restart permitted')
        import pynvml as nv
        self.nv=nv;nv.nvmlInit();self.handle=nv.nvmlDeviceGetHandleByIndex(0)
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True)
        self.layer_sleep=layer_sleep;self.state='starting';self.reason=None
        self.latest={};self.peak=0;self.hooks=[];self.quit=threading.Event()
        self.thread=threading.Thread(target=self._monitor,daemon=True)
        self.thread.start()
        deadline=time.monotonic()+5
        while not self.latest and time.monotonic()<deadline:time.sleep(.05)
        if not self.latest:raise SafetyStop('NVML monitor did not initialize')
        self.wait()

    def _monitor(self):
        last_write=0
        with (self.path/'gpu_samples.jsonl').open('a',encoding='utf-8',buffering=1) as log:
            while not self.quit.is_set():
                try:
                    u=self.nv.nvmlDeviceGetUtilizationRates(self.handle)
                    m=self.nv.nvmlDeviceGetMemoryInfo(self.handle)
                    t=self.nv.nvmlDeviceGetTemperature(self.handle,self.nv.NVML_TEMPERATURE_GPU)
                    s={'unix_time':time.time(),'gpu':int(u.gpu),'temperature':int(t),'memory_used':int(m.used),'memory_total':int(m.total)}
                    state,reason=classify(s)
                    if self.global_stop.exists():state,reason='stop','user_requested_stop'
                    self.peak=max(self.peak,s['gpu']);self.latest=s
                    if self.state=='pause' and state=='run' and (s['gpu']>60 or s['temperature']>65):state,reason='pause',self.reason
                    self.state,self.reason=state,reason
                    status={'state':state,'reason':reason,'peak_utilization':self.peak,**s}
                    log.write(json.dumps(status)+'\n')
                    if time.monotonic()-last_write>.5 or state=='stop':
                        (self.path/'gpu_status.json').write_text(json.dumps(status,indent=2));last_write=time.monotonic()
                    if state=='stop':
                        # Safety overrides checkpoint completion; the parent runner
                        # observes exit 90 and does not restart the GPU job.
                        self.safety_stop.write_text(json.dumps(status,indent=2))
                        log.flush();os._exit(90)
                except Exception as exc:
                    self.state='stop';self.reason='monitor_failure:'+repr(exc)
                    (self.path/'gpu_status.json').write_text(json.dumps({'state':'stop','reason':self.reason}))
                    self.safety_stop.write_text(json.dumps({'state':'stop','reason':self.reason}))
                    os._exit(91)
                self.quit.wait(.1)

    def wait(self):
        start=time.monotonic()
        while self.state in ['starting','pause']:
            if time.monotonic()-start>60:raise SafetyStop('Paused for 60 seconds: '+str(self.reason))
            time.sleep(.1)
        if self.state=='stop':raise SafetyStop(str(self.reason))

    def pulse(self,minimum=None):
        import torch
        self.wait()
        if torch.cuda.is_initialized():torch.cuda.synchronize()
        time.sleep(self.layer_sleep if minimum is None else minimum)
        self.wait()

    def attach(self,model):
        count=0
        for name,module in model.named_modules():
            if 'DecoderLayer' in type(module).__name__ or 'EncoderLayer' in type(module).__name__:
                self.hooks.append(module.register_forward_pre_hook(lambda *args:self.wait()))
                self.hooks.append(module.register_forward_hook(lambda *args:self.pulse()))
                self.hooks.append(module.register_full_backward_pre_hook(lambda *args:self.pulse()))
                count+=1
        if count==0:raise SafetyStop('No Transformer-layer throttle hooks were installed')
        return count

    def close(self):
        self.quit.set();self.thread.join(timeout=2)
        for hook in self.hooks:hook.remove()
        (self.path/'gpu_summary.json').write_text(json.dumps({'peak_utilization':self.peak,'last_state':self.state,'last_sample':self.latest},indent=2))

if __name__=='__main__':
    for utilization,expected in [(0,'run'),(70,'run'),(84,'run'),(85,'pause'),(89,'pause'),(90,'stop'),(100,'stop')]:
        assert classify({'gpu':utilization,'temperature':45,'memory_used':0})[0]==expected
    assert classify({'gpu':0,'temperature':78,'memory_used':0})[0]=='stop'
    assert classify({'gpu':0,'temperature':45,'memory_used':7*1024**3})[0]=='stop'
    print('PASS: monitor threshold decisions')
