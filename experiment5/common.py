import json
from pathlib import Path
import importlib.util
import os

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('EXPERIMENT5_OUT',str(ROOT/'work/experiment5'))).resolve()
def load(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/'experiment3'/filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
_io=load('experiment3_io','prepare.py')
atomic_text=_io.atomic_text;read=_io.read;write=_io.write;sha=_io.sha
GpuGuard=load('experiment3_gpu_safety','gpu_guard.py').GpuGuard

def guard(path,layer_sleep=.005):return GpuGuard(path,layer_sleep=layer_sleep,control_root=OUT)

def prevent_idle_sleep(enabled):
    import os,ctypes
    if os.name!='nt':return
    api=ctypes.WinDLL('kernel32',use_last_error=True).SetThreadExecutionState
    api.argtypes=[ctypes.c_uint];api.restype=ctypes.c_uint
    if not api(0x80000000 | (1 if enabled else 0)):raise ctypes.WinError(ctypes.get_last_error())

def save_checkpoint(path,obj):
    import torch,time
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(obj,tmp)
    deadline=__import__('time').monotonic()+3
    while True:
        try:tmp.replace(path);return
        except PermissionError:
            if time.monotonic()>deadline:raise
            time.sleep(.02)
