"""One-GPU experiment queue. C/D are shared fixed references across settings."""
import datetime
import html
import os
import subprocess
import sys
import time
from common import ROOT,OUT,read,write,sha,atomic_text,prevent_idle_sleep
from swing import VARIANTS

def stopped():return any((OUT/f).exists() for f in ['STOP','SAFETY_STOP.json','QUALITY_STOP.json'])

def publish(state):
    write(OUT/'run_status.json',state);rows=[];data=OUT/'Musical_Instruments'
    for v in VARIANTS:
        for k in ['linear','ortho','mrl','pca','lora']:
            path=data/(k if k in ['mrl','pca'] else v+'/'+k)
            def load(f):return read(path/f) if (path/f).exists() else {}
            sid=load('sid.json');p=load('gr/progress.json');metrics=load('gr/metrics.json')
            rows.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [v,k,'共享固定参照' if k in ['mrl','pca'] else '独立训练',sid.get('usage',''),sid.get('collision_rate',''),p.get('state','pending'),p.get('epoch',p.get('epochs','')),p.get('step',''),metrics.get('recall@100','')])+'</tr>')
    swing=read(data/'swing/complete.json') if (data/'swing/complete.json').exists() else {}
    atomic_text(OUT/'report.html','<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30"><style>body{font:16px system-ui;margin:32px}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}pre{white-space:pre-wrap}</style><h1>Musical Instruments：三种 i2i 设置 × 五种方案</h1><p>全局时间80/10/10；GR最多10 epoch，patience=2。C/D共享。全部训练完成后统一做beam100全量测试。</p><pre>'+html.escape(__import__('json').dumps(state,ensure_ascii=False,indent=2))+'</pre><table><tr><th>设置</th><th>方案</th><th>口径</th><th>每层激活/256</th><th>SID碰撞率</th><th>GR状态</th><th>epoch</th><th>step</th><th>Recall100</th></tr>'+''.join(rows)+'</table><h2>Swing覆盖率</h2><pre>'+html.escape(str(swing.get('variants',{})))+'</pre>')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if '--stop' in sys.argv:(OUT/'STOP').write_text('user requested stop');return
    if stopped():raise RuntimeError('Persistent stop marker')
    lock=OUT/'RUNNING.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    source={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'experiment5').glob('*.py')}
    source['experiment3/gpu_guard.py']=sha(ROOT/'experiment3/gpu_guard.py')
    write(OUT/'run_provenance.json',{'started':datetime.datetime.now().isoformat(),'sources':source,
      'dataset':'Amazon Reviews2023 official Musical_Instruments 5core before split','variants':VARIANTS,
      'fixed_references_shared':True,'GR_cases':11,'GR':{'max_epochs':10,'patience':2,'batch':32,'length_grouping_window':2048,'layer_sleep':.00075,'step_sleep':.002,'beam':100},
      'embedding':{'updates':2000,'pairs_per_update':32,'temperature':.1,'same_budget_for_A_B_E':True},
      'SID':{'levels':3,'codebook_sizes':[256]*3,'min_usage':240,'max_collision_rate':.15,'max_group':100,'force_equal_collisions':False},
      'GPU':{'pause':85,'stop':90,'automatic_retry':False}})
    env={**os.environ,'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    child=None;state={};done=[];start=time.time();dataset='Musical_Instruments';data=OUT/dataset;raw=OUT/'raw'/dataset
    def execute(label,args,marker,gpu=False):
        nonlocal child,state
        if marker.exists():done.append(label);return
        if stopped():raise RuntimeError('Stop requested')
        if any(sha(ROOT/p)!=v for p,v in source.items()):raise RuntimeError('Source changed after start')
        state={'state':'running','phase':label,'completed':done.copy(),'parent_pid':os.getpid(),'started_unix':start}
        logpath=OUT/(label.replace('/','_')+'.log')
        with logpath.open('a',encoding='utf-8') as log:
            taskenv=env.copy()
            if not gpu:taskenv['CUDA_VISIBLE_DEVICES']=''
            child=subprocess.Popen([sys.executable,str(ROOT/'experiment5'/args[0]),*args[1:]],cwd=ROOT,env=taskenv,stdout=log,stderr=subprocess.STDOUT)
            state.update(child_pid=child.pid,log=str(logpath));print('START',label,child.pid,flush=True)
            while child.poll() is None:
                if stopped():child.terminate();raise RuntimeError('Stop marker observed; no automatic retry')
                state['updated_unix']=time.time();publish(state);time.sleep(3)
            if child.returncode:raise RuntimeError(f'{label} exited {child.returncode}; no automatic retry')
        if not marker.exists():raise RuntimeError('Missing marker '+str(marker))
        done.append(label);print('DONE',label,flush=True)
    try:
        prevent_idle_sleep(True)
        while not (raw/'complete.json').exists():
            if stopped():raise RuntimeError('Stop requested')
            state={'state':'running','phase':'download','parent_pid':os.getpid(),'started_unix':start,'updated_unix':time.time()};publish(state);time.sleep(3)
        execute('prepare',['prepare.py','--dataset',dataset,'--events',str(raw/'events.csv.gz'),'--source-manifest',str(raw/'source.json')],data/'manifest.json')
        execute('swing',['swing.py'],data/'swing/complete.json')
        execute('verify_swing',['verify_swing.py'],data/'swing/verified.json')
        execute('embedding_tuning',['profile_embedding.py'],data/'embedding_tuning.json',True)
        execute('cache',['cache_catalog.py','--dataset',dataset,'--catalog',str(raw/'catalog.jsonl')],data/'embedding_cache/complete.json',True)
        execute('batching_profile',['profile_batching.py'],data/'batching_profile.json')
        cases=[('all_cooccurrence',k) for k in ['mrl','pca']]+[(v,k) for k in ['linear','ortho','lora'] for v in VARIANTS]
        for variant,kind in cases:
            key=kind if kind in ['mrl','pca'] else variant+'/'+kind;path=data/key
            execute(key+'/i2i',['fit_i2i.py','--dataset',dataset,'--kind',kind,'--variant',variant],path/'embedding_stage1.json',kind=='lora')
            execute(key+'/sid',['quantize.py','--dataset',dataset,'--kind',key],path/'sid.json')
            execute(key+'/train',['train_gr.py','--dataset',dataset,'--kind',key,'--epochs','10','--patience','2','--train-only'],path/'gr/training_complete.json',True)
        for variant,kind in cases:
            key=kind if kind in ['mrl','pca'] else variant+'/'+kind
            execute(key+'/test',['train_gr.py','--dataset',dataset,'--kind',key,'--epochs','10','--patience','2','--eval-only'],data/key/'gr/complete.json',True)
        publish({'state':'complete','completed':done,'seconds':time.time()-start})
    except BaseException as exc:
        if child is not None and child.poll() is None:child.terminate();child.wait(timeout=10)
        state.update(state='stopped' if stopped() else 'failed',error=repr(exc),updated_unix=time.time());publish(state);raise
    finally:prevent_idle_sleep(False);lock.unlink(missing_ok=True)

if __name__=='__main__':main()
