"""CPU-only export of existing artifacts. Never trains or evaluates a model."""
import csv
import datetime as dt
import hashlib
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'work/experiment5'
DEST=ROOT/'results/experiment5'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,list):return [clean(v) for v in x]
    if isinstance(x,str):return x.replace(str(ROOT),'<WORKSPACE>').replace(str(Path.home()),'<USER_HOME>')
    return x
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    DEST.mkdir(parents=True,exist_ok=True);data=SOURCE/'Musical_Instruments'
    stop=read(SOURCE/'SAFETY_STOP.json');rows=[];details=[]
    cases=[('fixed',k) for k in ['mrl','pca']]+[(v,k) for k in ['linear','ortho','lora'] for v in ['all_cooccurrence','swing_top1_top5','swing_mutual_top1']]
    for variant,kind in cases:
        key=kind if variant=='fixed' else variant+'/'+kind;p=data/key;gr=p/'gr'
        def get(name,base=p):return read(base/name) if (base/name).exists() else None
        sid=get('sid.json');stage=get('embedding_stage1.json');complete=get('training_complete.json',gr);progress=get('progress.json',gr)
        epochs=[{'epoch':int(f.stem[5:]),**read(f)} for f in sorted(gr.glob('epoch*.json'),key=lambda f:int(f.stem[5:]))]
        probes=[read(f) for f in gr.glob('probe_*.json')];probes.sort(key=lambda x:x['global_step'])
        best=min(epochs,key=lambda e:e['validation_loss']) if epochs else None
        metrics=get('metrics.json',gr)
        state='trained_pending_test' if complete else 'interrupted_gpu_memory' if progress else 'embedding_only' if stage else 'not_started'
        row={'setting':variant,'method':kind,'status':state,'sid_collision_rate':sid['collision_rate'] if sid else None,
             'sid_usage':sid['usage'] if sid else None,'largest_collision_group':sid['largest_collision_group'] if sid else None,
             'completed_epochs':len(epochs),'best_epoch':best['epoch'] if best else None,'best_validation_loss':best['validation_loss'] if best else None,
             'last_observed_epoch':progress.get('epoch',progress.get('epochs')) if progress else None,
             'last_observed_step':progress.get('step') if progress else None,'probe_checks':len(probes),'failed_probes':sum(not x['passed'] for x in probes),
             'recall_at_100':metrics.get('recall@100') if metrics else None}
        samples=gr/'gpu_samples.jsonl';telemetry=[]
        if samples.exists():
            with samples.open() as f:
                for line in f:
                    try:telemetry.append(json.loads(line))
                    except json.JSONDecodeError:pass
        gpu={'samples':len(telemetry),'peak_utilization':max((x['gpu'] for x in telemetry),default=None),
             'peak_temperature':max((x['temperature'] for x in telemetry),default=None),
             'peak_total_memory_GiB':max((x['memory_used']/1024**3 for x in telemetry),default=None),
             'pause_samples':sum(x['state']=='pause' for x in telemetry),'stop_samples':sum(x['state']=='stop' for x in telemetry)}
        detail={**row,'epochs':epochs,'latest_probe':probes[-1] if probes else None,'gpu_telemetry':gpu,
                'training_complete':complete,'embedding_stage':stage,'sid':sid}
        if kind=='linear' and variant=='all_cooccurrence' and (gr/'latest.pt').exists():
            import torch
            ck=torch.load(gr/'latest.pt',map_location='cpu',weights_only=False)
            detail['saved_checkpoint']={k:ck[k] for k in ['epoch','next_step','best','bad','recipe','sid_sha256']};del ck
        rows.append(row);details.append(detail)
        for f in p.glob('*.json'):
            if f.name in ['embedding_stage1.json','gradient_audit.json','sid.json','quantizer_progress.json','quantizer_quality_failure.json']:
                write(DEST/'evidence'/key/f.name,read(f))
        for f in gr.glob('*.json'):write(DEST/'evidence'/key/'gr'/f.name,read(f))
    receipts=['run_status.json','run_provenance.json','SAFETY_STOP.json','Musical_Instruments/manifest.json',
              'raw/Musical_Instruments/source.json','raw/Musical_Instruments/complete.json','Musical_Instruments/swing/complete.json',
              'Musical_Instruments/swing/verified.json','Musical_Instruments/batching_profile.json','Musical_Instruments/real_history_speed.json',
              'Musical_Instruments/embedding_tuning.json','diagnostics/path_fix_20261103/recovery.json',
              'diagnostics/path_fix_20261103/run_provenance.json']
    for rel in receipts:
        if (SOURCE/rel).exists():write(DEST/'evidence'/rel,read(SOURCE/rel))
    for rel in ['diagnostics/path_fix_20261103/all_cooccurrence_linear_sid.log','queue.err.log']:
        p=SOURCE/rel
        if p.exists():
            target=DEST/'evidence'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(clean(p.read_text(encoding='utf-8',errors='replace')),encoding='utf-8')
    inventory=[]
    for kind in ['mrl','pca','all_cooccurrence/linear']:
        for name in ['sid.npy','items128.npy','gr/best.pt','gr/latest.pt']:
            p=data/kind/name
            if p.exists():inventory.append({'path':'Musical_Instruments/'+kind+'/'+name,'bytes':p.stat().st_size,'sha256':digest(p),'uploaded':False})
    summary={'exported_at':dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),'status':'user_halted_partial_experiment',
      'no_new_training_or_evaluation':True,'dataset':read(data/'manifest.json')['counts'],'split':read(data/'manifest.json')['splits'],
      'gpu_safety_stop':stop,'safety_stop_time_Asia_Shanghai':dt.datetime.fromtimestamp(stop['unix_time'],dt.timezone(dt.timedelta(hours=8))).isoformat(),
      'unique_planned_cases':11,'trained_cases':sum(x['status']=='trained_pending_test' for x in rows),'test_evaluations':sum(x['recall_at_100'] is not None for x in rows),
      'cases':details,'local_artifact_inventory':inventory,
      'conclusions':['No Recall@100 exists; recommendation effectiveness and A/B/E/Swing rankings remain unknown.',
        'Among the three available SIDs, native MRL has the lowest excess collision rate.',
        'PCA has lower validation NTP loss but different SID targets prevent direct recommendation ranking.',
        'A stopped in epoch2; do not compare its partial GR with converged C/D.',
        'Strict Swing covers only18.7% or5.9% of catalogue items; those settings have not been trained.',
        'Memory safety observes whole-device memory; the artifacts do not identify which process caused the spike.']}
    write(DEST/'summary.json',summary)
    with (DEST/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for item in details:
        if item['epochs']:
            axes[0].plot([x['epoch'] for x in item['epochs']],[x['validation_loss'] for x in item['epochs']],marker='o',label=item['method']+(' (incomplete)' if item['status'].startswith('interrupted') else ''))
    axes[0].set(xlabel='Completed epoch',ylabel='Validation SID NTP loss',title='Training diagnostics, not recommendation ranking');axes[0].legend()
    available=[x for x in rows if x['sid_collision_rate'] is not None]
    axes[1].bar([x['method'] for x in available],[100*x['sid_collision_rate'] for x in available]);axes[1].set(ylabel='Excess collision rate (%)',title='Same 3 x 256 codebook capacity')
    fig.tight_layout();fig.savefig(DEST/'diagnostics.png',dpi=160);plt.close(fig)
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(x[k]))+'</td>' for k in ['setting','method','status','sid_collision_rate','best_epoch','best_validation_loss'])+'</tr>' for x in rows)
    body='''<!doctype html><meta charset="utf-8"><title>Musical Instruments：阶段性实验结果</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto;line-height:1.7}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}img{max-width:100%}.notice{background:#fff3cd;padding:18px}</style><h1>Musical Instruments：阶段性实验结果</h1><p class="notice">实验已停止，不再启动新的训练或评测。本轮没有正式 Recall@100，不支持对五种方案或三种 i2i 设置作推荐效果排名。</p><h2>已完成与中断情况</h2><p>C：原生 MRL 在第7轮早停，最佳第5轮；D：PCA在第8轮早停，最佳第6轮。A：无top-k线性头完成i2i和SID，GR在第2轮因显存安全阈值中断。B/E和两种Swing设置均未开始。</p><table><tr><th>设置</th><th>方案</th><th>状态</th><th>SID碰撞率</th><th>最佳轮次</th><th>验证损失</th></tr>'''+table+'''</table><h2>结果分析</h2><p>三个已生成SID每层均使用256/256编码，未见码本塌缩。C的碰撞率2.38%，A为2.78%，D为6.39%；最大冲突组分别5、7、8。这里碰撞率定义为1−唯一完整SID数量/商品数量。</p><p>PCA的最佳验证损失3.4865低于MRL的3.5931，但SID分布、熵及碰撞率不同，较低交叉熵不等于更高推荐命中率。A仅完成一轮验证，不能据此比较收敛表现。C/D/A共40次条件依赖检查全部通过：置换历史或错误SID前缀均使损失增大，这说明模型使用了条件信息，仍不能替代检索评测。</p><h2>中断原因与安全</h2><p>11月2日23:10曾因新增目录后清单路径读取错误停止；11月3日人工检查修复后继续。11月3日08:49再次因整张GPU显存约6.55 GiB超过6.2 GiB阈值停止，当时利用率14%、温度65°C；累计利用率峰值89%。没有证据能从现有记录确定显存上升属于哪个进程，不能直接断言训练显存泄漏。安全标记与用户停止标记均保留，未重启。</p><h2>预算与解释边界</h2><p>数据为Amazon2023官方5-core乐器：57,439用户，24,587商品，511,836交互。全局时间划分409,468/51,184/51,184；商品元数据完整。C/D为固定参照，不接收i2i监督；不是五个方案等监督的结构消融。Swing只使用训练期数据：top1/top5有2,827对，覆盖18.7%商品；互为top1有723对，覆盖5.9%。三个设置的比较尚未执行。</p><p>GR实际每轮含验证约10分钟；C共71.6分钟，D共81.5分钟。短程真实历史测速约2.33倍提速，使用临时SID，仅用于速度诊断。A/B/E每个embedding设置计划2,000更新；GR最多10轮、patience2。所有测试原计划排在训练队列之后，因此尚未产生测试指标。</p><img src="diagnostics.png" alt="验证损失曲线与SID碰撞率"><p>机器可读结果：<a href="summary.json">summary.json</a> / <a href="summary.csv">summary.csv</a>。代码见仓库experiment5；证据见evidence。商品交互、用户标识、商品向量和模型权重未上传，权重在本机保留并记录校验和。</p>'''
    (DEST/'report.html').write_text(body,encoding='utf-8')
    hashes={p.relative_to(DEST).as_posix():digest(p) for p in DEST.rglob('*') if p.is_file() and p.name!='SHA256.json'};write(DEST/'SHA256.json',hashes)
    print(json.dumps({'cases':len(rows),'completed':summary['trained_cases'],'test_evaluations':summary['test_evaluations'],'all_checks':sum(x['probe_checks'] for x in rows),'failed_checks':sum(x['failed_probes'] for x in rows),'safety_time':summary['safety_stop_time_Asia_Shanghai']},ensure_ascii=False))

if __name__=='__main__':main()
