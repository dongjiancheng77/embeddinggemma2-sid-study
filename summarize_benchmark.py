"""Aggregate raw seed/user outputs; report paired user-bootstrap uncertainty."""
import csv
import html
import itertools
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent/'work/benchmark_20261029'
LABEL={'linear':'A 线性头','ortho':'B 正交矩阵','mrl':'C 原生 MRL'}

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    seeds=read(ROOT/'manifest.json')['config']['seeds']
    timings=[json.loads(line) for line in (ROOT/'timings.jsonl').read_text().splitlines()]
    records=[]
    users={}
    for seed,kind in itertools.product(seeds,LABEL):
        d=ROOT/f'seed{seed}'
        result=read(d/f'{kind}_test.json')
        head=read(d/f'{kind}_128.head_stats.json')
        rq=read(d/f'{kind}_sid.stats.json')
        records.append({'seed':seed,'method':kind,'recall10':result['item_level_recall'],
                        'ndcg10':result['item_level_ndcg'],'proxy_val_recall20':head['proxy_val_recall20'],
                        'proxy_val_ndcg20':head['proxy_val_ndcg20'],
                        'neighbor_overlap20':head['continuous_neighbor_overlap20'],
                        'rq_recon':rq['recon_loss'],'collision_excess':rq['base_collision_items'],
                        'suffix_vocab':rq['suffix_vocab'],
                        'usage1':rq['codebook_stage_usage'][0],'usage2':rq['codebook_stage_usage'][1],
                        'usage3':rq['codebook_stage_usage'][2],'head_parameters':head['trainable_parameters'],
                        'head_seconds':next(t['seconds'] for t in timings if t['seed']==seed and t['kind']==kind and t['stage']=='head'),
                        'total_seconds':sum(t['seconds'] for t in timings if t['seed']==seed and t['kind']==kind),
                        'orthogonality_error':head.get('orthogonality_max_error')})
        users[seed,kind]=read(d/f'{kind}_test.users.json')
    with (ROOT/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    means={}
    for kind in LABEL:
        rows=[r for r in records if r['method']==kind]
        means[kind]={m:{'mean':float(np.mean([r[m] for r in rows])),
                         'sd':float(np.std([r[m] for r in rows],ddof=1))} for m in rows[0] if m not in ['method','seed','orthogonality_error']}
    comparisons=[]
    rng=np.random.default_rng(20261029)
    for a,b in itertools.combinations(LABEL,2):
        for metric in ['recall','ndcg']:
            diffs=[]
            for seed in seeds:
                ra,rb=users[seed,a],users[seed,b]
                assert [r['user'] for r in ra]==[r['user'] for r in rb]
                diffs.append(np.array([r[metric] for r in ra])-np.array([r[metric] for r in rb]))
            d=np.mean(diffs,axis=0)
            boots=np.array([d[rng.integers(len(d),size=len(d))].mean() for _ in range(5000)])
            comparisons.append({'a':a,'b':b,'metric':metric,'mean_difference':float(d.mean()),
                                'paired_user_ci95':np.quantile(boots,[.025,.975]).tolist(),
                                'per_seed_differences':[float(x.mean()) for x in diffs]})
    (ROOT/'aggregate.json').write_text(json.dumps({'means':means,'comparisons':comparisons,
        'uncertainty_note':'Paired bootstrap resamples users after averaging seed results; it does not quantify training-seed population uncertainty.'},indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
    for ax,metric,title in zip(axs,['recall10','ndcg10'],['Item Recall@10','Item NDCG@10']):
        ax.bar(['A Linear','B Ortho','C MRL'],[means[k][metric]['mean']*100 for k in LABEL],
               yerr=[means[k][metric]['sd']*100 for k in LABEL],capsize=5,color=['#3978ad','#c17730','#658e55'])
        ax.set_title(title);ax.set_ylabel('% (mean + seed SD)');ax.spines[['top','right']].set_visible(False)
    fig.savefig(ROOT/'comparison.png',dpi=180)
    rows=''
    for k in LABEL:
        v=means[k]
        fmt=lambda m:f"{v[m]['mean']*100:.3f} ± {v[m]['sd']*100:.3f}%"
        rows+=f"<tr><td>{LABEL[k]}</td><td>{fmt('recall10')}</td><td>{fmt('ndcg10')}</td><td>{fmt('neighbor_overlap20')}</td><td>{v['collision_excess']['mean']:.0f}</td><td>{v['head_parameters']['mean']:.0f}</td></tr>"
    pairs=''.join(f"<li>{LABEL[c['a']]} − {LABEL[c['b']]}，{c['metric']}：{c['mean_difference']*100:+.3f} 个百分点，配对用户 95% CI [{c['paired_user_ci95'][0]*100:+.3f}, {c['paired_user_ci95'][1]*100:+.3f}]；各种子差值 {np.round(np.array(c['per_seed_differences'])*100,3).tolist()}</li>" for c in comparisons)
    data=read(ROOT.parent/'beauty.json')
    winner=max(LABEL,key=lambda k:means[k]['ndcg10']['mean'])
    ab=next(c for c in comparisons if c['a']=='linear' and c['b']=='ortho' and c['metric']=='ndcg')
    ab_note='A 与 B 的 NDCG 配对区间跨过 0，本轮不足以确认二者谁更好。' if ab['paired_user_ci95'][0]<=0<=ab['paired_user_ci95'][1] else 'A 与 B 的 NDCG 配对区间未跨过 0；优势范围仅适用于当前数据、用户样本和训练预算。'
    cost_note='；'.join(f"{LABEL[k]}：头阶段 {means[k]['head_seconds']['mean']:.1f} 秒，整条流水线 {means[k]['total_seconds']['mean']:.1f} 秒" for k in LABEL)
    doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>EmbeddingGemma 2 本机 A/B/C 实验</title>
<style>body{{max-width:1050px;margin:40px auto;font:16px/1.7 system-ui;color:#223}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}img{{width:100%}}code{{background:#eee}}a{{color:#286fa0}}</style>
<h1>EmbeddingGemma 2：线性头、正交矩阵与 MRL</h1>
<p>2026-10-29 · 本机 RTX 5070 Laptop · 三个随机种子 {seeds}</p>
<p>以本轮预先指定的商品级 NDCG@10 均值排序，领先的是 <b>{LABEL[winner]}</b>。{ab_note}</p>
<p>{cost_note}。时间为三种子平均，含独立进程启动、诊断与磁盘保存，不含模型下载和一次性的文本缓存。</p>
<img src="comparison.png"><table><tr><th>方案</th><th>Recall@10</th><th>NDCG@10</th><th>语义邻居保留@20</th><th>重复三元码超额商品数</th><th>头参数数</th></tr>{rows}</table>
<p>± 是三个训练种子的样本标准差。重构误差、代理推荐分数、各层码本利用率等逐种子详情见 <a href="summary.csv">summary.csv</a>。</p>
<h2>对比口径</h2><p>数据为 Amazon 2014 Beauty 5-core，保留评分 ≥4、去除重复商品、元数据匹配后序列长度 ≥5。实际 {len(data['users'])} 位用户、{len(data['items'])} 件商品。每用户最后两次交互分别作为验证和测试，其余用于训练；同时间按 ASIN 排序。文本只含商品标题、品牌、类别和描述。切分不是全局时间因果切分；商品文本目录具有传导性。</p>
<p>固定 EmbeddingGemma 2 文本骨干，768 维缓存复用。A：带偏置线性头；B：QR 参数化的 128 个严格正交方向；C：原生 MRL 前 128 维再归一化，无新增训练。A/B 使用相同训练窗口、负样本流、优化器和 3 个 epoch 的推荐对比监督，以验证代理 NDCG@20 选头。A/B 与 C 的对比包含额外监督效果；B 是仅矩阵的正交读出实验，不能称为原论文 SpecMRL 的完整复现。</p>
<p>每个方法独立拟合同架构 RQ-VAE（128→64→32，3×256，先预热自编码器 10 epochs 再做码本初始化与量化训练 15 epochs），加显式碰撞后缀构成四个 token。后缀词表大小因碰撞不同，会改变推荐器少量参数。相同 GRU 推荐器训练 5 epochs，验证 CE 选模型，beam=30，合法目录前缀约束。固定抽样相同 1000 位用户进行全目录 SID 搜索，过滤已见商品，无抽样候选负例；beam 为近似搜索。</p>
<h2>配对统计</h2><ul>{pairs}</ul><p>区间先对同一用户的三个种子取均值，再重采样用户 5000 次；只衡量用户样本不确定性，不代表完整训练随机性。相同用户必须在所有方法逐一对齐。</p>
<h2>如何理解优劣</h2><p>A 有任意缩放、非正交混合和偏置，表达能力强；B 约束保留方向正交，但每步 QR 带来训练开销；C 无头训练成本、直接使用预训练语义结构。最终推荐成绩优先于语义邻居保留率或 RQ 重构误差；不同表示空间的重构误差不能单独视为全维语义失真。此轮是小规模方法筛选，固定短训练预算和小型 GRU，尚不能外推至完整 TIGER/T5 或其他数据集。</p>
<p>来源：<a href="https://huggingface.co/docs/transformers/model_doc/embedding_gemma2">官方模型实现</a>；<a href="https://arxiv.org/abs/2610.26824">SpecMRL</a>。模型固定版本、数据/代码校验和与运行环境见 <a href="manifest.json">manifest.json</a>；各阶段耗时见 <a href="timings.jsonl">timings.jsonl</a>；逐用户预测和训练日志保存在 seed 目录。</p></html>'''
    (ROOT/'report.html').write_text(doc,encoding='utf-8')
    print(json.dumps({'means':means,'comparisons':comparisons},indent=2))

if __name__=='__main__':main()
