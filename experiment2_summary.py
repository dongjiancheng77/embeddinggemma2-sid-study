"""Five-method/two-setting paired comparison with semantic-holdout diagnostics."""
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from experiment2_data import ROOT,OUT,read

METHODS={'linear':'A 线性头','ortho':'B 正交矩阵','mrl':'C 冻结 MRL','pca':'D PCA','lora':'E LoRA'}
SETTINGS={'collab':'仅协同','mixed':'普通文本＋协同'}

def main():
    rows=[];users={}
    timings=[json.loads(line) for line in (OUT/'timings.jsonl').read_text().splitlines()]
    for setting,seed,kind in itertools.product(SETTINGS,[0,1,2],METHODS):
        p=OUT/setting/f'seed{seed}'/kind
        result=read(p/'test.json');stage=read(p/'stage1.json');rq=read(p/'sid.stats.json')
        rows.append({'setting':setting,'seed':seed,'method':kind,'recall10':result['item_level_recall'],
                     'ndcg10':result['item_level_ndcg'],'text_val_ndcg10':stage['text_validation_ndcg10'],
                     'text_val_recall10':stage['text_validation_recall10'],'rec_proxy_val_ndcg10':stage['rec_validation_ndcg10'],
                     'parameters':stage['parameters'],'train_seconds':stage['train_seconds'],'peak_GB':stage['peak_bytes']/1e9,
                     'stage1_seconds':next(t['seconds'] for t in timings if t['seed']==seed and t['kind']==kind and t['stage']=='stage1' and t['setting']==('collab' if setting=='mixed' and kind in ['mrl','pca'] else setting)),
                     'rq_recon':rq['recon_loss'],'collision_excess':rq['base_collision_items'],'suffix_vocab':rq['suffix_vocab'],
                     **{f'usage{i+1}':n for i,n in enumerate(rq['codebook_stage_usage'])}})
        users[setting,seed,kind]=read(p/'test.users.json')
    with (OUT/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    means={}
    for setting in SETTINGS:
        means[setting]={}
        for kind in METHODS:
            group=[r for r in rows if r['setting']==setting and r['method']==kind]
            means[setting][kind]={m:{'mean':float(np.mean([r[m] for r in group])),
                                    'sd':float(np.std([r[m] for r in group],ddof=1))} for m in group[0] if m not in ['setting','seed','method']}
    comparisons=[];rng=np.random.default_rng(20261029)
    def compare(a,b,metric):
        diffs=[]
        for seed in [0,1,2]:
            ra=users[a[0],seed,a[1]];rb=users[b[0],seed,b[1]]
            assert [r['user'] for r in ra]==[r['user'] for r in rb]
            diffs.append(np.array([r[metric] for r in ra])-np.array([r[metric] for r in rb]))
        d=np.mean(diffs,axis=0)
        bs=[float(d[rng.integers(len(d),size=len(d))].mean()) for _ in range(5000)]
        comparisons.append({'a':list(a),'b':list(b),'metric':metric,'difference':float(d.mean()),
                            'paired_user_ci95':np.quantile(bs,[.025,.975]).tolist(),
                            'per_seed_differences':[float(x.mean()) for x in diffs]})
    for setting in SETTINGS:
        for a,b in itertools.combinations(METHODS,2):
            for metric in ['recall','ndcg']:compare((setting,a),(setting,b),metric)
    for kind in METHODS:
        for metric in ['recall','ndcg']:compare(('mixed',kind),('collab',kind),metric)
    aggregate={'means':means,'comparisons':comparisons,'uncertainty':'Seed SD reported separately. Paired-user bootstrap averages the three seed outcomes per user then resamples users, and does not cover training-seed population uncertainty.'}
    (OUT/'aggregate.json').write_text(json.dumps(aggregate,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=np.arange(5);labels=['A Linear','B Ortho','C MRL','D PCA','E LoRA']
    for filename,metrics,titles in [('recommendation.png',['recall10','ndcg10'],['SID item Recall@10','SID item NDCG@10']),
                                    ('semantic.png',['text_val_ndcg10','rec_proxy_val_ndcg10'],['Held-out text retrieval NDCG@10','Stage-1 recommendation proxy NDCG@10'])]:
        fig,axs=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
        fig.suptitle('Mean ± SD across 3 seeds',fontsize=10)
        for ax,metric,title in zip(axs,metrics,titles):
            for j,(setting,color,label) in enumerate([('collab','#638095','Behavior only'),('mixed','#cf8538','Text + behavior')]):
                ax.bar(x+(j-.5)*.36,[means[setting][k][metric]['mean'] for k in METHODS],width=.36,
                       yerr=[means[setting][k][metric]['sd'] for k in METHODS],capsize=3,color=color,label=label)
            ax.set_xticks(x,labels);ax.set_title(title);ax.spines[['top','right']].set_visible(False)
            ax.legend(fontsize=8);ax.set_ylim(bottom=0)
        fig.savefig(OUT/filename,dpi=180);plt.close(fig)
    tables=''
    for setting in SETTINGS:
        body=''
        for kind in METHODS:
            m=means[setting][kind]
            fmt=lambda key:f"{m[key]['mean']:.5f} ± {m[key]['sd']:.5f}"
            body+=f"<tr><td>{METHODS[kind]}</td><td>{fmt('recall10')}</td><td>{fmt('ndcg10')}</td><td>{fmt('text_val_ndcg10')}</td><td>{m['parameters']['mean']:.0f}</td><td>{m['stage1_seconds']['mean']:.1f}</td></tr>"
        tables+=f"<h2>{SETTINGS[setting]}</h2><table><tr><th>方法</th><th>Recall@10</th><th>NDCG@10</th><th>文本验证 NDCG@10</th><th>训练参数</th><th>第一阶段总秒数</th></tr>{body}</table>"
    deltas=''
    for kind in ['linear','ortho','lora']:
        c=next(c for c in comparisons if c['a']==['mixed',kind] and c['b']==['collab',kind] and c['metric']=='ndcg')
        sem=means['mixed'][kind]['text_val_ndcg10']['mean']-means['collab'][kind]['text_val_ndcg10']['mean']
        ci=c['paired_user_ci95']
        interpretation='区间跨过零，尚不能确认推荐收益' if ci[0]<=0<=ci[1] else ('当前用户样本区间支持正向变化' if ci[0]>0 else '当前用户样本区间支持负向变化')
        deltas+=f"<li>{METHODS[kind]}：混合 − 仅协同，SID NDCG@10 差值 {c['difference']:+.5f}，配对用户 95% CI [{ci[0]:+.5f}, {ci[1]:+.5f}]；各训练种子差值 {np.round(c['per_seed_differences'],5).tolist()}。{interpretation}。文本检索 NDCG 变化 {sem:+.5f}。</li>"
    winners='；'.join(f"{SETTINGS[s]}的 NDCG 均值最高为 {METHODS[max(METHODS,key=lambda k:means[s][k]['ndcg10']['mean'])]}" for s in SETTINGS)
    b_gain=(means['mixed']['ortho']['ndcg10']['mean']/means['collab']['ortho']['ndcg10']['mean']-1)*100
    mixed_be=next(c for c in comparisons if c['a']==['mixed','ortho'] and c['b']==['mixed','lora'] and c['metric']=='ndcg')
    be_ci=mixed_be['paired_user_ci95']
    doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>EmbeddingGemma 2 五方案 × 两种监督设置</title>
<style>body{{max-width:1180px;margin:36px auto;padding:0 20px;font:16px/1.75 system-ui;color:#203040}}table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;border-bottom:1px solid #d7dfe4;text-align:left}}th{{background:#eef3f5}}img{{width:100%}}a{{color:#22668c}}.note{{background:#f4f6f8;padding:16px}}</style>
<h1>EmbeddingGemma 2：五种方案与两种监督设置</h1><p>运行日期 2026-10-29—30 · RTX 5070 Laptop 8 GB · 种子 0、1、2 · 30 个条件，C/D 固定参照跨设置复用</p>
<p>{winners}。排名仅为当前短预算筛选的结果，须结合种子波动和配对区间解释。</p>
<p class="note"><b>本轮建议优先保留 B＋普通文本混合训练。</b>它的推荐 NDCG 相对自身仅协同版本提高 {b_gain:.1f}%，三个种子均为正向变化，同时保持较好的 128 维文本检索质量。LoRA 在仅协同设置下的 NDCG 均值略高于 B，但混合设置下没有显示出稳定优势，且训练成本显著更高。混合 B − E 的 NDCG 配对用户区间为 [{be_ci[0]:+.5f}, {be_ci[1]:+.5f}]，包含零，因此不能声称 B 已被证明优于 LoRA。加入普通文本对 A/B/E 的语义检索均有正向均值变化，但最终推荐收益并不普遍成立。</p>
<img src="recommendation.png">{tables}<p>表内 ± 和图中误差线均是三个训练种子的样本标准差。第一阶段总秒数包括进程启动、模型初始化、拟合或优化、商品向量导出及诊断，不包含下游训练。共享冻结 768 维向量缓存的首次生成成本未计入表内；LoRA 适配后的重新编码计入其各次运行。C/D 的第一阶段不做梯度更新，D 仍有 PCA 拟合成本。原始 CSV 另外列出优化阶段耗时。</p>
<h2>加入普通文本带来了什么</h2><ul>{deltas}</ul><p>C/D 在两个设置中完全相同是预先确认的实验设计，不是遗漏训练。它们用于提供无任务适配的参照，不能声称它们接受了与 A/B/E 相同的监督。</p>
<img src="semantic.png"><p>普通文本检索使用独立的 768 条查询和 1,536 条正/负候选，无训练文本交叉。该小型验证任务不是 MTEB 全量测试。第一阶段推荐代理分数使用历史标题查询与连续商品向量打分，不是最终 SID 推荐成绩。</p>
<h2>数据与第一阶段训练</h2><p>商品与行为沿用 Amazon 2014 Beauty：14,993 位用户、12,069 件商品、130,455 次去重交互，评分 ≥4。用户最后两次交互分作验证和测试。推荐训练仅从训练区间取固定 8,192 个窗口，历史最多 10 件商品的标题按先后顺序组成查询，正例为下一商品内容。文本只含标题、品牌、类别和描述，不含评论、评分统计或共购字段。</p>
<p>普通文本来自 SpecMRL 指定的 <a href="https://huggingface.co/datasets/KaLM-Embedding/KaLM-embedding-finetuning-data">KaLM-embedding-finetuning-data</a>。下载并校验了 quora、snli、msmarco-passage 的选定官方文件，训练抽样数分别为 3,072、2,048、3,072，总计 8,192；每来源另留 256 条验证。固定数据 revision 为 e9443ab6f5d4dc29c79cea03834e932428ed6ab1。训练/验证按原始查询、正例和使用的负例文本去重。保留一条源困难负例，原文件中其他负例仍保留在 Parquet 下载中。本实验不是论文的 1,152,000 样本完整复现。</p>
<p>A/B 冻结骨干，训练 768→128 线性头／QR 正交矩阵；C 使用原生 MRL 前 128 维；D 对全商品冻结内容向量拟合带中心化、不白化的 PCA128；E 在原 EmbeddingGemma 2 文本塔全部 24 层的 q_proj、v_proj 上训练 rank=8、alpha=16、dropout=0 的 LoRA，最后取 128 维前缀。E 的原始模型权重冻结，适配器参数约 52 万。</p>
<p>A/B/E 使用同一查询—文档对比目标：有效 batch=32，31 个其他正例加该批次共享的 32 个显式负例；屏蔽重复正例和已见商品。温度 0.1，AdamW、weight decay=0.0001、梯度裁剪 1.0，A/B 学习率 0.0003，E 为 0.0001。第一阶段固定一轮，使用最终状态，无测试集选优。仅协同 256 次更新；混合设置交替执行推荐和普通文本批次，共 512 次更新，因此推荐样本曝光量相同但总计算量不同。</p>
<p>E 使用 BF16 token-forward 和微批次 32 的梯度缓存重算，不使用无梯度 encode 来训练。已与直接反向传播核对损失和梯度，并检查非零适配器梯度、原权重无梯度。模型固定 revision f9972ac22e696a105c482202646979e33e4b5a1e，原生池化与投影保留，最大 192 tokens。每条第一阶段路线独立从相同原模型开始，混合设置不会接着仅协同检查点训练。</p>
<p>输入长度审计：推荐训练查询 2,206/8,192 条触及 192-token 截断，推荐正例 248/8,192 条；普通文本查询无截断，正例 55/8,192 条。历史查询按从旧到新排列，默认右截断可能丢失较晚的历史标题，是本轮查询协议的局限，不能把“最多 10 件标题”视为全部均被编码。所有可训练方法输入一致。完整统计见 <a href="tokenization_audit.json">tokenization_audit.json</a>。</p>
<h2>相同的 SID 与生成式推荐设置</h2><p>第一阶段完成后冻结表示，分别训练同架构 RQ-VAE：128→64→32，3 层×256 码字，10 轮 AE 预热和 15 轮量化训练。追加碰撞后缀得到唯一的四 token SID，后缀词表大小因方法而异，推荐器参数量因此有少量差别。GRU 生成式推荐器使用同一训练窗口抽样规则、历史长度 10、最多 60,000 窗口、5 epochs、相同优化器，以验证 CE 选模型。第一阶段与下游重复使用训练行为，不使用验证/测试行为训练。C/D 所有下游检查点也跨设置复用。</p>
<p>最终对所有条件固定相同 1,000 位用户，beam=30、有效 SID 前缀约束，在全商品目录中近似搜索并过滤历史商品。测试历史包含已观测的验证事件。没有抽样候选负例。静态商品目录与 PCA/RQ-VAE 拟合具有传导性，用户内时间切分不等于全局严格时间因果协议。</p>
<h2>结论的边界</h2><p>本轮可回答加入普通文本后的实际流水线变化，以及轻量读出与 LoRA 在当前预算下的表现。混合设置额外获得一倍优化步骤，收益不能完全归因为数据类型；C/D 没有同等监督；LoRA 的训练参数和计算成本高于 A/B。B 是直接正交矩阵，不是 SpecMRL 的参数谱完整复现。短固定预算与单一 Beauty 数据、小型 GRU 不能支持 SOTA 或充分收敛结论。配对 bootstrap 先对同一用户的三个种子取均值，再重采样用户 5,000 次，区间只描述用户样本不确定性，不覆盖训练种子总体不确定性；多个配对区间属于探索性比较，未做多重检验校正。本轮使用历史标题文本查询和一轮适配的新协议，与上一轮历史向量均值、不同训练预算的结果单独保存，不宜直接跨轮解释差值。</p>
<p>来源：<a href="https://github.com/CentreChen/spec-mrl">SpecMRL 官方代码</a>（下载版本 59db2450bc1cf386f30413a3de32dbe82b47bbd0）；<a href="https://arxiv.org/html/2610.26824v1#A1">论文数据说明</a>；<a href="https://huggingface.co/docs/transformers/model_doc/embedding_gemma2">模型官方实现</a>。</p>
<p>产物：<a href="summary.csv">逐种子汇总 CSV</a> · <a href="aggregate.json">均值与配对区间</a> · <a href="run_manifest.json">配置、版本与校验和</a> · <a href="verification.json">结果核验</a> · <a href="adapter_reload_verification.json">六个适配器回读核验</a> · <a href="timings.jsonl">实际耗时</a> · <a href="../../data/kalm/manifest.json">下载数据清单</a>。适配器、读出层、PCA、SID、GRU、逐用户预测和日志在各设置/seed/方法目录中。</p></html>'''
    (OUT/'report.html').write_text(doc,encoding='utf-8')
    print(json.dumps(aggregate,indent=2))

if __name__=='__main__':main()
