from html import escape
import time
from common import OUT,read,atomic_text

def render(status):
    cards=[]
    for dataset in ['Beauty','Movies_and_TV','Books']:
        manifest=OUT/dataset/'manifest.json'
        detail='等待下载与预处理'
        if manifest.exists():
            m=read(manifest);detail=f"{m['counts']['events']:,} 次交互 · {m['counts']['items']:,} 件商品 · 全局时间 80/10/10"
        rows=[]
        for method,name in [('mrl','C 原生 MRL'),('ortho','B 正交投影'),('linear','A 线性头'),('pca','D PCA'),('lora','E EmbeddingGemma2 LoRA')]:
            root=OUT/dataset/method;done=(root/'gr/complete.json').exists()
            text='已完成' if done else '未完成'
            if (root/'gr/metrics.json').exists():text=f"Recall@100 = {read(root/'gr/metrics.json')['recall@100']:.4%}"
            if (root/'sid.json').exists():text+=f"；SID碰撞 {read(root/'sid.json')['collision_rate']:.2%}"
            rows.append(f'<li>{name}：{text}</li>')
        cards.append(f'<article><h2>{dataset}</h2><p>{detail}</p><ul>'+''.join(rows)+'</ul></article>')
    progress=''
    if status.get('phase')=='assets':
        p=OUT/'raw'/status['dataset']/'download_progress.json'
        if p.exists():
            d=read(p);progress=f"下载 {d['file']}：{d['bytes']/1024**2:.1f} / {d.get('total',0)/1024**2:.1f} MiB"
        p=OUT/'raw'/status['dataset']/'normalize_progress.json'
        if p.exists():progress+='；'+str(read(p))
    log=status.get('log','')
    root=OUT/status.get('dataset','')/status.get('kind','')
    for path in [root/'gr/progress.json',root/'progress.json',OUT/status.get('dataset','')/'embedding_cache/progress.json']:
        if path.exists():progress+=' '+str(read(path))
    state_label={'paused':'已按用户要求暂停','running':'运行中','complete':'已完成','stopped':'已停止','failed':'运行失败'}.get(status.get('state'),status.get('state','准备中'))
    diagnostic=''
    if (OUT/'repair_validation.json').exists():
        repair=read(OUT/'repair_validation.json');q=repair['quantizer'];g=repair['gr_preflight']
        diagnostic=f'<div class="box"><h2>修复版验证已通过</h2><p>独立目录保存旧结果，重新构建SID后从随机初始化训练GR。码字使用 {q["usage"]}，碰撞率 {q["collision_rate"]:.2%}；仍保留三段SID及自然碰撞，不强行把碰撞改为零。</p><p>3,000步真实数据验证：正常历史损失 {g["loss"]["history"]:.3f}，替换历史 {g["loss"]["other_history"]:.3f}，破坏目标前缀 {g["loss"]["wrong_prefix"]:.3f}，频率基线 {g["marginal_frequency_loss"]:.3f}。这是恢复训练的质量检查，不是最终推荐指标。</p><p>修复内容：输入尺度平衡；LayerNorm/词元向量不做weight decay；1,000步warmup与余弦学习率衰减。后续每5,000步验证损失和条件敏感性，连续失败两次即停止。</p></div>'
    if (OUT/'diagnostics/pause_review/summary.json').exists():
        diagnostic='<div class="box"><h2>暂停期间的 CPU 排查</h2><p>已复现码本联合训练失效，并发现 GR 对历史和目标前缀利用不足。原始数据和检查点已保留，GPU训练未恢复。</p><p><a href="diagnostics/pause_review/report.html">查看完整排查报告与证据</a></p></div>'
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="20"><title>实验四：基础 GR 与无序 i2i</title>
<style>body{{font:16px/1.6 system-ui,"Microsoft YaHei";background:#f4f6fa;color:#192a40}}main{{max-width:1100px;margin:32px auto;padding:20px}}article,.box{{background:white;border:1px solid #d9e1ec;border-radius:12px;padding:20px;margin:16px 0}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}}small{{color:#526078}}</style><main>
<h1>实验四：基础 GR × 无序 i2i</h1><div class="box"><b>{escape(state_label)}</b><p>{escape(status.get('dataset',''))} / {escape(status.get('phase',''))} / {escape(status.get('kind',''))}</p><p>{escape(progress)}</p><p>{escape(status.get('error',''))}</p></div>{diagnostic}
<div class="grid">{''.join(cards)}</div><div class="box"><h2>独立复现口径</h2><p>从头训练 8 层、8 heads、hidden512、FFN2304 的 27,666,944 参数基础 GR。普通 NTP，三段 SID，KV-cache beam100；不使用 MiniOneRec、Qwen 推荐底座或 SPROUT 额外损失。</p>
<p>公开版本固定为 Amazon Reviews2023 官方 5-core（Books、Movies and TV、Beauty and Personal Care），重新按全局时间切为 80/10/10。作者未公开代码和数据处理，因此实际规模不保证与论文表1相同，逐数据集保留实际计数和差异。</p>
<p>A/B/E 的表示监督只来自训练期完整用户商品集合的无序共现：所有距离的商品对均可采样，没有历史文本查询和下一商品的特殊角色。C/D 固定；下游 GR 仍使用按时间排列的历史预测下一商品。</p>
<p>全量训练集逐轮训练，验证损失选模型，完整测试切分报告 Recall@100；历史商品保留，输出去重。GPU 达85%暂停、达90%停止，温度/显存保护保持启用。</p><small>状态更新时间 {time.strftime('%Y-%m-%d %H:%M:%S')}，页面每20秒刷新。{escape(log)}</small></div></main></html>'''
    atomic_text(OUT/'report.html',text)
