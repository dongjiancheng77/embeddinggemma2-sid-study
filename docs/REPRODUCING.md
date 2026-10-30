# 复现步骤

## 1. 只核对发布结果

`results/` 已包含汇总、逐种子指标、匿名逐用户评分、图表和原始核验记录。匿名用户编号在所有方法、设置和随机种子间保持一致。

```powershell
python check_published_results.py
python reproduce_report.py
```

第二条命令直接使用发布快照重算第二轮 CSV、均值、标准差、5,000 次用户配对 bootstrap 和图表，无需 GPU、模型权重或原始数据。需要安装相应 Python 库；为避免结果文件被覆盖，应先保留副本或在 Git 工作区核对差异。

## 2. 准备独立运行环境

实测环境为 Windows、Python 3.12、RTX 5070 Laptop 8 GB。以下命令在仓库根目录执行；不需要修改系统执行策略。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-experiment2.lock.txt
.\.venv\Scripts\python.exe -m pip check
```

完整重新训练只写入默认忽略的 `work/`，不会覆盖 `results/` 快照。不要把历史 `run_manifest.json` 拷进新的 `work/`：其中的代码/数据哈希对应原始机器，新的训练会生成自己的清单。

## 3. 固定本地模型

先按 Google 的模型使用流程下载 `google/embeddinggemma-2` 的固定 revision `f9972ac22e696a105c482202646979e33e4b5a1e`，然后指向实际本地 snapshot 文件夹：

```powershell
$env:EMBEDDINGGEMMA2_MODEL='D:\models\embeddinggemma-2\snapshots\f9972ac22e696a105c482202646979e33e4b5a1e'
$env:HF_HUB_OFFLINE='1'
```

这里是路径示例，需要改成本机实际目录。程序保持 `local_files_only=True`；不设置变量时使用已缓存的 `google/embeddinggemma-2`。为严格控制版本，应使用完整 snapshot 路径。GPU 用 BF16，不使用 FP16；CPU 回读核验用 FP32。未使用的音频/图像塔通过配置关闭，原生池化、任务前缀与 512→768 投影保留。

## 4. 下载与预处理 Beauty

```powershell
.\.venv\Scripts\python.exe download_data.py --out data
.\.venv\Scripts\python.exe mini_tiger.py prepare --reviews data/reviews_Beauty_5.json.gz --metadata data/meta_Beauty.json.gz --out work/beauty.json --min-rating 4
.\.venv\Scripts\python.exe mini_tiger.py embed --model $env:EMBEDDINGGEMMA2_MODEL --data work/beauty.json --out work/full768.npy --batch-size 16 --device cuda
.\.venv\Scripts\python.exe mini_tiger.py smoke
```

原始文件是 Amazon 2014 Beauty；不要替换成 2018/2023 All Beauty。用户—商品重复交互保留最早时间，时间相同按 ASIN 排序。评分过滤后不再保证每个商品满足原始 5-core。

## 5. 下载 SpecMRL 使用的普通文本数据

```powershell
.\.venv\Scripts\python.exe download_kalm.py
New-Item -ItemType Directory -Path sources -Force | Out-Null
Copy-Item provenance/spec-mrl-revision.json sources/spec-mrl-revision.json
.\.venv\Scripts\python.exe experiment2_data.py prepare
.\.venv\Scripts\python.exe experiment2_data.py cache
.\.venv\Scripts\python.exe experiment2_token_audit.py
.\.venv\Scripts\python.exe verify_lora_gradcache.py
```

下载器固定 HF revision，校验官方 LFS SHA256，从 Quora/SNLI/MS MARCO 指定文件按固定随机种子抽样。8,192 条训练和 768 条验证按实际使用的查询、正例、负例文本去重。源文件含更多负例，本轮每条只使用一个显式负例，并在批次中共享。文件路径、源行号和清单留在 `data/kalm`，不会自动上传。

`sources/spec-mrl-revision.json` 仅提供已记录的参考代码来源元信息，无需执行第三方代码。普通文本训练数据来自 KaLM，不来自该代码压缩包。

## 6. 完整第二轮实验

```powershell
.\.venv\Scripts\python.exe experiment2_run.py
.\.venv\Scripts\python.exe experiment2_verify.py
.\.venv\Scripts\python.exe experiment2_reload_check.py
.\.venv\Scripts\python.exe experiment2_summary.py
```

第一阶段：

| 设置 | 推荐样本 | 普通文本样本 | 更新次数 | 可训练方法 |
|---|---:|---:|---:|---|
| `collab` | 8,192 | 0 | 256 | A/B/E |
| `mixed` | 8,192 | 8,192 | 512 | A/B/E |

A/B 学习率 `3e-4`，E 为 `1e-4`，AdamW weight decay=`1e-4`、有效 batch=`32`、温度=`0.1`、梯度裁剪=`1`，固定一轮。LoRA `r=8, alpha=16, dropout=0`，全部文本层 `q_proj/v_proj`，微批次 32 梯度缓存重算。C/D 不做第一阶段梯度更新，PCA 只拟合全目录商品内容。所有方法独立从原模型开始。

然后采用相同 RQ-VAE 和 GRU 配置。结果写入 `work/experiment2/{setting}/seed{0,1,2}/{kind}`。`complete.json` 标识完整条件；固定 C/D 在两个设置间复用。运行器会跳过已完成阶段，并检查注册清单中的源代码/输入哈希；修改代码或数据后应新开输出目录，不应混用旧产物。

GPU 训练可能受精度、算子和驱动影响而无法逐位复现，尤其小的浮点差异可能影响 KMeans、离散 SID 和后续训练。应比较多种子的结果分布，不能仅要求最终分数逐位相同。

## 7. 第一轮实验

第一轮使用历史投影向量均值作为上下文，A/B 训练最多 60,000 个窗口、3 epochs，和第二轮历史文本查询／8,192 窗口的一轮协议不同。

```powershell
.\.venv\Scripts\python.exe benchmark_local.py
.\.venv\Scripts\python.exe verify_benchmark.py
.\.venv\Scripts\python.exe summarize_benchmark.py
```

第一轮结果写在 `work/benchmark_20261029`，发布快照位于 `results/experiment1`。两轮均使用同配置 RQ-VAE、GRU 和相同抽样测试用户，但不可直接将跨轮分数差归因于某一项数据变化。
