# Musical Instruments: three i2i settings, five embedding methods

**Stopped, 2026-11-03. No new training or evaluation is scheduled.**
Published artifacts: [report](../results/experiment5/report.html), [summary](../results/experiment5/summary.json), [CSV](../results/experiment5/summary.csv). Only C/D completed GR training; A stopped in epoch2 due to whole-device GPU memory >6.2GiB. B/E and both Swing settings did not run. No Recall@100 exists. C/D/A passed 40 conditioning probes in total; these are diagnostics, not retrieval metrics. The quantizer nested-directory manifest lookup was repaired before A SID generation; the earlier code and error evidence are retained in the result snapshot.

The commands below document reproduction only. Do not run them to inspect the archived results; STOP and SAFETY_STOP markers remain in the original local workspace. CPU-only artifact export: `python experiment5/export_results.py`. The original scripts reference a local model-cache path in fit_i2i.py; point MODEL to the downloaded EmbeddingGemma2 snapshot before any separately authorized reproduction.

Authorized November 2, 2026. This is an isolated experiment; experiment4_v2 remains stopped.

Data: Amazon Reviews2023 official deduplicated 5-core Musical_Instruments, 57,439 users,
24,587 items, 511,836 events. Global timestamp split, equal timestamps never separated:
409,468 train / 51,184 validation / 51,184 test. Only TRAIN events build the i2i graph.
Public product metadata is used for all catalogue items; this is a content-aware,
transductive catalogue setting, not a strictly unseen-catalogue evaluation.

Settings (lowercase a/b below denote items, not methods A/B):
1. all_cooccurrence: original uniform-user, uniform unordered-pair sampling.
2. swing_top1_top5: b is a's top-1 and a is in b's top-5; union qualifying directed
   relations into unordered positive edges.
3. swing_mutual_top1: a and b are each other's top-1.

Exact Swing uses distinct user pairs u<v, user weight 1/sqrt(|Iu|), alpha=1,
contribution w(u)w(v)/(1+|Iu intersect Iv|) to every pair of common items.
No user/item cap or approximate neighbor search. Positive scores only, deterministic
item-index tie breaking. Reference: https://arxiv.org/abs/2010.05525 .
Coverage and edge counts are saved, with no fallback to weaker relations.

Loss: normalized cosine / 0.1, symmetric multi-positive soft-target cross entropy.
Target weights are shared-TRAIN-user counts; Swing settings zero out non-eligible
edges. Other distinct batch items are negatives unless also eligible positives.
Duplicate identities are masked. 32 sampled unordered pairs / update, 2,000 updates
per trainable embedding case, seed0. Thus strict filtering also changes pair sampling
from uniform users to uniform eligible edges. These experiments compare whole recipes.
The embedding stage retains the previously used fixed-update budget; the 10-epoch
limit and validation early stopping below apply to downstream GR.

A: linear 768->128; B: QR orthonormal 768x128; C: frozen native MRL128;
D: fixed PCA128; E: real q/v LoRA r8 alpha16. C/D accept no collaborative supervision
and share one embedding/SID/GR result each across all three settings (11 unique GRs).
A/B use CPU, E uses guarded GPU gradient caching, microbatch4.

SID: same 3x256 RQ-VAE and seed for every case, five AE warmup and 20 joint epochs,
per-epoch residual KMeans refresh. Fail closed if any level uses <240 codes,
collision rate >15%, or largest full-SID group >100. Do not tune or postprocess to
force equal collision rates. Collision rate = 1 - unique_full_SIDs / item_count.

GR: scratch 8-layer, 8-head, width512, FFN2304 decoder, ordinary SID NTP;
maximum10 epochs, patience2 on full validation NTP CE, batch32, lr3e-4,
1000-step warmup and cosine schedule. Window2048 local length grouping visits every
training event once per epoch; identical order seed for all cases. Training set,
max history100 and model size are unchanged. Quality probes remain validation-only.
All 11 cases train before the costly full-test beam100 evaluation phase.

Safety: one GPU worker, existing NVML 100ms monitor, pause85%, stop90%,
temperature pause70/stop78C, total memory pause5.8/stop6.2GiB. Per-layer GR delay
0.75ms, step2ms; inference batch is selected by guarded batch8/16/32 probes,
inference layers1.5ms, chunks10ms. LoRA retains microbatch4, layers3ms,
chunks10ms. No power or
driver settings change. A safety stop is persistent, with no automatic retry.

Run using the existing environment:
`../embeddinggemma2_minitiger/.venv/Scripts/python.exe experiment5/run.py`
Stop: same command plus `--stop`. Output defaults to work/experiment5;
EXPERIMENT5_OUT selects a separate output root. report.html is refreshed by the queue.
No scientific Recall results are claimed until full test completes. The real-history
speed probe uses synthetic SID labels and cannot supply recommendation accuracy.
