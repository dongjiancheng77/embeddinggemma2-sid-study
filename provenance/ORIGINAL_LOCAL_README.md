# EmbeddingGemma 2 → 128d → RQ-VAE → Mini-TIGER

## Experiment 2: five methods and two supervision settings

`experiment2_run.py` runs A=`linear`, B=`ortho`, C=`mrl`, D=`pca`, E=`lora` with paired seeds 0/1/2. C/D are fixed references and are reused exactly across `collab` and `mixed`, as requested. This experiment uses a new common query/document first-stage objective and is reported separately from the earlier mean-history experiment.

The fixed first-stage recommendation sample has 8,192 training-only windows. Up to ten historical product titles form a native search query; the next product content is the positive. `mixed` alternates those same recommendation batches with 8,192 general text examples from the official KaLM dataset used by SpecMRL. Selected English source shards are Quora, SNLI, and MS MARCO; 768 text examples are held out with no raw text overlap. This is a sampled dataset adaptation, not the paper's full million-example recipe. Download source hashes and rows are in `data/kalm/manifest.json`; the official repository archive is fixed to commit `59db2450bc1cf386f30413a3de32dbe82b47bbd0`.

First-stage training is one fixed epoch, effective batch 32, temperature 0.1, 31 other positive documents plus 32 shared explicit negatives per batch, with positive duplicates and seen recommendation items masked. A/B update only their heads. E updates rank-8 LoRA adapters on all text attention `q_proj` and `v_proj` modules, uses real differentiable token forward passes, BF16, and microbatch-32 gradient caching. Its output uses the native 128-dimensional prefix. The original backbone weights remain frozen. Mixed training has twice the optimizer updates but identical recommendation exposure. Native MRL stays frozen and PCA fits centered, unwhitened all-item content only.

All methods then receive the same RQ-VAE and GRU training settings as Experiment 1. The GRU still uses up to 60,000 training windows and five epochs. All 30 conditions use the same 1,000 held-out test users. General-text retrieval validation is reported separately to measure semantic retention. Exact data, code hashes, packages, and budgets are in `work/experiment2/run_manifest.json`.

Commands after activating the local environment:

```powershell
python download_kalm.py
python experiment2_data.py prepare
python experiment2_data.py cache
python verify_lora_gradcache.py
python experiment2_run.py
python experiment2_verify.py
python experiment2_summary.py
```

The final report is `work/experiment2/report.html`, with raw per-seed `summary.csv`, paired-user confidence intervals in `aggregate.json`, adapters/checkpoints, per-user predictions, and stage logs under each setting/seed/method. General data caches and model snapshots are shared; each trained method/setting starts from the original frozen model rather than another run's adapter. The older results below remain available independently.

## Local GPU experiment (2026-10-29)

Real-data A/B/C runs use the downloaded fixed EmbeddingGemma 2 snapshot, an isolated `.venv`, and RTX 5070 Laptop CUDA. A=`linear`, B=`ortho`, C=`mrl`. Seeds are 0/1/2; the same 1,000 test users (sampling seed 42) are paired across all nine runs. The processed Beauty dataset contains 14,993 users, 12,069 items, and 130,455 interactions.

Run `./.venv/Scripts/python.exe benchmark_local.py` after prepare/embed, then `./.venv/Scripts/python.exe verify_benchmark.py` and `./.venv/Scripts/python.exe summarize_benchmark.py`. The runner resumes existing stages. Results are in [the report](work/benchmark_20261029/report.html), [raw summary](work/benchmark_20261029/summary.csv), and [provenance manifest](work/benchmark_20261029/manifest.json). Exact installed versions are in `requirements-local.lock.txt`; install CUDA Torch/Torchvision from the official cu128 index first.

Local fixes include serializable preprocessing configuration, GPU-compatible validation indices, exact validation sample limits, matched sampling streams after head initialization, FP32 cache normalization, and per-user predictions. All RQ-VAEs now receive **10 identical autoencoder warmup epochs before centroid initialization**, followed by 15 quantized epochs, to avoid codebook collapse observed during preflight. Failed preflight outputs are retained in `work/preflight_20261029` and excluded from the report. Head training remains 3 epochs and generator training 5 epochs. This is a fixed short-budget screen, not a convergence claim. The original portability/synthetic-test description below describes the incoming bundle before these local experiments.

A runnable **small-scale methodology baseline** for comparing:

- `mrl`: pretrained 768d embedding prefix truncated to 128d, L2-normalized; **no new head training**.
- `linear`: frozen backbone + learned affine `Linear(768,128)`, initialized to the same MRL prefix.
- `ortho`: frozen backbone + learned `768×128` **exact orthonormal-column** projection (`torch.linalg.qr`), initialized near the MRL prefix.

All three use the **same behavioral next-item supervision for the two trainable heads**, the same 128→32 RQ-VAE architecture, the same 3×256 codebook structure, the same collision disambiguation procedure, and the same small autoregressive SID generator.

**What it is not:** This is *not* a faithful SpecMRL reproduction, not a SOTA TIGER training recipe, and not an experiment demonstrating any improvement on Amazon. No actual model weights or Amazon data were downloaded for the tests in this environment. The code passed an artificial-input smoke test and a complete artificial-data train/quantize/generate/evaluate test; real-data results remain to be run locally.

## Hardware and data

- RTX 5070 Laptop (typically 8GB VRAM): only load **EmbeddingGemma 2 text tower (around 270M)** in **bfloat16**, then cache item embeddings; never train/fine-tune it in this baseline.
- Download [Amazon Beauty 2014 5-core](https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/reviews_Beauty_5.json.gz) and [Beauty product metadata](https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/meta_Beauty.json.gz).
- Raw subset: 198,502 reviews; actual number of samples changes after the optional rating≥4 filter, deduplicating users/items, and metadata matching.
- Product text uses **title, brand, categories, description only**, not review body, user review, rating average, also-bought/also-viewed, or other behavior-derived metadata.
- Each user: sort by review timestamp, remove repeat item reviews, keep sequences length≥5, use all but the last two events for training; penultimate event is validation, last event is test. Exact timestamp ties are broken by ASIN. Review timestamp is a coarse and noisy proxy for true consumption order, so treat ranking scores as an initial methodology check, not a publication-level result.

## Installation (Python 3.11 recommended)

From this folder:

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
```

First install a **compatible CUDA-enabled PyTorch** using the selector at https://pytorch.org/get-started/locally/ (for Blackwell/RTX 5070, use a build that supports SM120; avoid outdated cu11x wheels). Then:

```bash
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
python mini_tiger.py smoke
```

If `torch.cuda.is_available()` is false, stop and repair the PyTorch + NVIDIA driver installation first. EmbeddingGemma 2 is **not compatible with fp16 activation**; use BF16 or FP32.

## Four commands for first experiment

All commands are single-line and work in PowerShell, CMD, or Bash. Replace model path with your *downloaded* model directory containing configuration, tokenizer, and checkpoint files. Use `--batch-size 8` if GPU memory is tight.

```bash
python download_data.py --out data
python mini_tiger.py prepare --reviews data/reviews_Beauty_5.json.gz --metadata data/meta_Beauty.json.gz --out work/beauty.json --min-rating 4
python mini_tiger.py embed --model D:/models/embeddinggemma-2 --data work/beauty.json --out work/full768.npy --batch-size 8 --device cuda
python run_all.py --data work/beauty.json --embeddings work/full768.npy --out work/experiment1 --device cuda --eval-users 1000
```

The third command executes the frozen text-only EmbeddingGemma 2 once per item (batch inference, FP32 output). It caches a `(n_items, 768)` float32 matrix. Every subsequent stage only loads that matrix, so comparing three heads does not repeat expensive model inference.

If your file bundle already contains the 768d numpy matrix in exactly the same order as `work/beauty.json` → `items`, skip `embed` and supply that file via `--embeddings`.

### Outputs

```text
work/experiment1/
  mrl_128.npy,    linear_128.npy,    ortho_128.npy
  *_128.head_stats.json           # neighbor overlap + proxy validation metrics
  linear_128.pt, ortho_128.pt     # trained head parameters
  *_sid.npy                       # [n_items, 4] 3 RQ codes + collision suffix
  *_sid.stats.json                # RQ reconstruction, code usage, collisions
  *_sid.pt                        # RQ-VAE checkpoint
  *_gen.pt                        # small autoregressive generator checkpoint
  *_test.json                     # true item-level Recall/NDCG
  summary.csv                     # primary cross-method comparison
```

### What is trained?

**Head stage.** The two learned heads optimize the same next-item contrastive cross-entropy. A context is the normalized average of the **previous up to 10** projected item embeddings. Candidates are the actual next item versus 20 random negatives, with accidental history/positive collisions masked. Training uses up to 60,000 next-item windows, batch 256, 3 epochs, and validation proxy NDCG@20 selects the head checkpoint. MRL baseline is not trained, because this experiment tests whether a trainable low-dimensional readout beats the pretrained MRL prefix. This does not isolate parameterization from extra supervision; **A vs B** is controlled, while **MRL vs A/B** includes adaptation effects.

Note: This is a lightweight recommendation supervision **baseline**, not the proposed novel compression-distortion loss. It still consumes user behavior. Replace this loss in follow-up experiments if you want to test loss-specific novelty.

**RQ-VAE stage.** Independently for each 128d item representation, `encoder=128→64→32`, **three** residual nearest-centroid codebooks of 256×32 each, `decoder=32→64→128`. The first codebook is initialized by MiniBatchKMeans on encoder latents; the subsequent codebooks initialize on residuals. Model uses straight-through reconstruction gradient, codebook and commitment losses. Hyperparameters/seed are equal across methods. VAE is not pretrained on full 768d and does not see user labels.

**Unique SID.** The three RQ codewords are a nominal 24-bit prefix. Different items can share all three codes. For every colliding group, append deterministic `collision_index` as a fourth token (0 for all non-collisions). This makes item-level evaluation unambiguous, but **adds a suffix vocabulary** whose size is explicitly reported. Code collision is not silently counted as a hit.

**Generator stage.** The user's latest 10 items are replaced by their four-token SIDs and flattened. A GRU encodes that sequence, a second GRU autoregressively generates the four tokens of the next item's SID. Each stage uses a separate token range; validation teacher-forced cross-entropy selects the generator checkpoint. There are about **400k trainable parameters** (actual count depends on suffix vocabulary). For inference, trie-constrained beam search (`beam=30` by default) produces only catalog-valid codes; items already in history are filtered. It is deliberately simpler than TIGER's T5; the experiment is a feasible methodological screen.

**Testing stage.** Evaluate up to 1,000 **deterministically sampled** users by default (same subset across methods). For each test user, the observed history includes the penultimate validation event, as in chronological leave-one-out testing. Return item Recall@10/Hit@10 and NDCG@10 (single test positive), **full catalog with no sampled candidate negatives**. Use `--eval-users 0` for all available users, once the pipeline is stable. Report seeds and confidence intervals before interpreting small differences.

## Important scientific controls

- Use **identical user splits, item catalog and encoder inputs** across all methods; do not include reviews/rating/co-purchases in the item content text.
- Use the same RQ-VAE *architecture*, hyperparameters, seeds and all-item clustering corpus. **Each method refits its own RQ-VAE**; that is a fair comparison for replacing the SID generator, but is *not* a fixed-codebook deployment comparison.
- Do **not** interpret RQ reconstruction, codebook utilization or 128d cosine retrieval as final generative recommendation results.
- Record collision suffix vocabulary and full item-level evaluation. The suffix changes actual SID length/complexity relative to an exactly-three-token system; it is applied equally by rule across all methods.
- The simple RQ-VAE does not implement a balancing objective, adaptive schedules or independent random-IDs baseline. If results look interesting, add random SID + full-768→RQ-VAE + PCA128 baselines, 3+ random seeds and a T5/Transformer generator. Also evaluate behavior-free content tests before concluding semantic preservation.
- Candidate cold-start items whose labels exist only in validation/test may still be in the metadata catalog and the RQ-VAE training set (content available). This is **transductive item metadata**, not future user interaction leakage; report cold/warm results separately in a serious experiment.

## Further experiments

1. Replace the head's next-item InfoNCE with a **fixed-teacher recommendation decision distortion** target, so recommendation supervision changes which information is preserved during compression rather than merely repeating i2i ranking.
2. Add `PCA128` and `EmbeddingGemma2 full768→RQ-VAE` baselines to distinguish reader value from downstream quantizer encoder capacity.
3. Train 128d head jointly with a differentiable surrogate for RQ code decisions, then evaluate both with a fresh independently trained RQ-VAE and with the joint one. This tests whether representation has intrinsic quantization quality, rather than relying on a specific co-adapted codebook.

## Sources

- EmbeddingGemma 2 official model card: https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2
- Text-only 270M and BF16 model usage: https://huggingface.co/docs/transformers/model_doc/embedding_gemma2
- Amazon original Beauty 5-core: https://mcauleylab.ucsd.edu/public_datasets/data/amazon/index_snap.html
- TIGER architecture: https://arxiv.org/abs/2305.05065
