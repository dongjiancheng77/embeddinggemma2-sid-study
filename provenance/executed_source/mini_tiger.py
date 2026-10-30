#!/usr/bin/env python3
"""A lightweight EmbeddingGemma2 -> (MRL/Linear/Ortho) -> RQ-VAE -> AR-SID experiment.

Only the `embed` subcommand imports sentence-transformers. All other stages
operate on cached numpy embeddings; the pretrained model never gets gradients.
"""
import argparse
import ast
import collections
import gzip
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested='auto'):
    return 'cuda' if requested == 'auto' and torch.cuda.is_available() else ('cpu' if requested == 'auto' else requested)


def read_records(path):
    # Amazon 2014 exports include both JSON-lines and Python-dict-like lines.
    open_fn = gzip.open if str(path).endswith('.gz') else open
    with open_fn(path, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield ast.literal_eval(line)


def as_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ' '.join(as_text(x) for x in value)
    return '' if value is None else str(value)


def prep(args):
    set_seed(args.seed)
    # Review event is treated as an implicit preference (optionally rating-filtered).
    interactions = collections.defaultdict(dict)
    review_count = 0
    raw_review_count = 0
    for obj in read_records(args.reviews):
        raw_review_count += 1
        rating = float(obj.get('overall', 0))
        if rating < args.min_rating:
            continue
        user, asin = obj.get('reviewerID'), obj.get('asin')
        if not user or not asin:
            continue
        timestamp = int(obj.get('unixReviewTime', 0))
        previous = interactions[user].get(asin)
        if previous is None or timestamp < previous:
            interactions[user][asin] = timestamp
        review_count += 1
    eligible = {}
    relevant_asins = set()
    for user, items in interactions.items():
        seq = [asin for asin, _ in sorted(items.items(), key=lambda r: (r[1], r[0]))]
        if len(seq) >= args.min_user_len:
            eligible[user] = seq
            relevant_asins.update(seq)
    meta = {}
    for obj in read_records(args.metadata):
        asin = obj.get('asin')
        if asin not in relevant_asins:
            continue
        title = as_text(obj.get('title', '')).strip()
        brand = as_text(obj.get('brand', '')).strip()
        categories = obj.get('categories', [])
        if isinstance(categories, (list, tuple)) and categories:
            categories = categories[0] if isinstance(categories[0], (list, tuple)) else categories
        category_text = as_text(categories).strip()
        desc = as_text(obj.get('description', '')).strip()
        # Explicitly exclude reviews, related/co-bought, sales ranks and ratings.
        doc = f'title: {title or "Beauty product"} | text: brand: {brand}; category: {category_text}; description: {desc}'
        meta[asin] = {'asin': asin, 'text': doc[:args.max_chars], 'title': title}
    known = set(meta)
    sequences = {}
    for user, seq in eligible.items():
        seq = [x for x in seq if x in known]
        if len(seq) >= args.min_user_len:
            sequences[user] = seq
    asins = sorted({x for seq in sequences.values() for x in seq})
    id_map = {asin: i for i, asin in enumerate(asins)}
    catalog = [meta[asin] for asin in asins]
    users = [dict(user=u, seq=[id_map[x] for x in sequences[u]]) for u in sorted(sequences)]
    obj = {'items': catalog, 'users': users, 'config': {k:v for k,v in vars(args).items() if k!='fn'}}
    obj['audit'] = {'raw_review_events':raw_review_count,'positive_review_events':review_count,
                    'eligible_users_before_metadata':len(eligible),'retained_users':len(users),
                    'retained_items':len(catalog),'missing_metadata_items':len(relevant_asins-known),
                    'empty_titles':sum(not item['title'] for item in catalog),
                    'retained_interactions':sum(len(u['seq']) for u in users),
                    'train_interactions':sum(len(u['seq'])-2 for u in users),
                    'split_sha256':hashlib.sha256(json.dumps(users,sort_keys=True).encode()).hexdigest()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(obj, ensure_ascii=False), encoding='utf-8')
    print(f"PREP: {len(users)} users; {len(catalog)} items; {review_count} positive review events before other filters.")
    print('NOTE: legacy 5-core describes the *raw* subset; applying rating and metadata filters changes the population.')


def embed(args):
    from sentence_transformers import SentenceTransformer
    device = get_device(args.device)
    dtype = torch.bfloat16 if device.startswith('cuda') and torch.cuda.is_bf16_supported() else torch.float32
    if dtype == torch.float32:
        print('WARNING: BF16 unavailable. Using FP32; never use FP16 for EmbeddingGemma 2.')
    data = json.loads(Path(args.data).read_text(encoding='utf-8'))
    model = SentenceTransformer(
        args.model,
        local_files_only=True,
        device=device,
        config_kwargs={'vision_config': None, 'audio_config': None},
        model_kwargs={'torch_dtype': dtype},
    )
    model.max_seq_length = args.max_tokens
    texts = [item['text'] for item in data['items']]
    probe = texts[:3]
    one = model.encode(probe, batch_size=1, normalize_embeddings=True, convert_to_numpy=True)
    batched = model.encode(probe, batch_size=3, normalize_embeddings=True, convert_to_numpy=True)
    consistency = float(np.max(np.abs(one.astype('float32')-batched.astype('float32'))))
    assert consistency < 0.03, consistency
    token_lengths = []
    for start in range(0,len(texts),256):
        token_lengths.extend(map(len, model.tokenizer(texts[start:start+256], truncation=False)['input_ids']))
    # 'title: ... | text: ...' follows Google's model card. Avoid prepending a
    # second Document prompt that would inject an additional "title: none".
    arr = model.encode(
        texts, batch_size=args.batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True,
    ).astype('float32')
    arr /= np.linalg.norm(arr, axis=1, keepdims=True).clip(1e-12)
    assert arr.shape == (len(texts), 768), arr.shape
    assert np.isfinite(arr).all(), 'Non-finite embeddings; ensure BF16/FP32 rather than FP16.'
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, arr)
    Path(args.out).with_suffix('.metadata.json').write_text(json.dumps({
        'model':args.model,'device':device,'dtype':str(dtype),'max_tokens':args.max_tokens,
        'batch_size':args.batch_size,'shape':list(arr.shape),'finite':bool(np.isfinite(arr).all()),
        'batch_consistency_max_abs':consistency,'truncated_items':sum(x>args.max_tokens for x in token_lengths),
        'token_length_max':max(token_lengths),'token_length_mean':float(np.mean(token_lengths)),
        'norm_min':float(np.linalg.norm(arr,axis=1).min()),'norm_max':float(np.linalg.norm(arr,axis=1).max()),
        'gpu':torch.cuda.get_device_name(0) if device.startswith('cuda') else None,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated() if device.startswith('cuda') else None,
    },indent=2),encoding='utf-8')
    print(f'EMBED: {arr.shape} -> {args.out}; dtype={dtype}; device={device}')


def load_data(data_path):
    return json.loads(Path(data_path).read_text(encoding='utf-8'))


def sample_train_windows(users, hist_len, max_windows, seed):
    windows = []
    for u in users:
        seq = u['seq'][:-2]
        for t in range(2, len(seq)):
            windows.append((seq[max(0, t-hist_len):t], seq[t]))
    rng = random.Random(seed)
    if max_windows and len(windows) > max_windows:
        windows = rng.sample(windows, max_windows)
    rng.shuffle(windows)
    if not windows:
        raise ValueError('No train windows. Check the data filtering and min_user_len.')
    return windows


def val_windows(users, hist_len, test=False):
    out = []
    for u in users:
        seq = u['seq']
        prefix = seq[:-1] if test else seq[:-2]
        tgt = seq[-1] if test else seq[-2]
        out.append((prefix[-hist_len:], tgt))
    return out


def pack_windows(windows, hist_len):
    h = np.full((len(windows), hist_len), -1, np.int64)
    y = np.zeros(len(windows), np.int64)
    for k, (hist, tgt) in enumerate(windows):
        hs = hist[-hist_len:]
        h[k, -len(hs):] = hs
        y[k] = tgt
    return torch.from_numpy(h), torch.from_numpy(y)


class Head(nn.Module):
    def __init__(self, kind, dim=768, out=128):
        super().__init__()
        self.kind = kind
        self.dim = dim
        self.out = out
        if kind == 'linear':
            self.proj = nn.Linear(dim, out, bias=True)
            with torch.no_grad():
                self.proj.weight.zero_()
                self.proj.weight[:, :out] = torch.eye(out)
                self.proj.bias.zero_()
        elif kind == 'ortho':
            p = torch.zeros(dim, out)
            p[:out, :] = torch.eye(out)
            self.raw = nn.Parameter(p + 1e-4 * torch.randn(dim, out))
        elif kind != 'mrl':
            raise ValueError(kind)

    def forward(self, x):
        if self.kind == 'linear':
            x = self.proj(x)
        elif self.kind == 'ortho':
            q, _ = torch.linalg.qr(self.raw.float(), mode='reduced')
            x = x @ q
        else:
            x = x[..., :self.out]
        return F.normalize(x, p=2, dim=-1)


def evaluate_head(head, full, packed, device, topk=20, limit=1000):
    head.eval()
    hist, y = packed
    n = min(len(y), limit)
    with torch.no_grad():
        z = head(full.to(device))
        z = z.to('cpu')
        correct = 0.0
        ndcg = 0.0
        for start in range(0, n, 128):
            end = min(start + 128, n)
            h = hist[start:end]
            mask = (h >= 0)
            hist_z = z[h.clamp_min(0)] * mask.unsqueeze(-1)
            u = F.normalize(hist_z.sum(1) / mask.sum(1).clamp_min(1).unsqueeze(-1), dim=1)
            scores = u @ z.T
            # Do not recommend previously seen items.
            rows = torch.arange(len(h))[:, None].expand_as(h)
            scores[rows[mask], h[mask]] = float('-inf')
            inds = scores.topk(topk, dim=1).indices
            hit = inds.eq(y[start:end, None])
            anyhit = hit.any(1)
            correct += anyhit.sum().item()
            if anyhit.any():
                ranks = hit.float().argmax(1)[anyhit]
                ndcg += (1 / torch.log2(ranks.float() + 2)).sum().item()
    return dict(recall=correct/n, ndcg=ndcg/n)



def representation_stats(full, z, sample_size=256, k=20, seed=42):
    """Content-neighborhood preservation diagnostic, no behavior labels needed."""
    rng = np.random.default_rng(seed)
    n = len(full)
    queries = rng.choice(n, size=min(sample_size, n), replace=False)
    with torch.no_grad():
        # Use laptop GPU if available, or manageable CPU query blocks.
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        base = F.normalize(full.float().to(device), dim=-1)
        projected = F.normalize(torch.from_numpy(z).float().to(device), dim=-1)
        overlap = []
        for ix in np.array_split(queries, max(1, math.ceil(len(queries) / 128))):
            # CPU with small query blocks; O(sampled_queries * catalog_size) only.
            ix_t = torch.as_tensor(ix, device=device, dtype=torch.long)
            a = base[ix_t] @ base.T
            b = projected[ix_t] @ projected.T
            rows = torch.arange(len(ix), device=device)
            a[rows, ix_t] = float('-inf')
            b[rows, ix_t] = float('-inf')
            ka = a.topk(min(k,n-1),dim=-1).indices.cpu().numpy()
            kb = b.topk(min(k,n-1),dim=-1).indices.cpu().numpy()
            overlap.extend(len(set(u) & set(v)) / len(u) for u,v in zip(ka,kb))
    return {'continuous_neighbor_overlap20': float(np.mean(overlap)),
            'overlap_query_count': int(len(queries))}


def save_head_stats(full, z, head, data, device, args):
    stats = representation_stats(full, z, seed=args.seed)
    val_events = val_windows(data['users'], args.hist_len)
    random.Random(args.seed).shuffle(val_events)
    val = pack_windows(val_events, args.hist_len)
    score = evaluate_head(head, full, val, device, limit=args.val_users)
    stats.update({'proxy_val_recall20': score['recall'],
                  'proxy_val_ndcg20':score['ndcg'],
                  'note': 'Proxy: normalized mean history embedding scores items; not autoregressive SID metrics.'})
    stats['trainable_parameters'] = sum(p.numel() for p in head.parameters())
    if head.kind == 'ortho':
        with torch.no_grad():
            q, _ = torch.linalg.qr(head.raw.float(), mode='reduced')
            stats['orthogonality_max_error'] = (q.T @ q - torch.eye(128, device=q.device)).abs().max().item()
    Path(args.out).with_suffix('.head_stats.json').write_text(json.dumps(stats,indent=2))
    print('HEAD DIAGNOSTIC:',json.dumps(stats))


def train_head(args):
    set_seed(args.seed)
    device = get_device(args.device)
    data = load_data(args.data)
    full = torch.from_numpy(np.load(args.embeddings).astype('float32'))
    assert full.shape[0] == len(data['items']) and full.shape[1] == 768
    head = Head(args.kind).to(device)
    # Equal sampling/dropout streams after different head initializations.
    set_seed(args.seed)
    started = time.perf_counter()
    history = []
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.kind == 'mrl':
        z = F.normalize(full[:, :128], dim=-1).numpy()
        np.save(args.out, z)
        save_head_stats(full, z, head, data, device, args)
        print(f'MRL: pretrained 128d prefix -> {args.out}')
        return
    windows = sample_train_windows(data['users'], args.hist_len, args.max_windows, args.seed)
    packed = pack_windows(windows, args.hist_len)
    val_events = val_windows(data['users'], args.hist_len)
    random.Random(args.seed).shuffle(val_events)
    val = pack_windows(val_events, args.hist_len)
    # Keep embedding corpus on device: ~40 MB for 12k x 768 FP32.
    full_dev = full.to(device)
    optim = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    best = -1
    best_state = None
    for epoch in range(args.epochs):
        head.train()
        order = torch.randperm(len(windows))
        sums = []
        for start in range(0, len(order), args.batch_size):
            ids = order[start:start+args.batch_size]
            h, pos = (t[ids].to(device) for t in packed)
            mask = h.ge(0)
            z_h = head(full_dev[h.clamp_min(0)])
            context = F.normalize((z_h * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp_min(1).unsqueeze(-1), dim=1)
            z_pos = head(full_dev[pos])
            neg_ids = torch.randint(full_dev.size(0), (len(ids), args.negatives), device=device)
            z_neg = head(full_dev[neg_ids])
            cand = torch.cat([z_pos.unsqueeze(1), z_neg], dim=1)
            logits = torch.bmm(cand, context.unsqueeze(-1)).squeeze(-1) / args.temperature
            # Prevent accidental false negatives: repeated target or an item in history.
            invalid = neg_ids.eq(pos[:, None]) | (neg_ids.unsqueeze(-1) == h[:, None, :]).any(-1)
            logits[:, 1:] = logits[:, 1:].masked_fill(invalid, float('-inf'))
            loss = F.cross_entropy(logits, torch.zeros(len(ids), device=device, dtype=torch.long))
            optim.zero_grad()
            loss.backward()
            optim.step()
            sums.append(loss.item())
        score = evaluate_head(head, full, val, device, limit=args.val_users)
        history.append({'epoch':epoch+1, 'loss':float(np.mean(sums)), **score})
        print(f'HEAD {args.kind} epoch {epoch+1} loss={np.mean(sums):.4f} val_R20={score["recall"]:.4f} val_N20={score["ndcg"]:.4f}', flush=True)
        if score['ndcg'] > best:
            best = score['ndcg']
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    assert best_state is not None
    head.load_state_dict(best_state)
    head = head.to(device)
    head.eval()
    with torch.no_grad():
        z = torch.cat([head(t.to(device)).cpu() for t in full.split(1024)]).numpy()
    np.save(args.out, z.astype('float32'))
    torch.save({'kind': args.kind, 'state_dict': best_state, 'out_dim': 128}, str(Path(args.out).with_suffix('.pt')))
    save_head_stats(full, z, head, data, device, args)
    Path(args.out).with_suffix('.training.json').write_text(json.dumps({'config':vars(args) | {'fn':str(args.fn)}, 'epochs':history, 'seconds':time.perf_counter()-started}, indent=2))
    print(f'HEAD: {args.kind} -> {args.out}')


class RQVAE(nn.Module):
    def __init__(self, in_dim=128, latent_dim=32, n_codes=256, n_levels=3):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.n_codes = n_codes
        self.n_levels = n_levels
        self.encoder = nn.Sequential(nn.Linear(in_dim, 64), nn.GELU(), nn.Linear(64, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 64), nn.GELU(), nn.Linear(64, in_dim))
        self.codebooks = nn.Parameter(torch.randn(n_levels, n_codes, latent_dim) * 0.01)

    def quantize(self, latent):
        residual = latent
        total = torch.zeros_like(latent)
        codes = []
        cb_loss = 0.0
        commit_loss = 0.0
        for level in range(self.n_levels):
            centers = self.codebooks[level]
            d2 = (residual.square().sum(1, keepdim=True)
                  - 2 * residual @ centers.T
                  + centers.square().sum(1).unsqueeze(0))
            idx = d2.argmin(1)
            chosen = F.embedding(idx, centers)
            cb_loss = cb_loss + (chosen - residual.detach()).square().sum(-1).mean()
            commit_loss = commit_loss + (residual - chosen.detach()).square().sum(-1).mean()
            total = total + chosen
            residual = residual - chosen.detach()
            codes.append(idx)
        return total, torch.stack(codes, 1), cb_loss, commit_loss

    def forward(self, x):
        z = self.encoder(x)
        q, code_ids, cb, commit = self.quantize(z)
        q_st = z + (q - z).detach()
        reconstructed = self.decoder(q_st)
        recon_loss = (reconstructed - x).square().sum(-1).mean()
        return reconstructed, code_ids, recon_loss, cb, commit

    @torch.no_grad()
    def init_kmeans(self, features, seed=42):
        from sklearn.cluster import MiniBatchKMeans
        self.eval()
        device = next(self.parameters()).device
        zs = []
        for batch in torch.from_numpy(features).float().split(1024):
            zs.append(self.encoder(batch.to(device)).cpu().numpy())
        residual = np.concatenate(zs, axis=0)
        for j in range(self.n_levels):
            km = MiniBatchKMeans(
                n_clusters=self.n_codes, batch_size=1024, n_init=2,
                max_iter=75, random_state=seed+j,
            )
            km.fit(residual)
            centers = km.cluster_centers_.astype('float32')
            self.codebooks[j].copy_(torch.from_numpy(centers).to(device))
            residual = residual - centers[km.predict(residual)]


def make_unique_codes(base):
    """Base RQ codes can collide. Add an explicit deterministic suffix.

    For every same-3-code group, suffix = item index within sorted group.
    This makes each full SID unambiguous. Suffix vocabulary is reported.
    """
    groups = collections.defaultdict(list)
    for i, codes in enumerate(base):
        groups[tuple(int(x) for x in codes)].append(i)
    suffix = np.zeros((len(base), 1), dtype='int64')
    for group in groups.values():
        for r, item_id in enumerate(group):
            suffix[item_id, 0] = r
    full = np.concatenate([base, suffix], axis=1)
    assert len({tuple(row) for row in full}) == len(full)
    return full, {'base_unique': len(groups), 'base_collision_items': len(base)-len(groups),
                  'base_max_group': max(map(len, groups.values())),
                  'suffix_vocab': int(suffix.max())+1}


def train_rqvae(args):
    set_seed(args.seed)
    device = get_device(args.device)
    x_np = np.load(args.embeddings).astype('float32')
    assert x_np.ndim == 2 and x_np.shape[1] == 128
    x = torch.from_numpy(x_np).to(device)
    model = RQVAE(n_codes=args.n_codes).to(device)
    # Stabilize latent geometry before fitting residual centroids.
    warm_optim = torch.optim.AdamW(list(model.encoder.parameters())+list(model.decoder.parameters()), lr=args.lr, weight_decay=0)
    for epoch in range(args.warmup_epochs):
        for ids in torch.randperm(len(x),device=device).split(args.batch_size):
            recon = model.decoder(model.encoder(x[ids]))
            loss = (recon-x[ids]).square().sum(-1).mean()
            warm_optim.zero_grad(); loss.backward(); warm_optim.step()
        print(f'AE WARMUP {epoch+1} loss={loss.item():.5f}',flush=True)
    model.init_kmeans(x_np, seed=args.seed)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    best = float('inf')
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(x), device=device)
        running = []
        for ids in order.split(args.batch_size):
            _, _, recon, cb, commit = model(x[ids])
            loss = recon + args.cb_weight * cb + args.commit_weight * commit
            optim.zero_grad()
            loss.backward()
            optim.step()
            running.append([recon.item(), cb.item(), commit.item()])
        model.eval()
        with torch.no_grad():
            losses = [model(chunk)[2].item() * len(chunk) for chunk in x.split(args.batch_size)]
            val_recon = sum(losses) / len(x)
        print(f'RQVAE epoch {epoch+1} recon={val_recon:.5f} train_parts={np.mean(running,axis=0)}', flush=True)
        if val_recon < best:
            best, best_state = val_recon, {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model = model.to(device).eval()
    with torch.no_grad():
        codes = torch.cat([model.quantize(model.encoder(chunk))[1] for chunk in x.split(1024)]).cpu().numpy()
    full_codes, stats = make_unique_codes(codes)
    stats['codebook_stage_usage'] = [int(len(set(codes[:, i]))) for i in range(3)]
    stats['recon_loss'] = float(best)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, full_codes)
    torch.save({'state_dict': best_state, 'n_codes': args.n_codes}, str(Path(args.out).with_suffix('.pt')))
    Path(args.out).with_suffix('.stats.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(f'RQVAE SID: {args.out}: {stats}')


class AutoRegSID(nn.Module):
    def __init__(self, vocab, pad, bos, dim=128, dropout=0.1):
        super().__init__()
        self.vocab = vocab
        self.pad = pad
        self.bos = bos
        self.dim = dim
        self.emb = nn.Embedding(vocab, dim, padding_idx=pad)
        self.hist_gru = nn.GRU(dim, dim, batch_first=True)
        self.dec_gru = nn.GRU(dim, dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.to_logits = nn.Linear(dim, vocab)

    def history_state(self, tokens):
        b, l, steps = tokens.shape
        x = self.emb(tokens.reshape(b, l*steps))
        _, h = self.hist_gru(x)
        return h

    def forward(self, history_tokens, target_tokens, offsets, n_codes, suffix_vocab):
        h = self.history_state(history_tokens)
        bos = torch.full((len(target_tokens), 1), self.bos, device=target_tokens.device, dtype=torch.long)
        teacher_inputs = torch.cat([bos, target_tokens[:, :-1]], dim=1)
        y, _ = self.dec_gru(self.emb(teacher_inputs), h)
        logits = self.to_logits(self.dropout(y))
        classes = [n_codes,n_codes,n_codes,suffix_vocab]
        losses = []
        for level, nclass in enumerate(classes):
            if nclass == 1:
                continue
            level_logits = logits[:, level, offsets[level]: offsets[level]+nclass]
            target = target_tokens[:, level]-offsets[level]
            losses.append(F.cross_entropy(level_logits, target))
        return sum(losses)/len(losses)


def make_tokenized_codes(sids, n_codes):
    assert sids.ndim == 2 and sids.shape[1] == 4
    assert sids[:, :3].min() >= 0 and sids[:, :3].max() < n_codes
    suffix_vocab = int(sids[:, 3].max()) + 1
    offsets = [0,n_codes,2*n_codes,3*n_codes]
    tokens = sids + np.array(offsets, dtype='int64')[None,:]
    bos = 3*n_codes+suffix_vocab
    pad = bos+1
    vocab = pad+1
    return torch.from_numpy(tokens.astype('int64')), offsets, suffix_vocab, bos, pad, vocab


def make_hist_tokens(hist_items, token_codes, pad):
    hist_items = hist_items.to(token_codes.device)
    h = hist_items.clamp_min(0)
    toks = token_codes[h]
    toks = toks.masked_fill(hist_items.lt(0).unsqueeze(-1), pad)
    return toks


def load_gen_dataset(data_path, sid_path, n_codes):
    data = load_data(data_path)
    sid = np.load(sid_path)
    assert len(sid) == len(data['items'])
    t, offsets, suffix_vocab, bos, pad, vocab = make_tokenized_codes(sid, n_codes)
    return data, t, offsets, suffix_vocab, bos, pad, vocab


def teacher_forced_loss(model, packed, tokens, pad, offsets, n_codes, suffix_vocab, device, bs=256, limit=3000):
    model.eval()
    h, y = packed
    n = min(len(y), limit)
    total = 0
    with torch.no_grad():
        for start in range(0,n,bs):
            end = min(start+bs,n)
            hb = make_hist_tokens(h[start:end], tokens, pad).to(device)
            tg = tokens[y[start:end]].to(device)
            loss = model(hb, tg, offsets, n_codes, suffix_vocab)
            total += len(tg) * loss.item()
    return total/n


def train_gen(args):
    set_seed(args.seed)
    device = get_device(args.device)
    data, tok, offsets, suffix_vocab, bos, pad, vocab = load_gen_dataset(args.data, args.sids, args.n_codes)
    tok = tok.to(device)
    windows = sample_train_windows(data['users'], args.hist_len, args.max_windows, args.seed)
    train = pack_windows(windows, args.hist_len)
    val = pack_windows(val_windows(data['users'], args.hist_len), args.hist_len)
    model = AutoRegSID(vocab, pad, bos, dim=args.model_dim).to(device)
    set_seed(args.seed)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    best = float('inf')
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(windows))
        running = []
        for ix in order.split(args.batch_size):
            h, y = (x[ix].to(device) for x in train)
            ht = make_hist_tokens(h, tok, pad)
            targets = tok[y]
            loss = model(ht, targets, offsets, args.n_codes, suffix_vocab)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            running.append(loss.item())
        val_loss = teacher_forced_loss(model, val, tok, pad, offsets, args.n_codes, suffix_vocab, device, limit=args.val_users)
        print(f'GEN epoch {epoch+1} trainCE={np.mean(running):.4f} valCE={val_loss:.4f}', flush=True)
        if val_loss < best:
            best, best_state = val_loss, {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict':best_state,'vocab':vocab,'bos':bos,'pad':pad,'n_codes':args.n_codes,
                'model_dim':args.model_dim,'suffix_vocab':suffix_vocab,'val_loss':best}, args.out)
    print(f'GEN: saved {args.out}')


def build_trie(token_codes):
    trie = collections.defaultdict(set)
    lookup = {}
    for item_id, tokens in enumerate(token_codes.tolist()):
        for j in range(4):
            trie[tuple(tokens[:j])].add(int(tokens[j]))
        lookup[tuple(tokens)] = item_id
    return trie, lookup


@torch.no_grad()
def recommend(model, hist_items, token_codes, trie, lookup, bos, pad, device, hist_len=10, beam=30, topk=10):
    # Trie-constrained, *item-level* autoregressive beam search.
    his = torch.full((1,hist_len), -1, dtype=torch.long, device=device)
    hs = hist_items[-hist_len:]
    his[0,-len(hs):] = torch.tensor(hs,device=device)
    ht = make_hist_tokens(his, token_codes, pad)
    state = model.history_state(ht)
    beams = [((),0.0,state,bos)]
    for level in range(4):
        # Batch all partial beams into one GRU call per generated token.
        # This is much faster on laptop CUDA than a GPU launch for each beam.
        last_tokens = torch.tensor([[item[3]] for item in beams], device=device, dtype=torch.long)
        all_h = torch.cat([item[2] for item in beams], dim=1)
        output, updated_states = model.dec_gru(model.emb(last_tokens), all_h)
        all_logits = model.to_logits(output[:, 0, :])
        candidates = []
        for beam_idx, (prefix, score, _, last) in enumerate(beams):
            valid = sorted(trie[prefix])
            v = torch.tensor(valid, device=device, dtype=torch.long)
            log_probs = F.log_softmax(all_logits[beam_idx, v], dim=0)
            updated_h = updated_states[:, beam_idx:beam_idx+1]
            for code, lp in zip(valid, log_probs.tolist()):
                candidates.append((prefix+(code,), score+lp, updated_h, code))
        candidates.sort(key=lambda b:b[1], reverse=True)
        beams = candidates[:beam]
    seen = set(hist_items)
    out = []
    for codes, score, _, _ in beams:
        item = lookup[codes]
        if item not in seen:
            out.append(item)
            if len(out)>=topk:
                break
    return out


def evaluate(args):
    set_seed(args.seed)
    device = get_device(args.device)
    data, tokens, offsets, suffix_vocab, bos, pad, vocab = load_gen_dataset(args.data,args.sids,args.n_codes)
    checkpoint = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    assert checkpoint['vocab']==vocab and checkpoint['n_codes']==args.n_codes
    model = AutoRegSID(vocab,pad,bos,dim=checkpoint['model_dim']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    tokens = tokens.to(device)
    trie,lookup = build_trie(tokens.cpu())
    eval_users = list(data['users'])
    random.Random(args.seed).shuffle(eval_users)
    if args.eval_users:
        eval_users = eval_users[:args.eval_users]
    hit, ndcg = 0.0,0.0
    rows = []
    for u_idx,u in enumerate(eval_users):
        seq = u['seq']
        hist = seq[:-1] if args.split=='test' else seq[:-2]
        target = seq[-1] if args.split=='test' else seq[-2]
        pred = recommend(model,hist,tokens,trie,lookup,bos,pad,device,
                         hist_len=args.hist_len,beam=args.beam,topk=args.topk)
        if target in pred:
            rank = pred.index(target)
            hit += 1
            ndcg += 1 / math.log2(rank+2)
        rows.append({'user':u['user'], 'target':target, 'predictions':pred,
                     'recall':int(target in pred), 'ndcg':1/math.log2(pred.index(target)+2) if target in pred else 0.0})
        if (u_idx+1) % 100 == 0:
            print(f'EVAL {u_idx+1}/{len(eval_users)}',flush=True)
    result = {
        'split':args.split,'users':len(eval_users),'item_level_recall':hit/len(eval_users),
        'item_level_ndcg':ndcg/len(eval_users),'topk':args.topk,'beam':args.beam,
        'generator_validation_ce':checkpoint.get('val_loss'),
        'note':'Single held-out item per user; all-catalog unique SID; history items removed from outputs.'
    }
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
    Path(args.out).with_suffix('.users.json').write_text(json.dumps(rows),encoding='utf-8')
    print(json.dumps(result,indent=2))


def smoke(args):
    """Toy functional test: no model download or Amazon data required."""
    set_seed(42)
    torch.set_num_threads(2)
    N, D = 300, 768
    full = torch.randn(N,D)
    full = F.normalize(full,dim=-1)
    for kind in ['mrl','linear','ortho']:
        head = Head(kind)
        z = head(full)
        assert z.shape==(N,128)
        assert torch.isfinite(z).all()
        if kind=='ortho':
            q,_=torch.linalg.qr(head.raw, mode='reduced')
            assert torch.allclose(q.T@q,torch.eye(128),atol=1e-5)
    x = Head('mrl')(full)
    vae = RQVAE(n_codes=16)
    vae.init_kmeans(x.detach().numpy())
    vae.train()
    optim = torch.optim.AdamW(vae.parameters(),lr=1e-3)
    for _ in range(3):
        _,_,rec,cb,cm=vae(x)
        loss=rec+cb+0.25*cm
        optim.zero_grad();loss.backward();optim.step()
    with torch.no_grad():
        ids = vae.quantize(vae.encoder(x))[1].numpy()
    ids, stats = make_unique_codes(ids)
    tok, offsets, suffix_size,bos,pad,vocab=make_tokenized_codes(ids,16)
    model = AutoRegSID(vocab,pad,bos,dim=32)
    hist = torch.randint(0,N,(8,5))
    targets = torch.randint(0,N,(8,))
    ht=make_hist_tokens(hist,tok,pad)
    loss=model(ht,tok[targets],offsets,16,suffix_size)
    assert torch.isfinite(loss)
    loss.backward()
    trie,lookup=build_trie(tok)
    result=recommend(model,hist[0].tolist(),tok,trie,lookup,bos,pad,'cpu',hist_len=5,beam=12,topk=5)
    assert all(x not in hist[0].tolist() for x in result)
    print('SMOKE PASS: all 3 heads, orthogonality, differentiable RQ-VAE, unique SID, AR generator, constrained beam.')
    print('Toy statistics:',stats,'; top items:',result)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='cmd',required=True)
    pp=sub.add_parser('prepare')
    pp.add_argument('--reviews',required=True)
    pp.add_argument('--metadata',required=True)
    pp.add_argument('--out',default='work/beauty.json')
    pp.add_argument('--min-rating',type=float,default=4.0)
    pp.add_argument('--min-user-len',type=int,default=5)
    pp.add_argument('--max-chars',type=int,default=750)
    pp.add_argument('--seed',type=int,default=42)
    pp.set_defaults(fn=prep)

    ee=sub.add_parser('embed')
    ee.add_argument('--data',default='work/beauty.json')
    ee.add_argument('--model',required=True,help='Path to downloaded Hugging Face / SentenceTransformers checkpoint')
    ee.add_argument('--out',default='work/full768.npy')
    ee.add_argument('--device',default='auto')
    ee.add_argument('--batch-size',type=int,default=16)
    ee.add_argument('--max-tokens',type=int,default=192)
    ee.set_defaults(fn=embed)

    hh=sub.add_parser('head')
    hh.add_argument('--data',default='work/beauty.json')
    hh.add_argument('--embeddings',default='work/full768.npy')
    hh.add_argument('--kind',choices=['mrl','linear','ortho'],required=True)
    hh.add_argument('--out',required=True)
    hh.add_argument('--device',default='auto')
    hh.add_argument('--hist-len',type=int,default=10)
    hh.add_argument('--max-windows',type=int,default=60000)
    hh.add_argument('--val-users',type=int,default=1000)
    hh.add_argument('--negatives',type=int,default=20)
    hh.add_argument('--temperature',type=float,default=0.15)
    hh.add_argument('--epochs',type=int,default=3)
    hh.add_argument('--batch-size',type=int,default=256)
    hh.add_argument('--lr',type=float,default=3e-4)
    hh.add_argument('--seed',type=int,default=42)
    hh.set_defaults(fn=train_head)

    rr=sub.add_parser('rqvae')
    rr.add_argument('--embeddings',required=True)
    rr.add_argument('--out',required=True)
    rr.add_argument('--device',default='auto')
    rr.add_argument('--n-codes',type=int,default=256)
    rr.add_argument('--epochs',type=int,default=15)
    rr.add_argument('--warmup-epochs',type=int,default=10)
    rr.add_argument('--batch-size',type=int,default=512)
    rr.add_argument('--lr',type=float,default=3e-4)
    rr.add_argument('--cb-weight',type=float,default=1.0)
    rr.add_argument('--commit-weight',type=float,default=0.25)
    rr.add_argument('--seed',type=int,default=42)
    rr.set_defaults(fn=train_rqvae)

    gg=sub.add_parser('train-gen')
    gg.add_argument('--data',default='work/beauty.json')
    gg.add_argument('--sids',required=True)
    gg.add_argument('--out',required=True)
    gg.add_argument('--n-codes',type=int,default=256)
    gg.add_argument('--device',default='auto')
    gg.add_argument('--epochs',type=int,default=5)
    gg.add_argument('--hist-len',type=int,default=10)
    gg.add_argument('--batch-size',type=int,default=256)
    gg.add_argument('--max-windows',type=int,default=60000)
    gg.add_argument('--model-dim',type=int,default=128)
    gg.add_argument('--val-users',type=int,default=1000)
    gg.add_argument('--lr',type=float,default=1e-3)
    gg.add_argument('--seed',type=int,default=42)
    gg.set_defaults(fn=train_gen)

    ev=sub.add_parser('eval')
    ev.add_argument('--data',default='work/beauty.json')
    ev.add_argument('--sids',required=True)
    ev.add_argument('--checkpoint',required=True)
    ev.add_argument('--out',required=True)
    ev.add_argument('--split',choices=['val','test'],default='test')
    ev.add_argument('--device',default='auto')
    ev.add_argument('--n-codes',type=int,default=256)
    ev.add_argument('--hist-len',type=int,default=10)
    ev.add_argument('--eval-users',type=int,default=1000,help='0 = all users')
    ev.add_argument('--beam',type=int,default=30)
    ev.add_argument('--topk',type=int,default=10)
    ev.add_argument('--seed',type=int,default=42)
    ev.set_defaults(fn=evaluate)

    sm=sub.add_parser('smoke')
    sm.set_defaults(fn=smoke)
    return p


if __name__=='__main__':
    torch.set_num_threads(4)
    args=parser().parse_args()
    args.fn(args)
