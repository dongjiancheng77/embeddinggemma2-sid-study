#!/usr/bin/env python3
"""Run MRL, Linear and Ortho branches with the same split/hyperparameters."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def run(argv):
    command = [sys.executable, str(Path(__file__).with_name('mini_tiger.py'))] + [str(v) for v in argv]
    print('\n>>>', ' '.join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='work/beauty.json')
    p.add_argument('--embeddings', default='work/full768.npy')
    p.add_argument('--out', default='work/experiment1')
    p.add_argument('--device', default='auto')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-codes', type=int, default=256)
    p.add_argument('--head-epochs', type=int, default=3)
    p.add_argument('--rq-epochs', type=int, default=15)
    p.add_argument('--gen-epochs', type=int, default=5)
    p.add_argument('--max-windows', type=int, default=60000)
    p.add_argument('--eval-users', type=int, default=1000)
    p.add_argument('--beam', type=int, default=30)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = []
    for kind in ['mrl', 'linear', 'ortho']:
        head = out / f'{kind}_128.npy'
        sid = out / f'{kind}_sid.npy'
        gen = out / f'{kind}_gen.pt'
        result = out / f'{kind}_test.json'
        run(['head','--kind',kind,'--data',args.data,'--embeddings',args.embeddings,
             '--out',head,'--device',args.device,'--seed',args.seed,
             '--epochs',args.head_epochs,'--max-windows',args.max_windows])
        run(['rqvae','--embeddings',head,'--out',sid,'--device',args.device,
             '--n-codes',args.n_codes,'--seed',args.seed,'--epochs',args.rq_epochs])
        run(['train-gen','--data',args.data,'--sids',sid,'--out',gen,
             '--device',args.device,'--n-codes',args.n_codes,'--seed',args.seed,
             '--max-windows',args.max_windows,'--epochs',args.gen_epochs])
        run(['eval','--data',args.data,'--sids',sid,'--checkpoint',gen,'--out',result,
             '--device',args.device,'--seed',args.seed,'--n-codes',args.n_codes,
             '--beam',args.beam,'--eval-users',args.eval_users,'--topk',10])
        summary = json.loads(result.read_text())
        s = json.loads(sid.with_suffix('.stats.json').read_text())
        d = json.loads(head.with_suffix('.head_stats.json').read_text())
        stats.append({'method':kind,'recall10':summary['item_level_recall'],
                      'ndcg10':summary['item_level_ndcg'],
                      'base_unique':s['base_unique'],
                      'colliding_items':s['base_collision_items'],
                      'suffix_vocab':s['suffix_vocab'],
                      'usage_l1':s['codebook_stage_usage'][0],
                      'usage_l2':s['codebook_stage_usage'][1],
                      'usage_l3':s['codebook_stage_usage'][2],
                      'rq_reconstruction':s['recon_loss'],
                      'semantic_neighbor_overlap20':d['continuous_neighbor_overlap20'],
                      'proxy_val_recall20':d['proxy_val_recall20']})
    with (out/'summary.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(stats[0]))
        writer.writeheader(); writer.writerows(stats)
    print('\n=== SUMMARY ===\n', (out/'summary.csv').read_text())


if __name__ == '__main__':
    main()
