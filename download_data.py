#!/usr/bin/env python3
"""Download the small legacy Amazon Beauty 5-core + category metadata files."""
import argparse
from pathlib import Path
import urllib.request

BASE = 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/'
FILES = ['reviews_Beauty_5.json.gz', 'meta_Beauty.json.gz']

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data')
    args = p.parse_args()
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = dest / name
        if target.exists() and target.stat().st_size:
            print('Existing file (skip):', target)
            continue
        print('Downloading:', name, flush=True)
        try:
            urllib.request.urlretrieve(BASE+name, target)
        except Exception:
            target.unlink(missing_ok=True)
            raise
        print(f'Download complete: {target} ({target.stat().st_size} bytes)', flush=True)

if __name__ == '__main__':
    main()
