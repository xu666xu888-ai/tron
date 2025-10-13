# -*- coding: utf-8 -*-
"""
對比測試：GPU secp256k1 視窗法（w4） vs 參考內核（bit-scan）

執行：
  PYTHONPATH=src python -m tron_vanity.test_ecc_w4_vs_ref --n 64
"""
from __future__ import annotations

import argparse
import secrets
import cupy as cp

import time

from .gpu_secp256k1 import (
    gpu_secp256k1_batch,
    gpu_secp256k1_batch_window4,
    warmup_window4_table,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--repeat', type=int, default=1, help='重複測試次數（預設 1）')
    ap.add_argument('--warmup', action='store_true', help='測試前預先建立 Window4 預計算表')
    args = ap.parse_args()

    if args.warmup:
        warmup_window4_table(force=True)

    for round_idx in range(args.repeat):
        privs = [secrets.token_bytes(32) for _ in range(args.n)]
        privs_gpu = cp.array([list(p) for p in privs], dtype=cp.uint8)
        start = time.perf_counter()

        ref = cp.asnumpy(gpu_secp256k1_batch(privs_gpu))
        w4  = cp.asnumpy(gpu_secp256k1_batch_window4(privs_gpu))

        mismatch = 0
        for i in range(args.n):
            if bytes(ref[i]) != bytes(w4[i]):
                mismatch += 1
                print(f'[X] mismatch at {i} (round {round_idx + 1})')
                break

        elapsed = time.perf_counter() - start
        print(f'[Round {round_idx + 1}/{args.repeat}] 耗時 {elapsed:.3f}s')

        if mismatch:
            print(f'[!] 本輪發現 {mismatch} 個不一致')
            return 2

    print(f'[OK] w4 與參考內核一致（n={args.n}, rounds={args.repeat}）')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
