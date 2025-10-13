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

from .gpu_secp256k1 import gpu_secp256k1_batch, gpu_secp256k1_batch_window4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=64)
    args = ap.parse_args()

    privs = [secrets.token_bytes(32) for _ in range(args.n)]
    privs_gpu = cp.array([list(p) for p in privs], dtype=cp.uint8)

    ref = cp.asnumpy(gpu_secp256k1_batch(privs_gpu))
    w4  = cp.asnumpy(gpu_secp256k1_batch_window4(privs_gpu))

    mism = 0
    for i in range(args.n):
        if bytes(ref[i]) != bytes(w4[i]):
            mism += 1
            print(f'[X] mismatch at {i}')
    if mism == 0:
        print(f'[OK] w4 與參考內核一致（n={args.n}）')
        return 0
    else:
        print(f'[!] 不一致數量：{mism}/{args.n}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

