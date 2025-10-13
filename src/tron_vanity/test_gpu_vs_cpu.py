# -*- coding: utf-8 -*-
"""
GPU 與 CPU 的 secp256k1 公鑰導出一致性測試
- 產生隨機私鑰並比較 GPU/CPU 輸出是否一致
執行：
  PYTHONPATH=src python -m tron_vanity.test_gpu_vs_cpu --n 16
"""
from __future__ import annotations

import argparse
import secrets

import cupy as cp
import coincurve

from .gpu_secp256k1 import gpu_secp256k1_batch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=16, help="測試樣本數")
    args = parser.parse_args()

    privs = [secrets.token_bytes(32) for _ in range(args.n)]
    privs_gpu = cp.array([list(p) for p in privs], dtype=cp.uint8)
    pubs_gpu = gpu_secp256k1_batch(privs_gpu)
    pubs_gpu = cp.asnumpy(pubs_gpu)

    ok = True
    for i, pk in enumerate(privs):
        pub_cpu = coincurve.PrivateKey(pk).public_key.format(compressed=False)
        pub_gpu = bytes(pubs_gpu[i])
        if pub_cpu != pub_gpu:
            print(f"[X] 不一致 index={i}")
            print(f"    pk  = {pk.hex()}")
            print(f"    cpu = {pub_cpu.hex()}")
            print(f"    gpu = {pub_gpu.hex()}")
            ok = False
            break
    if ok:
        print(f"[OK] {args.n} 筆全部一致")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

