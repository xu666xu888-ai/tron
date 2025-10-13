# -*- coding: utf-8 -*-
"""
GPU Base58Check 與 CPU 參考實作的一致性測試。
僅考慮 TRON 地址格式：0x41 + 20 bytes，並附上雙 SHA-256 校驗碼。

執行：
    PYTHONPATH=src python -m tron_vanity.test_base58_gpu_vs_cpu --n 64
"""
from __future__ import annotations

import argparse
import secrets
import hashlib

import base58
import cupy as cp
import numpy as np

from .gpu_addr import gpu_base58check_batch


def _cpu_base58(tron21: bytes) -> str:
    checksum = hashlib.sha256(hashlib.sha256(tron21).digest()).digest()[:4]
    return base58.b58encode(tron21 + checksum).decode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=64, help="測試樣本數")
    args = parser.parse_args()

    payloads = [b"\x41" + secrets.token_bytes(20) for _ in range(args.n)]
    tron21_np = np.stack([np.frombuffer(p, dtype=np.uint8) for p in payloads], axis=0)
    tron21_gpu = cp.asarray(tron21_np)

    gpu_out = gpu_base58check_batch(tron21_gpu)
    ok = True
    for idx, tron21 in enumerate(payloads):
        cpu_val = _cpu_base58(tron21)
        if gpu_out[idx] != cpu_val:
            print(f"[X] mismatch at {idx}")
            print(f"    tron21 : {tron21.hex()}")
            print(f"    cpu    : {cpu_val}")
            print(f"    gpu    : {gpu_out[idx]}")
            ok = False
            break

    if ok:
        print(f"[OK] {args.n} 筆 Base58Check 全部一致")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
