# -*- coding: utf-8 -*-
"""
GPU Keccak-256 與 CPU 參考實作的一致性測試。
固定輸入為 64 bytes（未壓縮公鑰去掉首位），與 gpu_keccak.keccak256_xy_batch 對應。

執行範例：
    PYTHONPATH=src python -m tron_vanity.test_keccak_gpu_vs_cpu --n 32
"""
from __future__ import annotations

import argparse
import secrets

import cupy as cp
import numpy as np
import sha3

from .gpu_keccak import keccak256_xy_batch


def _cpu_keccak(batch: np.ndarray) -> np.ndarray:
    """以 CPU 版 keccak_256 驗證輸出，輸入/輸出皆為 uint8 陣列。"""
    out = np.zeros((batch.shape[0], 32), dtype=np.uint8)
    for idx, row in enumerate(batch):
        hasher = sha3.keccak_256()
        hasher.update(row.tobytes())
        out[idx, :] = np.frombuffer(hasher.digest(), dtype=np.uint8)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=16, help="測試樣本數（預設 16 筆）")
    args = parser.parse_args()

    # 產生隨機 64 bytes 批次
    samples = np.stack(
        [np.frombuffer(secrets.token_bytes(64), dtype=np.uint8) for _ in range(args.n)],
        axis=0,
    )
    gpu_in = cp.asarray(samples)
    gpu_out = keccak256_xy_batch(gpu_in)
    cpu_out = _cpu_keccak(samples)

    gpu_np = cp.asnumpy(gpu_out)
    mismatch = np.nonzero(~np.all(gpu_np == cpu_out, axis=1))[0]
    if mismatch.size:
        idx = int(mismatch[0])
        print(f"[X] 第 {idx} 筆結果不一致")
        print(f"    輸入: {samples[idx].tobytes().hex()}")
        print(f"    CPU : {cpu_out[idx].tobytes().hex()}")
        print(f"    GPU : {gpu_np[idx].tobytes().hex()}")
        return 1

    print(f"[OK] {args.n} 筆全部一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
