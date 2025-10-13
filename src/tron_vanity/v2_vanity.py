# -*- coding: utf-8 -*-
"""
V2：TRON 靚號地址搜尋器（CUDA 輔助）
- 以 GPU 批量產生候選私鑰（CuPy），CPU 使用 libsecp256k1 計算公鑰/地址
- 多進程平行檢查 Base58 地址是否符合前綴（Prefix）

注意：
- CUDA 僅用於「亂數候選產生」，EC 乘法仍使用 coincurve（C 綁定）在 CPU 上進行；
  若需完整的 GPU 橢圓曲線計算，需以 C++/CUDA（例如 CGBN）另行實作並以 pybind11 綁定。

執行範例：
  python -m tron_vanity.v2_vanity --prefix T777 --threads 0 --gpu-batch 8192 --batch 8192
"""
from __future__ import annotations

import os
import sys
import time
import math
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Optional, List

from .addr import privkey_to_tron_address
from .gpu_random import has_cupy, generate_gpu_secrets
from .hardware_config import get_hardware_config


def _suggest_gpu_batch(prefix: str, base_batch: int, target_hits: int = 4, max_batch: int = 1 << 18) -> int:
    """根據前綴長度估算合適的 GPU 批次大小，避免過低命中率。"""
    if base_batch <= 0:
        base_batch = 16384
    prefix = prefix.strip()
    if not prefix:
        return base_batch
    length = len(prefix)
    target_hits = max(1, target_hits)
    try:
        denom = pow(58, length)
    except OverflowError:
        denom = max_batch
    if denom <= 0:
        denom = max_batch
    if base_batch * target_hits >= denom:
        suggested = base_batch
    else:
        needed = target_hits * denom
        suggested = min(max_batch, max(base_batch, needed))
    align = 256
    suggested = int(max(align, min(max_batch, ((suggested + align - 1) // align) * align)))
    return suggested


def derive_and_check(privkey: bytes, prefix: str) -> Optional[Tuple[str, str]]:
    """子進程工作：由私鑰導出地址，命中則回傳 (priv_hex, b58)。"""
    hex_addr, b58 = privkey_to_tron_address(privkey)
    if b58.startswith(prefix):
        return (privkey.hex(), b58)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="TRON 靚號地址搜尋（CUDA 亂數輔助）")
    parser.add_argument("--prefix", type=str, required=True, help="欲匹配的 Base58 前綴，建議以 'T' 開頭")
    parser.add_argument("--threads", type=int, default=0, help="工作進程數；0 代表自動=CPU 核心數")
    parser.add_argument("--batch", type=int, default=4096, help="每一輪分派的私鑰數量（CPU 產生）")
    parser.add_argument("--gpu-batch", type=int, default=0, help="若>0，啟用 GPU 亂數，一輪產生此數量的候選")
    parser.add_argument("--timeout", type=int, default=0, help="搜尋逾時秒數；0 表示不限")
    parser.add_argument("--gpu-full", action="store_true", help="啟用完整 GPU 管線（含 Keccak/SHA256 校驗）；建議配合 VANITY_EXPERIMENTAL_GPU_SECP=1")
    args = parser.parse_args()

    # 正規化前綴（TRON 主網地址皆以 'T' 開頭；此處允許自定義大小寫）
    prefix = args.prefix

    # 進程數
    max_workers = args.threads or os.cpu_count() or 1

    print(f"[V2] 目標前綴: {prefix}")
    print(f"[V2] 進程數: {max_workers}")
    if args.gpu_full:
        init_batch = args.gpu_batch or args.batch or 16384
        print(f"[V2] 使用 GPU-FULL 模式，初始批次 {init_batch} 筆")
    else:
        if args.gpu_batch > 0:
            print(f"[V2] 使用 GPU 亂數，每輪 {args.gpu_batch} 筆")
            if not has_cupy():
                print("[V2] 警告：未偵測到 CuPy，將退化為 CPU 亂數。")
        else:
            print(f"[V2] 使用 CPU 亂數，每輪 {args.batch} 筆")

    deadline = time.time() + args.timeout if args.timeout > 0 else None

    if args.gpu_full:
        # 完整 GPU 管線模式：單行程在 GPU 上批量生成並檢查前綴
        from .gpu_addr import generate_tron_addresses_gpu
        hw_cfg = get_hardware_config()
        round_idx = 0
        base_batch = args.gpu_batch or args.batch or 16384
        if prefix.strip() and args.gpu_batch <= 0:
            gpu_batch = _suggest_gpu_batch(prefix, base_batch)
            print(f"[V2] 動態調整 GPU 批次為 {gpu_batch}")
        else:
            gpu_batch = base_batch
        dynamic_plan = hw_cfg.default_batches[:4] if hw_cfg.default_batches else (16384, 32768, 65536)
        stream_hint = max(hw_cfg.default_streams, 8 if len(prefix) >= 4 else 6)
        while True:
            round_idx += 1
            if deadline and time.time() > deadline:
                print("[V2] 已達逾時，結束搜尋。")
                return 2

            addrs, privs = generate_tron_addresses_gpu(
                gpu_batch,
                prefix=prefix,
                max_hits=1,
                dynamic_batches=dynamic_plan,
                stream_count=stream_hint,
            )
            if addrs:
                (hex_addr, b58), pk = addrs[0], privs[0]
                print("[V2] 命中靚號！（GPU-FULL）")
                print("[V2] 私鑰(HEX):", pk.hex())
                print("[V2] 地址(B58):", b58)
                return 0

            if round_idx % 10 == 0:
                print(f"[V2] [GPU-FULL] 已完成 {round_idx} 輪，尚未命中…")

        return 1

    # 預設：CPU/混合模式（原流程，多進程）
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        round_idx = 0
        while True:
            round_idx += 1
            if deadline and time.time() > deadline:
                print("[V2] 已達逾時，結束搜尋。")
                return 2

            # 準備一批候選私鑰
            if args.gpu_batch > 0:
                secrets_batch = generate_gpu_secrets(args.gpu_batch)
            else:
                secrets_batch = [os.urandom(32) for _ in range(args.batch)]

            # 分派到進程池檢查
            futures = [ex.submit(derive_and_check, sk, prefix) for sk in secrets_batch]

            for fut in as_completed(futures):
                hit = fut.result()
                if hit is not None:
                    priv_hex, b58 = hit
                    print("[V2] 命中靚號！")
                    print("[V2] 私鑰(HEX):", priv_hex)
                    print("[V2] 地址(B58):", b58)
                    return 0

            if round_idx % 10 == 0:
                print(f"[V2] 已完成 {round_idx} 輪，尚未命中…")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
