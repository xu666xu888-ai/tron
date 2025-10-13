# -*- coding: utf-8 -*-
"""
TRON 地址生成性能基準測試
- 比較 CPU vs GPU 的生成速率
- 驗證第 100,000 筆的準確性（本地演算法、tronpy 比對、節點 validateaddress、Base58Check）

使用方式（於專案根目錄執行）：
  python3 scripts/benchmark.py

注意：
- GPU 模式僅使用 CUDA/CuPy 來批量產生私鑰亂數；secp256k1 橢圓曲線運算仍由 CPU（coincurve）處理。
- 若未安裝 CuPy 或未偵測到 GPU，GPU 測試會自動略過。
"""
from __future__ import annotations

import os
import sys
import time
from typing import Optional, Tuple

# 將 src 加到匯入路徑，便於以腳本直接執行
_HERE = os.path.dirname(__file__)
_SRC = os.path.join(_HERE, "..", "src")
sys.path.insert(0, os.path.abspath(_SRC))

from tron_vanity.addr import generate_privkey, privkey_to_tron_address, is_valid_tron_base58  # noqa: E402
from tron_vanity.gpu_random import has_cupy, generate_gpu_secrets  # noqa: E402
from tron_vanity.validate import validate_private_key  # noqa: E402


def benchmark_cpu(count: int) -> Tuple[bytes, float, float]:
    """CPU 模式基準測試。
    - 逐筆產生私鑰，導出地址（包含 EC/Keccak/Base58）
    - 回傳：最後一筆私鑰、耗時秒數、keys/sec
    """
    print(f"\n[CPU] 開始生成 {count:,} 筆地址…")
    start = time.perf_counter()
    last_privkey: Optional[bytes] = None

    for i in range(count):
        pk = generate_privkey()
        # 將完整流程跑過（EC -> Keccak -> Base58Check）
        _hex_addr, _b58_addr = privkey_to_tron_address(pk)
        if i == count - 1:
            last_privkey = pk

    elapsed = time.perf_counter() - start
    rate = count / elapsed if elapsed > 0 else 0.0

    print(f"[CPU] 完成！耗時: {elapsed:.2f} 秒")
    print(f"[CPU] 速率: {rate:.2f} keys/sec")

    assert last_privkey is not None
    return last_privkey, elapsed, rate


def benchmark_gpu(count: int) -> Tuple[Optional[bytes], float, float]:
    """GPU 模式基準測試。
    - 以 CuPy 批量產生私鑰，再由 CPU 完成 EC/Keccak/Base58 計算
    - 若未偵測到 CuPy/GPU，回傳 (None, 0, 0) 並略過
    """
    if not has_cupy():
        print("[GPU] CuPy 不可用，跳過 GPU 測試")
        return None, 0.0, 0.0

    print(f"\n[GPU] 開始生成 {count:,} 筆地址…")
    start = time.perf_counter()
    last_privkey: Optional[bytes] = None

    # 以批量處理提升生成效率（亂數在 GPU 端產生）
    batch_size = 4096
    remaining = count

    while remaining > 0:
        current_batch = min(batch_size, remaining)
        secrets_batch = generate_gpu_secrets(current_batch)

        for idx, pk in enumerate(secrets_batch):
            _hex_addr, _b58_addr = privkey_to_tron_address(pk)
            remaining -= 1
            if remaining == 0:
                last_privkey = pk
                break

    elapsed = time.perf_counter() - start
    rate = count / elapsed if elapsed > 0 else 0.0

    print(f"[GPU] 完成！耗時: {elapsed:.2f} 秒")
    print(f"[GPU] 速率: {rate:.2f} keys/sec")

    return last_privkey, elapsed, rate


def verify_last_key(privkey: bytes, mode: str) -> bool:
    """使用 V1 完整驗證流程驗證最後一筆私鑰。
    - 本地導出（addr.py）
    - tronpy 導出比對
    - 節點 validateaddress（若已配置 API KEY 或可訪問公共節點）
    - Base58Check 檢查
    """
    print(f"\n[{mode}] 驗證第 100,000 筆：")
    print(f"[{mode}] 私鑰(HEX): {privkey.hex()}")

    result = validate_private_key(privkey)

    print(f"[{mode}] 地址(HEX): {result['hex']}")
    print(f"[{mode}] 地址(B58): {result['base58']}")
    print(f"[{mode}] tronpy 導出: {result['tronpy_base58']}")
    print(f"[{mode}] tronpy 比對: {result['tronpy_match']}")
    print(f"[{mode}] 節點驗證: {result['validateaddress']}")

    ok_b58 = result['base58'].upper().startswith('T') and is_valid_tron_base58(result['base58'])
    ok_tronpy = bool(result['tronpy_match']) if result['tronpy_match'] is not None else True
    ok = ok_b58 and ok_tronpy

    if ok:
        print(f"[{mode}] ✅ 驗證通過")
    else:
        print(f"[{mode}] ❌ 驗證失敗")
    return ok


def main() -> int:
    COUNT = 100_000  # 測試規模：十萬筆

    print("=" * 60)
    print("TRON 地址生成性能基準測試")
    print(f"測試規模: {COUNT:,} 筆")
    print("=" * 60)

    # CPU 測試
    cpu_last_key, cpu_time, cpu_rate = benchmark_cpu(COUNT)
    cpu_valid = verify_last_key(cpu_last_key, "CPU")

    # GPU 測試
    gpu_last_key, gpu_time, gpu_rate = benchmark_gpu(COUNT)
    if gpu_last_key:
        gpu_valid = verify_last_key(gpu_last_key, "GPU")
    else:
        gpu_valid = False

    # 完全 GPU（雛形模組）測試：若模組存在則執行
    def benchmark_gpu_full(count: int):
        try:
            from tron_vanity.gpu_addr import generate_tron_addresses_gpu
        except Exception as e:
            print(f"\n[GPU-FULL] 模組不可用或初始化失敗：{e}")
            return None, 0.0, 0.0

        print(f"\n[GPU-FULL] 開始生成 {count:,} 筆地址（完全 GPU 管線雛形）…")
        start = time.perf_counter()
        addrs, privs = generate_tron_addresses_gpu(count, batch_size=16384)
        elapsed = time.perf_counter() - start
        rate = count / elapsed if elapsed > 0 else 0.0
        print(f"[GPU-FULL] 完成！耗時: {elapsed:.2f} 秒")
        print(f"[GPU-FULL] 速率: {rate:.2f} keys/sec")
        last_priv = privs[-1] if privs else None
        return last_priv, elapsed, rate

    gpu_full_last_key, gpu_full_time, gpu_full_rate = benchmark_gpu_full(COUNT)
    if gpu_full_last_key:
        _ = verify_last_key(gpu_full_last_key, "GPU-FULL")

    # 比較結果
    print("\n" + "=" * 60)
    print("性能比較")
    print("=" * 60)
    print(f"CPU 速率: {cpu_rate:.2f} keys/sec")
    if gpu_rate > 0:
        print(f"GPU 速率: {gpu_rate:.2f} keys/sec")
        speedup = gpu_rate / cpu_rate if cpu_rate > 0 else 0.0
        print(f"加速比: {speedup:.2f}x (GPU 比 CPU 快 {speedup:.2f} 倍)")
    else:
        print("GPU: 不可用或已略過")

    if gpu_full_rate > 0:
        s2 = gpu_full_rate / cpu_rate if cpu_rate > 0 else 0.0
        print(f"完全 GPU 管線雛形速率: {gpu_full_rate:.2f} keys/sec（相對 CPU: {s2:.2f}x）")

    print("\n準確性驗證:")
    print(f"CPU 第 100,000 筆: {'✅ 通過' if cpu_valid else '❌ 失敗'}")
    if gpu_last_key:
        print(f"GPU 第 100,000 筆: {'✅ 通過' if gpu_valid else '❌ 失敗'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
