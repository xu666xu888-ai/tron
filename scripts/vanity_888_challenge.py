#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRON 靚號挑戰：搜尋末三碼為 888 的地址並輸出完整驗證資訊。

使用方式：
    VANITY_EXPERIMENTAL_GPU_SECP=1 PYTHONPATH=src python scripts/vanity_888_challenge.py
"""
from __future__ import annotations

import os
import time
from typing import Tuple

from tron_vanity.gpu_addr import generate_tron_addresses_gpu
from tron_vanity.validate import validate_private_key
from tron_vanity.addr import is_valid_tron_base58


TARGET_SUFFIX = "888"
# 針對 3 位後綴，理論期望約 195,112 筆；大批次有助 GPU 利用率。
BATCH_SIZE = 131_072
REPORT_INTERVAL = 5  # 每 5 筆報告一次（依批次而定）


def main() -> int:
    os.environ.setdefault("VANITY_EXPERIMENTAL_GPU_SECP", "1")

    total_checked = 0
    peak_rate = 0.0
    best_hit: Tuple[str, str, bytes] | None = None

    start_time = time.perf_counter()
    loop_index = 0

    while True:
        loop_index += 1

        batch_start = time.perf_counter()
        addrs, privs = generate_tron_addresses_gpu(BATCH_SIZE, batch_size=BATCH_SIZE)
        batch_time = time.perf_counter() - batch_start

        total_checked += len(addrs)
        current_rate = len(addrs) / batch_time if batch_time > 0 else 0.0
        peak_rate = max(peak_rate, current_rate)

        if loop_index % REPORT_INTERVAL == 0:
            elapsed = time.perf_counter() - start_time
            avg_rate = total_checked / elapsed if elapsed > 0 else 0.0
            print(
                f"[進度] 已檢查 {total_checked:,} 筆 (平均 {avg_rate:,.0f} addr/s，"
                f"當前批次 {current_rate:,.0f} addr/s)"
            )

        for (hex_addr, b58_addr), priv in zip(addrs, privs):
            if b58_addr.endswith(TARGET_SUFFIX):
                best_hit = (hex_addr, b58_addr, priv)
                break

        if best_hit is not None:
            break

    total_time = time.perf_counter() - start_time
    avg_rate = total_checked / total_time if total_time > 0 else 0.0

    hex_addr, b58_addr, priv_bytes = best_hit

    validation = validate_private_key(priv_bytes)
    tronpy_match = validation.get("tronpy_match")
    validate_address = validation.get("validateaddress")
    tronpy_addr = validation.get("tronpy_base58")

    print("\n🎯 TRON 靚號搜索：末三碼 888")
    print("=" * 80)
    print(
        f"總嘗試數：{total_checked:,} │ 總耗時：{total_time:.2f} 秒 │ "
        f"平均速率：{avg_rate:,.0f} addr/s │ 峰值速率：{peak_rate:,.0f} addr/s"
    )
    print("-" * 80)
    print(f"靚號地址：{b58_addr}")
    print(f"地址 HEX：{hex_addr}")
    print(f"私鑰 HEX：{priv_bytes.hex()}")
    print("-" * 80)
    print(f"tronpy 導出地址：{tronpy_addr}")
    print(f"tronpy 比對：{tronpy_match}")
    print(f"節點 validateaddress：{validate_address}")
    print(f"Base58 格式檢查：{is_valid_tron_base58(b58_addr)}")
    print("=" * 80)
    print("⚠️ 請妥善保存私鑰。建議於安全環境中使用實際資產。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
