# -*- coding: utf-8 -*-
"""GPU 管線效能評估腳本。

流程：
- 比較 Window4 與標準 secp256k1 核心效能。
- 測試多組批次大小，找出最佳吞吐量。
- 評估 Keccak 32-byte 與 20-byte 版本的 GPU 與記憶體傳輸耗時。

執行範例：
    PYTHONPATH=src python3 scripts/benchmark_gpu_pipeline.py
"""
from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cupy as cp

os.environ.setdefault("VANITY_WNAF_MAX_BATCH", "524288")

from tron_vanity import gpu_addr
from tron_vanity.hardware_config import get_hardware_config
from tron_vanity.gpu_secp256k1 import (
    gpu_secp256k1_batch,
    gpu_secp256k1_batch_window4,
    warmup_window4_table,
)
from tron_vanity.gpu_keccak import keccak256_xy_batch
from tron_vanity.addr import privkey_to_tron_address


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("benchmark")
CFG = get_hardware_config()
LOGGER.info("Hardware Profile: %s", CFG.summary())


def _sync() -> None:
    """確保 GPU 任務完成再量測時間。"""
    cp.cuda.Device().synchronize()


def benchmark_secp(batch: int) -> Dict[str, float]:
    """量測 Window4 與標準版 secp256k1 核心的平均耗時。"""
    LOGGER.info("[SECP] 測試批次 %d", batch)
    secrets = cp.random.randint(0, 256, size=(batch, 32), dtype=cp.uint8)

    warmup_window4_table(force=True)

    _sync()
    t0 = time.perf_counter()
    gpu_secp256k1_batch_window4(secrets)
    _sync()
    wnaf_time = time.perf_counter() - t0

    _sync()
    t1 = time.perf_counter()
    gpu_secp256k1_batch(secrets)
    _sync()
    base_time = time.perf_counter() - t1

    return {
        "batch": batch,
        "window4_time": wnaf_time,
        "window4_throughput": batch / wnaf_time,
        "baseline_time": base_time,
        "baseline_throughput": batch / base_time,
        "speedup": base_time / wnaf_time,
    }


def benchmark_pipeline(batch: int, use_wnaf: bool) -> Tuple[float, float]:
    """量測完整管線吞吐量，回傳 (耗時, 吞吐)。"""
    prev_default = gpu_addr._USE_WNAF_DEFAULT
    prev_ready = gpu_addr._WNAF_READY
    prev_broken = gpu_addr._WNAF_BROKEN

    gpu_addr._USE_WNAF_DEFAULT = use_wnaf
    gpu_addr._WNAF_BROKEN = False
    gpu_addr._WNAF_READY = False

    prefix = "T~"  # 不會命中，避免輸出大量資料
    dynamic = (batch,)

    _sync()
    t0 = time.perf_counter()
    gpu_addr.generate_tron_addresses_gpu(
        count=batch,
        batch_size=batch,
        prefix=prefix,
        max_hits=None,
        dynamic_batches=dynamic,
        stream_count=6,
    )
    _sync()
    elapsed = time.perf_counter() - t0
    throughput = batch / elapsed
    LOGGER.info(
        "[PIPE] 批次 %d / WNAF=%s -> %.2f ms, %.2f addr/s",
        batch,
        use_wnaf,
        elapsed * 1e3,
        throughput,
    )

    gpu_addr._USE_WNAF_DEFAULT = prev_default
    gpu_addr._WNAF_READY = prev_ready
    gpu_addr._WNAF_BROKEN = prev_broken
    return elapsed, throughput


def benchmark_prefix(prefix: str, total: int = 100000, max_hits: int = 5) -> None:
    LOGGER.info("[PREFIX] 測試前綴 '%s'，樣本 %d", prefix, total)
    addrs, privs = gpu_addr.generate_tron_addresses_gpu(
        total,
        min(total, CFG.default_batches[0] if CFG.default_batches else 32768),
        prefix=prefix,
        max_hits=max_hits,
    )
    gpu_hits = len(addrs)
    bad_gpu = [b58 for _, b58 in addrs if not b58.startswith(prefix)]
    cpu_mismatch = []
    for (_, b58_addr), priv in zip(addrs, privs):
        _, cpu_addr = privkey_to_tron_address(priv)
        if cpu_addr != b58_addr or not cpu_addr.startswith(prefix):
            cpu_mismatch.append((b58_addr, cpu_addr))
    LOGGER.info(
        "[PREFIX] '%s' GPU 命中 %d 筆，GPU 驗證異常 %d，CPU 驗證異常 %d",
        prefix,
        gpu_hits,
        len(bad_gpu),
        len(cpu_mismatch),
    )
    if gpu_hits > 0:
        LOGGER.info("[PREFIX] 範例：%s", [b58 for _, b58 in addrs[: min(3, gpu_hits)]])


@dataclass
class KeccakResult:
    batch: int
    gpu_time_full: float
    gpu_time_addr: float
    copy_time_full: float
    copy_time_addr: float


def benchmark_keccak(batch: int) -> KeccakResult:
    """比較 Keccak 完整輸出與地址輸出的 GPU 與資料傳輸耗時。"""
    xy = cp.random.randint(0, 256, size=(batch, 64), dtype=cp.uint8)

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record()
    full = keccak256_xy_batch(xy, address_only=False)
    end.record()
    end.synchronize()
    gpu_full = cp.cuda.get_elapsed_time(start, end) / 1000.0

    _sync()
    t0 = time.perf_counter()
    _ = cp.asnumpy(full)
    copy_full = time.perf_counter() - t0

    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    addr = keccak256_xy_batch(xy, address_only=True)
    end.record()
    end.synchronize()
    gpu_addr_t = cp.cuda.get_elapsed_time(start, end) / 1000.0

    _sync()
    t1 = time.perf_counter()
    _ = cp.asnumpy(addr)
    copy_addr = time.perf_counter() - t1

    LOGGER.info(
        "[KECCAK] 批次 %d -> full=%.3f ms/%.3f ms, addr=%.3f ms/%.3f ms",
        batch,
        gpu_full * 1e3,
        copy_full * 1e3,
        gpu_addr_t * 1e3,
        copy_addr * 1e3,
    )

    return KeccakResult(batch, gpu_full, gpu_addr_t, copy_full, copy_addr)


def main() -> None:
    os.environ.setdefault("VANITY_EXPERIMENTAL_GPU_SECP", "1")

    LOGGER.info("===== Window4 vs Baseline (secp256k1) =====")
    for batch in (16384, 32768):
        stats = benchmark_secp(batch)
        LOGGER.info(
            "批次 %(batch)d -> Window4 %(window4_throughput).0f keys/s, Baseline %(baseline_throughput).0f keys/s, Speedup x%(speedup).2f",
            stats,
        )

    LOGGER.info("===== Pipeline Throughput (32k/64k/128k/256k) =====")
    best = None
    for batch in (32768, 65536, 131072, 262144):
        _, tp_wnaf = benchmark_pipeline(batch, use_wnaf=True)
        _, tp_base = benchmark_pipeline(batch, use_wnaf=False)
        if best is None or tp_wnaf > best[1]:
            best = (batch, tp_wnaf)
        LOGGER.info(
            "[PIPE] 批次 %d -> Window4 %.0f addr/s, Baseline %.0f addr/s",
            batch,
            tp_wnaf,
            tp_base,
        )

    if best:
        LOGGER.info("最佳批次：%d (%.0f addr/s)", best[0], best[1])

    LOGGER.info("===== 前綴搜索驗證 =====")
    for prefix in ("T", "T7", "T77"):
        benchmark_prefix(prefix)

    LOGGER.info("===== Keccak 比較 (64-byte -> hash) =====")
    for batch in (65536, 131072):
        benchmark_keccak(batch)


if __name__ == "__main__":
    main()
