# -*- coding: utf-8 -*-
"""
完全 GPU 加速的 TRON 地址生成（雛形 / 可執行框架）

重要說明：
- 本模組提供「介面與管線」並以 GPU 生成私鑰；secp256k1、Keccak-256、Base58Check 皆已有 CUDA 內核，同時保留 CPU 後備以確保穩定性。
- 後續將持續優化 CUDA RawKernel/RawModule，力求達成 10x-100x 整體加速目標。

功能：
- `generate_tron_addresses_gpu(count, batch_size)`：以 GPU 亂數批量產生私鑰，並導出 (hex, base58) 地址，同時返回對應私鑰，以便基準測試與驗證。

注意：
- 需安裝 CuPy；若無法載入 CuPy，請改用 `scripts/benchmark.py` 的 GPU/CPU 模式或安裝對應的 `cupy-cudaXX`。
"""
from __future__ import annotations

import os
import hashlib
import asyncio
import threading
import logging
import time
from typing import Dict, List, Tuple, Optional, Union, Sequence
from dataclasses import dataclass, field
from concurrent.futures import Future
from collections import deque
import importlib

logger = logging.getLogger(__name__)

try:
    import cupy as cp  # GPU 陣列/Kernel
except Exception as e:  # pragma: no cover
    raise ImportError("需要安裝 CuPy 才能使用 gpu_addr 模組：pip install cupy-cuda11x/12x") from e

import base58
import sha3
import numpy as np

from .hardware_config import HARDWARE_CONFIG, HardwareAdaptiveConfig

_HARDWARE_CFG = HARDWARE_CONFIG

try:
    _GPU_FREE, _GPU_TOTAL = cp.cuda.runtime.memGetInfo()
    _GPU_TOTAL = int(_GPU_TOTAL)
except Exception:
    _GPU_TOTAL = int(getattr(_HARDWARE_CFG, "total_mem_gb", 0) * (1024**3))
    _GPU_FREE = 0
_GPU_MEM_LIMIT = int(_GPU_TOTAL * 0.9) if _GPU_TOTAL else _HARDWARE_CFG.memory_pool_limit_bytes or 0
if _GPU_MEM_LIMIT <= 0 and _HARDWARE_CFG.memory_pool_limit_bytes:
    _GPU_MEM_LIMIT = int(_HARDWARE_CFG.memory_pool_limit_bytes)

_DEVICE_POOL = cp.cuda.MemoryPool()
cp.cuda.set_allocator(_DEVICE_POOL.malloc)
if _GPU_MEM_LIMIT:
    try:
        _DEVICE_POOL.set_limit(_GPU_MEM_LIMIT)
        logger.info("[TUNER] 設定 GPU 記憶體池上限為 %.2f GiB (90%%)", _GPU_MEM_LIMIT / (1024**3))
    except Exception:  # pragma: no cover
        logger.warning("Memory pool limit 設定失敗，將使用預設值", exc_info=True)
_PINNED_POOL = cp.cuda.PinnedMemoryPool()
cp.cuda.set_pinned_memory_allocator(_PINNED_POOL.malloc)
if _GPU_MEM_LIMIT:
    try:
        _PINNED_POOL.set_limit(int(_GPU_MEM_LIMIT * 0.1))
    except Exception:  # pragma: no cover
        logger.debug("Pinned pool limit 設定失敗，忽略", exc_info=True)
try:
    from .gpu_keccak import keccak256_xy_batch as _gpu_keccak256_xy_batch
except Exception:
    _gpu_keccak256_xy_batch = None
try:
    from .gpu_secp256k1 import gpu_secp256k1_batch as secp_gpu_batch
except Exception:
    secp_gpu_batch = None
try:
    from .gpu_secp256k1 import gpu_secp256k1_batch_window4 as secp_gpu_batch_w4
except Exception:
    secp_gpu_batch_w4 = None
try:
    from .gpu_secp256k1 import warmup_window4_table as _warmup_window4_table
except Exception:
    _warmup_window4_table = None

try:
    _gpu_secp256k1_mod = importlib.import_module("tron_vanity.gpu_secp256k1")
except Exception:
    _gpu_secp256k1_mod = None
try:
    _gpu_keccak_mod = importlib.import_module("tron_vanity.gpu_keccak")
except Exception:
    _gpu_keccak_mod = None

_BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


@dataclass
class _DeviceState:
    pool: "_ArrayPool"
    rng: "cp.random.Generator"


class _ArrayPool:
    """簡易 GPU 陣列池，避免重覆配置大尺寸緩衝。"""

    def __init__(self) -> None:
        self._store: Dict[Tuple[str, Optional[int]], List["cp.ndarray"]] = {}
        self._lock = threading.Lock()

    def acquire(self, rows: int, cols: Optional[int], dtype: "cp.dtype") -> Tuple["cp.ndarray", "cp.ndarray"]:
        dtype_obj = cp.dtype(dtype)
        key = (dtype_obj.str, cols)
        with self._lock:
            candidates = self._store.get(key, [])
            for idx, arr in enumerate(candidates):
                if arr.shape[0] >= rows:
                    view = arr[:rows] if cols is None else arr[:rows, :]
                    candidates.pop(idx)
                    return view, arr
        capacity = _aligned_capacity(rows)
        shape = (capacity,) if cols is None else (capacity, cols)
        arr = cp.empty(shape, dtype=dtype)
        view = arr[:rows] if cols is None else arr[:rows, :]
        return view, arr

    def release(self, owner: Optional["cp.ndarray"]) -> None:
        if owner is None:
            return
        dtype_obj = cp.dtype(owner.dtype)
        key = (dtype_obj.str, owner.shape[1] if owner.ndim == 2 else None)
        with self._lock:
            self._store.setdefault(key, []).append(owner)


_DEVICE_STATE_LOCK = threading.Lock()
_DEVICE_STATES: Dict[int, _DeviceState] = {}


def _aligned_capacity(rows: int) -> int:
    if rows <= 0:
        return 0
    block = 8192
    return ((rows + block - 1) // block) * block


def _get_device_state() -> _DeviceState:
    dev_id = int(cp.cuda.Device())
    with _DEVICE_STATE_LOCK:
        state = _DEVICE_STATES.get(dev_id)
        if state is None:
            rng = cp.random.default_rng()
            state = _DeviceState(pool=_ArrayPool(), rng=rng)
            _DEVICE_STATES[dev_id] = state
        return state


def _acquire_gpu_array(rows: int, cols: Optional[int], dtype: "cp.dtype") -> Tuple["cp.ndarray", "cp.ndarray"]:
    state = _get_device_state()
    return state.pool.acquire(rows, cols, dtype)


def _release_gpu_array(owner: Optional["cp.ndarray"]) -> None:
    if owner is None:
        return
    state = _get_device_state()
    state.pool.release(owner)


def _fill_random_uint8(arr: "cp.ndarray") -> None:
    if arr.dtype != cp.uint8:
        raise ValueError("隨機緩衝需為 uint8")
    arr[...] = cp.random.randint(0, 256, size=arr.shape, dtype=cp.uint8)


def _compute_suffix_mod_params(suffix_bytes: Optional[bytes]) -> Optional[Tuple[int, int]]:
    if not suffix_bytes:
        return None
    value = 0
    mod = 1
    for ch in reversed(suffix_bytes):
        try:
            digit = _BASE58_ALPHABET.index(chr(ch))
        except ValueError as exc:
            raise ValueError("suffix 包含非 Base58 字元") from exc
        value += digit * mod
        mod *= 58
        if mod >= (1 << 63):
            raise ValueError("suffix 長度過長，超出快速模運算支援範圍")
    return value, mod

_USE_WNAF_DEFAULT = True
if os.environ.get("VANITY_DISABLE_WNAF") == "1":
    _USE_WNAF_DEFAULT = False
elif os.environ.get("VANITY_EXPERIMENTAL_GPU_SECP") == "0":
    _USE_WNAF_DEFAULT = False

_WNAF_READY = False
_WNAF_BROKEN = False
_WNAF_SIZE_LOGGED = False
_WNAF_LAST_USED = False
_WNAF_LAST_REASON = "init"
_WNAF_SKIP_STREAK = 0

if _USE_WNAF_DEFAULT and _warmup_window4_table is not None:
    try:
        _warmup_window4_table()
        _WNAF_READY = True
        logger.info("[WNAF] 模組初始化完成 Window4 預熱")
    except Exception as exc:  # pragma: no cover
        logger.warning("[WNAF] 模組初始化預熱失敗，將回退：%s", exc)
        _WNAF_BROKEN = True


def _wnaf_record_success(batch_size: int) -> None:
    global _WNAF_ADAPTIVE_LIMIT
    if _WNAF_ADAPTIVE_LIMIT <= 0:
        return
    cap = _dynamic_cap_for_streams(_CURRENT_STREAM_COUNT)
    if batch_size > _WNAF_ADAPTIVE_LIMIT:
        _WNAF_ADAPTIVE_LIMIT = min(batch_size, cap)
        logger.info("[WNAF] 自適應門檻提升至 %d", _WNAF_ADAPTIVE_LIMIT)


def _wnaf_record_failure(batch_size: int) -> None:
    global _WNAF_ADAPTIVE_LIMIT
    if _WNAF_ADAPTIVE_LIMIT <= 0:
        return
    if batch_size >= _WNAF_ADAPTIVE_LIMIT:
        new_limit = max(_WNAF_BASE_THRESHOLD, batch_size // 2)
        cap = _dynamic_cap_for_streams(_CURRENT_STREAM_COUNT)
        new_limit = min(new_limit, cap)
        if new_limit < _WNAF_ADAPTIVE_LIMIT:
            _WNAF_ADAPTIVE_LIMIT = new_limit
            logger.warning("[WNAF] 內核在批次 %d 失敗，自適應門檻降至 %d", batch_size, _WNAF_ADAPTIVE_LIMIT)

_DEFAULT_DYNAMIC_BATCHES = _HARDWARE_CFG.default_batches

_WNAF_BATCH_THRESHOLD = int(os.environ.get("VANITY_WNAF_MAX_BATCH", str(_HARDWARE_CFG.wnaf_threshold)))
_WNAF_BASE_THRESHOLD = _WNAF_BATCH_THRESHOLD
_WNAF_ADAPTIVE_LIMIT = _WNAF_BATCH_THRESHOLD if _WNAF_BATCH_THRESHOLD > 0 else 0
_BASE_STREAM_COUNT = int(os.environ.get("VANITY_STREAM_COUNT_DEFAULT", str(_HARDWARE_CFG.default_streams)))
_MAX_PENDING_MULTIPLIER_DEFAULT = int(os.environ.get("VANITY_MAX_PENDING_MULTIPLIER", str(_HARDWARE_CFG.max_pending_multiplier)))

_SECP_THREADS = int(os.environ.get("VANITY_SECP_THREADS", str(_HARDWARE_CFG.secp_threads)))
_KECCAK_THREADS = int(os.environ.get("VANITY_KECCAK_THREADS", str(_HARDWARE_CFG.keccak_threads)))
_SHA_THREADS = int(os.environ.get("VANITY_SHA_THREADS", str(_HARDWARE_CFG.sha_threads)))
_BASE58_THREADS = int(os.environ.get("VANITY_BASE58_THREADS", str(_HARDWARE_CFG.base58_threads)))

_AUTOTUNE_DISABLED = os.environ.get("VANITY_DISABLE_AUTOTUNE") == "1"
try:
    _AUTOTUNE_TRIAL_SIZE = int(os.environ.get("VANITY_AUTOTUNE_TRIAL", "262144"))
except ValueError:
    _AUTOTUNE_TRIAL_SIZE = 262144
_AUTO_TUNING = False
_CURRENT_STREAM_COUNT = _BASE_STREAM_COUNT or 1

_PER_ITEM_ESTIMATE = 220  # bytes per item per stream（估算）
_DEF_MAX_BATCH = _HARDWARE_CFG.max_batch_size or (_DEFAULT_DYNAMIC_BATCHES[-1] if _DEFAULT_DYNAMIC_BATCHES else 262144)
if _DEF_MAX_BATCH <= 0:
    _DEF_MAX_BATCH = 262144
if _GPU_MEM_LIMIT:
    mem_based_limit = int(_GPU_MEM_LIMIT / (_PER_ITEM_ESTIMATE * max(1, _BASE_STREAM_COUNT)))
    mem_based_limit = max(256, (mem_based_limit // 256) * 256)
    if mem_based_limit > 0:
        _DEF_MAX_BATCH = max(_DEF_MAX_BATCH, mem_based_limit)
_WNAF_DYNAMIC_BASE = max(_WNAF_BATCH_THRESHOLD, int(_DEF_MAX_BATCH * 0.9))
_WNAF_DYNAMIC_BASE = max(256, (_WNAF_DYNAMIC_BASE // 256) * 256)


def _dynamic_cap_for_streams(streams: Optional[int]) -> int:
    cap = _WNAF_DYNAMIC_BASE
    stream_val = max(1, streams or _BASE_STREAM_COUNT or 1)
    if _GPU_MEM_LIMIT:
        mem_cap = int(_GPU_MEM_LIMIT / (_PER_ITEM_ESTIMATE * stream_val))
        mem_cap = max(256, (mem_cap // 256) * 256)
        cap = min(cap, mem_cap) if cap else mem_cap
    return max(256, cap)


dynamic_cap_init = _dynamic_cap_for_streams(_BASE_STREAM_COUNT)
if _WNAF_ADAPTIVE_LIMIT > 0:
    _WNAF_ADAPTIVE_LIMIT = min(_WNAF_ADAPTIVE_LIMIT, dynamic_cap_init)
elif _WNAF_BATCH_THRESHOLD > 0:
    _WNAF_ADAPTIVE_LIMIT = min(_WNAF_BATCH_THRESHOLD, dynamic_cap_init)


def _apply_thread_setting(threads: int) -> int:
    """同步更新所有與 thread 數相關的全域設定。"""

    threads = int(max(128, min(1024, threads)))
    global _SECP_THREADS, _KECCAK_THREADS, _SHA_THREADS, _BASE58_THREADS
    _SECP_THREADS = threads
    _KECCAK_THREADS = threads
    _SHA_THREADS = threads
    _BASE58_THREADS = threads
    if _gpu_secp256k1_mod is not None:
        try:
            _gpu_secp256k1_mod._SECP_THREADS = threads
        except Exception:
            logger.debug("[TUNER] 無法更新 gpu_secp256k1 threads", exc_info=True)
    if _gpu_keccak_mod is not None:
        try:
            _gpu_keccak_mod._KECCAK_THREADS = threads
        except Exception:
            logger.debug("[TUNER] 無法更新 gpu_keccak threads", exc_info=True)
    logger.info("[TUNER] kernel threads 設定為 %d", threads)
    return threads


class _PerformanceTuner:
    """根據硬體自動探索最適 streams / threads 組合。"""

    def __init__(self, cfg: "HardwareAdaptiveConfig") -> None:
        self.cfg = cfg
        self.enabled = (
            cfg.backend == "GPU"
            and not _AUTOTUNE_DISABLED
            and secp_gpu_batch is not None
            and cp is not None
        )
        self.stream_max = min(20, max(4, (cfg.sm_count or 32)))
        self.base_stream = min(self.stream_max, max(2, cfg.default_streams))
        self.stream_target = self.base_stream
        self.best_stream = self.base_stream
        self.best_thread = cfg.secp_threads or 256
        self.best_throughput = 0.0
        self.completed = False
        candidates = [cfg.max_batch_size, _AUTOTUNE_TRIAL_SIZE]
        candidates.append(_dynamic_cap_for_streams(self.base_stream))
        positives = [c for c in candidates if c and c > 0]
        self.trial_batch = min(positives) if positives else _AUTOTUNE_TRIAL_SIZE
        if self.trial_batch <= 0:
            self.trial_batch = 0
        self.stream_candidates = self._build_stream_candidates()
        self.thread_candidates = self._build_thread_candidates()
        if not self.enabled or self.trial_batch <= 0:
            self.completed = True
            _apply_thread_setting(self.best_thread)
            return
        # 預先套用預設 threads，實測後再調整
        _apply_thread_setting(self.thread_candidates[0])

    def _build_stream_candidates(self) -> List[int]:
        base = self.base_stream
        offsets = (-4, -2, 0, 2, 4)
        candidates = {max(2, min(self.stream_max, base + off)) for off in offsets}
        candidates.add(base)
        ordered = sorted(candidates)
        if base in ordered:
            ordered.remove(base)
        return [base] + ordered

    def _build_thread_candidates(self) -> List[int]:
        base = max(128, self.cfg.secp_threads or 256)
        options = {base}
        if self.cfg.compute_capability and self.cfg.compute_capability[0] >= 8:
            options.add(512)
        options.add(384)
        options = {max(128, min(1024, opt)) for opt in options}
        ordered = sorted(options)
        return ordered

    def ensure_calibrated(self) -> None:
        if not self.enabled or self.completed:
            return
        logger.info(
            "[TUNER] 自動調整啟動：候選 streams=%s, threads=%s, trial=%d",
            self.stream_candidates,
            self.thread_candidates,
            self.trial_batch,
        )
        best_stream = self.best_stream
        best_thread = self.best_thread
        best_throughput = 0.0
        for thread in self.thread_candidates:
            _apply_thread_setting(thread)
            for stream in self.stream_candidates:
                throughput = self._measure(stream)
                if throughput <= 0:
                    continue
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_stream = stream
                    best_thread = thread
        if best_throughput > 0:
            self.best_stream = best_stream
            self.best_thread = best_thread
            self.best_throughput = best_throughput
            self.stream_target = best_stream
            self.completed = True
            _apply_thread_setting(best_thread)
            global _BASE_STREAM_COUNT
            _BASE_STREAM_COUNT = best_stream
            logger.info(
                "[TUNER] 自動調整完成：最佳 streams=%d, threads=%d, throughput=%.0f addr/s",
                best_stream,
                best_thread,
                best_throughput,
            )
        else:
            logger.warning("[TUNER] 未取得有效的自動調整結果，維持預設配置")
            self.completed = True
            _apply_thread_setting(self.best_thread)

    def select_stream_count(self, override: Optional[int]) -> int:
        if override is not None:
            return int(override)
        return self.stream_target

    def record_run(self, stream_used: int, throughput: float) -> None:
        if throughput <= 0:
            return
        if throughput > self.best_throughput:
            self.best_throughput = throughput
            self.best_stream = stream_used

    def _measure(self, streams: int) -> float:
        if self.trial_batch <= 0:
            return 0.0
        batch_size = self.trial_batch
        if self.cfg.max_batch_size and self.cfg.max_batch_size > 0:
            batch_size = min(batch_size, self.cfg.max_batch_size)
        batch_size = min(batch_size, _dynamic_cap_for_streams(streams))
        if batch_size <= 0:
            return 0.0
        global _AUTO_TUNING
        prev = _AUTO_TUNING
        _AUTO_TUNING = True
        try:
            _, _, stats = generate_tron_addresses_gpu(
                count=batch_size,
                batch_size=batch_size,
                prefix=None,
                suffix=None,
                max_hits=0,
                dynamic_batches=[batch_size],
                stream_count=streams,
                return_stats=True,
            )
        except Exception as exc:  # pragma: no cover - 失敗時回退
            logger.debug("[TUNER] 測試 streams=%d threads=%d 失敗：%s", streams, _SECP_THREADS, exc)
            return 0.0
        finally:
            _AUTO_TUNING = prev
        throughput = stats["processed"] / max(stats["elapsed_sec"], 1e-6)
        logger.debug(
            "[TUNER] 測試結果 streams=%d threads=%d → %.0f addr/s",
            streams,
            _SECP_THREADS,
            throughput,
        )
        return throughput


_PERF_TUNER = _PerformanceTuner(_HARDWARE_CFG)

logger.info(
    "偵測硬體配置：%s | 預設批次=%s | Streams(base)=%d | Window4 門檻=%d",
    _HARDWARE_CFG.summary(),
    _DEFAULT_DYNAMIC_BATCHES,
    _BASE_STREAM_COUNT,
    _WNAF_BATCH_THRESHOLD,
)

_ASYNC_LOOP: Optional[asyncio.AbstractEventLoop] = None
_ASYNC_THREAD: Optional[threading.Thread] = None
_ASYNC_LOOP_LOCK = threading.Lock()

def _parse_dynamic_batches(value: str) -> Tuple[int, ...]:
    parsed: List[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            num = int(token)
        except ValueError:
            continue
        if num > 0:
            parsed.append(num)
    return tuple(dict.fromkeys(sorted(parsed)))

def _run_async_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()

def _ensure_async_loop() -> asyncio.AbstractEventLoop:
    global _ASYNC_LOOP, _ASYNC_THREAD
    if _ASYNC_LOOP is not None:
        return _ASYNC_LOOP
    with _ASYNC_LOOP_LOCK:
        if _ASYNC_LOOP is None:
            loop = asyncio.new_event_loop()
            thread = threading.Thread(target=_run_async_loop, args=(loop,), daemon=True)
            thread.start()
            _ASYNC_LOOP = loop
            _ASYNC_THREAD = thread
    return _ASYNC_LOOP

_ENV_DYNAMIC_BATCHES = ()
_env_dynamic_raw = os.environ.get("VANITY_DYNAMIC_BATCHES")
if _env_dynamic_raw:
    _ENV_DYNAMIC_BATCHES = _parse_dynamic_batches(_env_dynamic_raw)

def _prepare_batch_plan(base_batch: int, dynamic_batches: Optional[Sequence[int]]) -> List[int]:
    plan: List[int] = []
    candidates = set()
    if base_batch and base_batch > 0:
        candidates.add(int(base_batch))
    source: Optional[Sequence[int]]
    if dynamic_batches is not None:
        source = dynamic_batches
    elif _ENV_DYNAMIC_BATCHES:
        source = _ENV_DYNAMIC_BATCHES
    else:
        source = _DEFAULT_DYNAMIC_BATCHES
    for item in source:
        try:
            val = int(item)
        except (TypeError, ValueError):
            continue
        if val > 0:
            candidates.add(val)
    plan = sorted(candidates)
    if not plan:
        plan = [max(1024, base_batch or 16384)]
    return plan

def _select_batch_size(remain: int, batch_plan: Sequence[int], stream_count: int) -> int:
    if remain <= 0:
        return 0
    try:
        free_mem, _ = cp.cuda.runtime.memGetInfo()
    except Exception:
        free_mem = 0
    stream_count = max(stream_count, 1)
    limit = int(free_mem * 0.85) if free_mem else 0
    per_item = 220  # bytes（重新估算的中間緩衝）
    best = batch_plan[0]
    aggressive_candidate = None
    for cand in batch_plan:
        cand = max(256, cand)
        if cand <= 0:
            continue
        mem_need = cand * per_item * stream_count
        if limit and mem_need > limit:
            if best == batch_plan[0]:
                approx = limit // (per_item * stream_count)
                if approx > 0:
                    return max(1, min(remain, approx))
            break
        best = cand
        if cand >= remain:
            break
    if free_mem and best < remain:
        headroom = limit - (best * per_item * stream_count)
        if headroom > 0:
            extra = int(best * 1.2)
            aggressive_candidate = min(remain, extra)
    target = aggressive_candidate or min(best, remain)
    if target <= 0:
        target = min(remain, batch_plan[0])
    return max(1, target)


def _snap_batch_to_plan(value: int, batch_plan: Sequence[int]) -> int:
    if not batch_plan:
        return value
    for cand in batch_plan:
        if cand >= value:
            return cand
    return batch_plan[-1]

# -------------------------
# GPU SHA-256（單區塊訊息）
# 專為長度 <= 55 bytes 的訊息設計（例如：TRON 21 bytes 主體、或 32 bytes 中間雜湊）
# -------------------------

_SHA256_KERNEL = r"""
extern "C" __global__ void sha256_oneblock(
    const unsigned char* __restrict__ msgs,
    const int* __restrict__ lens,
    unsigned char* __restrict__ digests,
    const int stride_in,
    const int stride_out,
    const int n
){
    // 每個 thread 處理一則訊息（長度 <= 55）
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= n) return;

    // 常數初始化向量（SHA-256 IV）
    unsigned int H0 = 0x6a09e667U;
    unsigned int H1 = 0xbb67ae85U;
    unsigned int H2 = 0x3c6ef372U;
    unsigned int H3 = 0xa54ff53aU;
    unsigned int H4 = 0x510e527fU;
    unsigned int H5 = 0x9b05688cU;
    unsigned int H6 = 0x1f83d9abU;
    unsigned int H7 = 0x5be0cd19U;

    // K 常數表
    const unsigned int K256[64] = {
        0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
        0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
        0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
        0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
        0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
        0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
        0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
        0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U
    };

    const unsigned char* m = msgs + (size_t)i * (size_t)stride_in;
    int len = lens[i];

    // 建立一個 64-byte block，做 padding（len <= 55）
    unsigned char block[64];
    #pragma unroll
    for(int j=0;j<64;++j) block[j] = 0;

    for(int j=0;j<len;++j) block[j] = m[j];
    block[len] = 0x80; // append 0x80
    unsigned long long bitlen = ((unsigned long long)len) * 8ULL;
    // 長度以大端序寫入最後 8 bytes
    block[63] = (unsigned char)(bitlen & 0xFFULL);
    block[62] = (unsigned char)((bitlen >> 8) & 0xFFULL);
    block[61] = (unsigned char)((bitlen >> 16) & 0xFFULL);
    block[60] = (unsigned char)((bitlen >> 24) & 0xFFULL);
    block[59] = (unsigned char)((bitlen >> 32) & 0xFFULL);
    block[58] = (unsigned char)((bitlen >> 40) & 0xFFULL);
    block[57] = (unsigned char)((bitlen >> 48) & 0xFFULL);
    block[56] = (unsigned char)((bitlen >> 56) & 0xFFULL);

    // W 陣列
    unsigned int W[64];
    #define ROTR(x,n) (((x) >> (n)) | ((x) << (32-(n))))
    #define SHR(x,n) ((x) >> (n))

    // 讀取 block（大端）
    #pragma unroll
    for(int t=0;t<16;++t){
        int b = t*4;
        W[t] = ((unsigned int)block[b] << 24) | ((unsigned int)block[b+1] << 16) | ((unsigned int)block[b+2] << 8) | (unsigned int)block[b+3];
    }
    // 擴展 W
    #pragma unroll
    for(int t=16;t<64;++t){
        unsigned int s0 = ROTR(W[t-15],7) ^ ROTR(W[t-15],18) ^ SHR(W[t-15],3);
        unsigned int s1 = ROTR(W[t-2],17) ^ ROTR(W[t-2],19) ^ SHR(W[t-2],10);
        W[t] = (W[t-16] + s0 + W[t-7] + s1);
    }

    // 初始工作變數
    unsigned int a=H0, b=H1, c=H2, d=H3, e=H4, f=H5, g=H6, h=H7;

    #pragma unroll
    for(int t=0;t<64;++t){
        unsigned int S1 = ROTR(e,6) ^ ROTR(e,11) ^ ROTR(e,25);
        unsigned int ch = (e & f) ^ ((~e) & g);
        unsigned int temp1 = h + S1 + ch + K256[t] + W[t];
        unsigned int S0 = ROTR(a,2) ^ ROTR(a,13) ^ ROTR(a,22);
        unsigned int maj = (a & b) ^ (a & c) ^ (b & c);
        unsigned int temp2 = S0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    H0 += a; H1 += b; H2 += c; H3 += d; H4 += e; H5 += f; H6 += g; H7 += h;

    unsigned char* out = digests + (size_t)i * (size_t)stride_out;
    // 大端輸出 32 bytes
    unsigned int H[8] = {H0,H1,H2,H3,H4,H5,H6,H7};
    #pragma unroll
    for(int k=0;k<8;++k){
        out[k*4+0] = (unsigned char)((H[k] >> 24) & 0xFFU);
        out[k*4+1] = (unsigned char)((H[k] >> 16) & 0xFFU);
        out[k*4+2] = (unsigned char)((H[k] >> 8) & 0xFFU);
        out[k*4+3] = (unsigned char)(H[k] & 0xFFU);
    }
}
"""

_sha256_mod = cp.RawModule(code=_SHA256_KERNEL, options=("-std=c++11",))
_sha256_kernel = _sha256_mod.get_function("sha256_oneblock")


_BASE58_KERNEL = r"""
__constant__ unsigned char BASE58_ALPHABET[58] = {
    '1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H',
    'J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z',
    'a','b','c','d','e','f','g','h','i','j','k','m','n','o','p','q','r','s','t','u','v','w','x','y','z'
};

extern "C" __global__ void base58_encode_25(
    const unsigned char* __restrict__ inputs,
    unsigned char* __restrict__ outputs,
    const int out_stride,
    int* __restrict__ lengths,
    const int n
){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    const unsigned char* src = inputs + (size_t)idx * 25;
    unsigned char buffer[25];
    for (int i = 0; i < 25; ++i) buffer[i] = src[i];

    unsigned char digits[60];
    int zero_count = 0;
    while (zero_count < 25 && buffer[zero_count] == 0) zero_count++;

    int size = 0;
    int start = zero_count;
    while (start < 25) {
        int remainder = 0;
        for (int i = start; i < 25; ++i) {
            int value = (remainder << 8) | buffer[i];
            buffer[i] = (unsigned char)(value / 58);
            remainder = value % 58;
        }
        digits[size++] = (unsigned char)remainder;
        while (start < 25 && buffer[start] == 0) start++;
    }

    int total_len = zero_count + size;
    if (size == 0) {
        if (zero_count == 0) {
            total_len = 1;
            zero_count = 1;
        }
    }

    unsigned char* dst = outputs + (size_t)idx * out_stride;
    for (int i = 0; i < out_stride; ++i) dst[i] = 0;

    for (int i = 0; i < zero_count; ++i) {
        dst[i] = '1';
    }

    for (int i = 0; i < size; ++i) {
        dst[zero_count + i] = BASE58_ALPHABET[digits[size - 1 - i]];
    }

    lengths[idx] = total_len;
}
"""

_base58_mod = cp.RawModule(code=_BASE58_KERNEL, options=("-std=c++11",))
_base58_kernel = _base58_mod.get_function("base58_encode_25")

_BASE58_FUSED_KERNEL = r"""
#define ROTR(x,n) (((x) >> (n)) | ((x) << (32-(n))))
#define SHR(x,n) ((x) >> (n))

__device__ void sha256_single(const unsigned char* msg, int len, unsigned char out[32]) {
    const unsigned int K256[64] = {
        0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
        0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
        0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
        0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
        0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
        0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
        0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
        0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U
    };

    unsigned int H0 = 0x6a09e667U;
    unsigned int H1 = 0xbb67ae85U;
    unsigned int H2 = 0x3c6ef372U;
    unsigned int H3 = 0xa54ff53aU;
    unsigned int H4 = 0x510e527fU;
    unsigned int H5 = 0x9b05688cU;
    unsigned int H6 = 0x1f83d9abU;
    unsigned int H7 = 0x5be0cd19U;

    unsigned char block[64];
    #pragma unroll
    for (int j=0;j<64;++j) block[j]=0;
    for (int j=0;j<len;++j) block[j]=msg[j];
    block[len] = 0x80;
    unsigned long long bitlen = ((unsigned long long)len) * 8ULL;
    block[63] = (unsigned char)(bitlen & 0xFFULL);
    block[62] = (unsigned char)((bitlen >> 8) & 0xFFULL);
    block[61] = (unsigned char)((bitlen >> 16) & 0xFFULL);
    block[60] = (unsigned char)((bitlen >> 24) & 0xFFULL);
    block[59] = (unsigned char)((bitlen >> 32) & 0xFFULL);
    block[58] = (unsigned char)((bitlen >> 40) & 0xFFULL);
    block[57] = (unsigned char)((bitlen >> 48) & 0xFFULL);
    block[56] = (unsigned char)((bitlen >> 56) & 0xFFULL);

    unsigned int W[64];
    #pragma unroll
    for (int t=0;t<16;++t) {
        int b = t*4;
        W[t] = ((unsigned int)block[b] << 24) | ((unsigned int)block[b+1] << 16) | ((unsigned int)block[b+2] << 8) | (unsigned int)block[b+3];
    }
    #pragma unroll
    for (int t=16;t<64;++t) {
        unsigned int s0 = ROTR(W[t-15],7) ^ ROTR(W[t-15],18) ^ SHR(W[t-15],3);
        unsigned int s1 = ROTR(W[t-2],17) ^ ROTR(W[t-2],19) ^ SHR(W[t-2],10);
        W[t] = (W[t-16] + s0 + W[t-7] + s1);
    }

    unsigned int a=H0,b=H1,c=H2,d=H3,e=H4,f=H5,g=H6,h=H7;
    #pragma unroll
    for (int t=0;t<64;++t) {
        unsigned int S1 = ROTR(e,6) ^ ROTR(e,11) ^ ROTR(e,25);
        unsigned int ch = (e & f) ^ ((~e) & g);
        unsigned int temp1 = h + S1 + ch + K256[t] + W[t];
        unsigned int S0 = ROTR(a,2) ^ ROTR(a,13) ^ ROTR(a,22);
        unsigned int maj = (a & b) ^ (a & c) ^ (b & c);
        unsigned int temp2 = S0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    H0 += a; H1 += b; H2 += c; H3 += d; H4 += e; H5 += f; H6 += g; H7 += h;
    unsigned int H[8] = {H0,H1,H2,H3,H4,H5,H6,H7};
    #pragma unroll
    for (int k=0;k<8;++k) {
        out[k*4+0] = (unsigned char)((H[k] >> 24) & 0xFFU);
        out[k*4+1] = (unsigned char)((H[k] >> 16) & 0xFFU);
        out[k*4+2] = (unsigned char)((H[k] >> 8) & 0xFFU);
        out[k*4+3] = (unsigned char)(H[k] & 0xFFU);
    }
}

__constant__ unsigned char BASE58_ALPHABET_FUSED[58] = {
    '1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H',
    'J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z',
    'a','b','c','d','e','f','g','h','i','j','k','m','n','o','p','q','r','s','t','u','v','w','x','y','z'
};

extern "C" __global__ void tron21_to_base58(
    const unsigned char* __restrict__ tron21,
    unsigned char* __restrict__ outputs,
    int* __restrict__ lengths,
    const int stride_in,
    const int out_stride,
    const int n
){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    const unsigned char* src = tron21 + (size_t)idx * stride_in;
    unsigned char payload[25];
    for (int i=0;i<21;++i) payload[i] = src[i];

    unsigned char hash1[32];
    sha256_single(payload, 21, hash1);
    unsigned char hash2[32];
    sha256_single(hash1, 32, hash2);
    payload[21] = hash2[0];
    payload[22] = hash2[1];
    payload[23] = hash2[2];
    payload[24] = hash2[3];

    unsigned char buffer[25];
    for (int i=0;i<25;++i) buffer[i] = payload[i];
    unsigned char digits[60];
    int zero_count = 0;
    while (zero_count < 25 && buffer[zero_count] == 0) zero_count++;

    int size = 0;
    int start = zero_count;
    while (start < 25) {
        int remainder = 0;
        for (int i = start; i < 25; ++i) {
            int value = (remainder << 8) | buffer[i];
            buffer[i] = (unsigned char)(value / 58);
            remainder = value % 58;
        }
        digits[size++] = (unsigned char)remainder;
        while (start < 25 && buffer[start] == 0) start++;
    }

    int total_len = zero_count + size;
    if (size == 0) {
        if (zero_count == 0) {
            total_len = 1;
            zero_count = 1;
        }
    }

    unsigned char* dst = outputs + (size_t)idx * out_stride;
    for (int i=0;i<out_stride;++i) dst[i] = 0;
    for (int i=0;i<zero_count;++i) dst[i] = '1';
    for (int i=0;i<size;++i) dst[zero_count + i] = BASE58_ALPHABET_FUSED[digits[size-1-i]];

    lengths[idx] = total_len;
}
"""

_base58_fused_mod = cp.RawModule(code=_BASE58_FUSED_KERNEL, options=("-std=c++11",))
_base58_fused_kernel = _base58_fused_mod.get_function("tron21_to_base58")

_PREFIX_FILTER_KERNEL = r"""
extern "C" __global__ void prefix_filter(
    const unsigned char* __restrict__ ascii_matrix,
    const int* __restrict__ lens,
    const unsigned char* __restrict__ prefix,
    const int prefix_len,
    const int stride,
    const int total,
    unsigned char* __restrict__ mask_out
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    unsigned char match = 1;
    const int cur_len = lens[idx];
    if (cur_len < prefix_len) {
        match = 0;
    } else {
        const unsigned char* row = ascii_matrix + (size_t)idx * stride;
        for (int i = 0; i < prefix_len; ++i) {
            if (row[i] != prefix[i]) {
                match = 0;
                break;
            }
        }
    }
    mask_out[idx] = match;
}
"""

_prefix_filter_mod = cp.RawModule(code=_PREFIX_FILTER_KERNEL, options=("-std=c++11",))
_prefix_filter_kernel = _prefix_filter_mod.get_function("prefix_filter")

_SUFFIX_FILTER_KERNEL = r"""
extern "C" __global__ void suffix_mod_filter(
    const unsigned char* __restrict__ tron21,
    const unsigned char* __restrict__ checksum,
    unsigned char* __restrict__ mask,
    const unsigned long long mod_base,
    const unsigned long long target,
    const int n
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    unsigned long long rem = 0ULL;
    const unsigned char* tron_ptr = tron21 + (size_t)idx * 21;
    #pragma unroll
    for (int i = 0; i < 21; ++i) {
        rem = (rem * 256ULL + (unsigned long long)tron_ptr[i]) % mod_base;
    }
    const unsigned char* chk_ptr = checksum + (size_t)idx * 4;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        rem = (rem * 256ULL + (unsigned long long)chk_ptr[j]) % mod_base;
    }
    mask[idx] = (rem == target) ? 1 : 0;
}
"""

_suffix_filter_mod = cp.RawModule(code=_SUFFIX_FILTER_KERNEL, options=("-std=c++11",))
_suffix_filter_kernel = _suffix_filter_mod.get_function("suffix_mod_filter")


def gpu_sha256_oneblock_batch(
    msgs_gpu: "cp.ndarray",
    lens_gpu: "cp.ndarray",
    out: Optional["cp.ndarray"] = None,
) -> "cp.ndarray":
    """在 GPU 上計算多筆 SHA-256（限制：每筆長度 <= 55，單區塊）。
    - msgs_gpu: uint8 (N, stride_in)
    - lens_gpu: int32 (N,)
    回傳: uint8 (N, 32)
    """
    if msgs_gpu.dtype != cp.uint8 or msgs_gpu.ndim != 2:
        raise ValueError("msgs_gpu 需為 uint8 (N, stride)")
    if lens_gpu.dtype != cp.int32 or lens_gpu.ndim != 1:
        raise ValueError("lens_gpu 需為 int32 (N,)")
    n = msgs_gpu.shape[0]
    stride_in = msgs_gpu.shape[1]
    if out is not None:
        if out.dtype != cp.uint8 or out.ndim != 2 or out.shape[0] != n or out.shape[1] != 32:
            raise ValueError("out 需為 uint8 (N,32)")
        out[:, :] = 0
    else:
        out = cp.zeros((n, 32), dtype=cp.uint8)
    threads = _SHA_THREADS
    blocks = (n + threads - 1) // threads
    _sha256_kernel((blocks,), (threads,), (msgs_gpu, lens_gpu, out, cp.int32(stride_in), cp.int32(32), cp.int32(n)))
    return out


def gpu_base58_encode_batch(payload25_gpu: "cp.ndarray", out_stride: int = 40) -> Tuple["cp.ndarray", "cp.ndarray"]:
    """以 GPU 將 25-byte payload 編碼為 Base58 字串。
    回傳：(ascii_bytes (N,out_stride), lengths (N,))
    """
    if payload25_gpu.dtype != cp.uint8 or payload25_gpu.ndim != 2 or payload25_gpu.shape[1] != 25:
        raise ValueError("payload25_gpu 需為 uint8 (N,25)")
    n = payload25_gpu.shape[0]
    inputs = cp.ascontiguousarray(payload25_gpu)
    outputs = cp.zeros((n, out_stride), dtype=cp.uint8)
    lengths = cp.zeros((n,), dtype=cp.int32)
    threads = _BASE58_THREADS
    blocks = (n + threads - 1) // threads
    _base58_kernel((blocks,), (threads,), (inputs, outputs, cp.int32(out_stride), lengths, cp.int32(n)))
    return outputs, lengths


def _sha256d(data: bytes) -> bytes:
    """雙重 SHA-256（用於 Base58Check 校驗碼）。"""
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def _keccak_256(data: bytes) -> bytes:
    """計算 Keccak-256（CPU 後備）。"""
    k = sha3.keccak_256()
    k.update(data)
    return k.digest()


def gpu_secp256k1_batch(
    privkeys_gpu: "cp.ndarray",
    out: Optional["cp.ndarray"] = None,
) -> "cp.ndarray":
    """
    在 GPU 上批量計算 secp256k1 公鑰（未壓縮 65 bytes）。
    
    使用完全 GPU 實現的 CUDA kernel，包括：
    - 256-bit 大數模運算
    - 橢圓曲線點加法和倍點
    - 標量乘法（double-and-add）

    參數：
    - privkeys_gpu: `uint8` 形狀為 (N, 32) 的 GPU 陣列

    回傳：
    - `uint8` 形狀為 (N, 65) 的 GPU 陣列（未壓縮公鑰：0x04 + X(32) + Y(32)）
    """
    from .gpu_secp256k1 import gpu_secp256k1_batch as _gpu_secp256k1
    return _gpu_secp256k1(privkeys_gpu, out=out)


def gpu_keccak256_batch(
    pubkey_xy_gpu: "cp.ndarray",
    address_only: bool = False,
    out: Optional["cp.ndarray"] = None,
) -> "cp.ndarray":
    """
    批量 Keccak-256：
    - 若可用 GPU kernel，使用 keccak256_xy_batch
    - 否則退化為 CPU 計算
    """
    if pubkey_xy_gpu.dtype != cp.uint8 or pubkey_xy_gpu.ndim != 2 or pubkey_xy_gpu.shape[1] != 64:
        raise ValueError("pubkey_xy_gpu 需為 uint8 (N,64)")

    if _gpu_keccak256_xy_batch is not None:
        try:
            return _gpu_keccak256_xy_batch(pubkey_xy_gpu, address_only=address_only, out=out)
        except Exception:
            pass

    xy_cpu: np.ndarray = cp.asnumpy(pubkey_xy_gpu)
    out_len = 20 if address_only else 32
    out_cpu = np.empty((xy_cpu.shape[0], out_len), dtype=np.uint8)
    for i in range(xy_cpu.shape[0]):
        h = _keccak_256(bytes(xy_cpu[i]))
        if address_only:
            out_cpu[i, :] = np.frombuffer(h[-20:], dtype=np.uint8)
        else:
            out_cpu[i, :] = np.frombuffer(h, dtype=np.uint8)
    out_gpu = cp.asarray(out_cpu)
    if out is not None:
        out[...] = out_gpu
        return out
    return out_gpu


def _gpu_base58check_raw(
    tron21_gpu: "cp.ndarray",
    ascii_buf: Optional["cp.ndarray"] = None,
    lens_buf: Optional["cp.ndarray"] = None,
) -> Tuple["cp.ndarray", "cp.ndarray"]:
    """回傳 Base58Check 的 ASCII 緩衝與對應長度。"""
    if tron21_gpu.dtype != cp.uint8 or tron21_gpu.ndim != 2 or tron21_gpu.shape[1] != 21:
        raise ValueError("tron21_gpu 需為 uint8 (N,21)")

    N = tron21_gpu.shape[0]
    tron21_contig = cp.ascontiguousarray(tron21_gpu)
    if ascii_buf is not None:
        if ascii_buf.dtype != cp.uint8 or ascii_buf.ndim != 2 or ascii_buf.shape[1] < 60:
            raise ValueError("ascii_buf 需為 uint8 (?,60)")
        if ascii_buf.strides[1] != 1:
            raise ValueError("ascii_buf 必須為連續記憶體")
        if ascii_buf.shape[0] < N:
            raise ValueError("ascii_buf 行數不足")
        ascii_gpu = ascii_buf[:N, :]
    else:
        ascii_gpu = cp.zeros((N, 60), dtype=cp.uint8)
    if lens_buf is not None:
        if lens_buf.dtype != cp.int32 or lens_buf.ndim != 1 or lens_buf.shape[0] < N:
            raise ValueError("lens_buf 需為 int32 (N,)")
        lens_gpu = lens_buf[:N]
        lens_gpu.fill(0)
    else:
        lens_gpu = cp.zeros((N,), dtype=cp.int32)
    threads = _BASE58_THREADS
    blocks = (N + threads - 1) // threads
    try:
        _base58_fused_kernel(
            (blocks,), (threads,),
            (tron21_contig, ascii_gpu, lens_gpu, cp.int32(21), cp.int32(60), cp.int32(N))
        )
        return ascii_gpu, lens_gpu
    except Exception:
        lens = cp.full((N,), 21, dtype=cp.int32)
        d1 = gpu_sha256_oneblock_batch(tron21_contig, lens)
        lens2 = cp.full((N,), 32, dtype=cp.int32)
        d2 = gpu_sha256_oneblock_batch(d1, lens2)
        checksum_gpu = d2[:, :4]
        payload25_gpu = cp.concatenate([tron21_contig, checksum_gpu], axis=1)
        payload25_gpu = cp.ascontiguousarray(payload25_gpu)
        return gpu_base58_encode_batch(payload25_gpu)


def gpu_base58check_batch(tron21_gpu: "cp.ndarray") -> List[str]:
    """
    Base58Check 編碼：
    - 輸入為 21 bytes（0x41 + addr20），形狀 (N,21)
    - 在 GPU 上完成雙 SHA-256 與 Base58 編碼，必要時退回 CPU。
    """
    if tron21_gpu.dtype != cp.uint8 or tron21_gpu.ndim != 2 or tron21_gpu.shape[1] != 21:
        raise ValueError("tron21_gpu 需為 uint8 (N,21)")

    try:
        ascii_gpu, lens_gpu = _gpu_base58check_raw(tron21_gpu)
        ascii_cpu = cp.asnumpy(ascii_gpu)
        lens_cpu = cp.asnumpy(lens_gpu)
        out: List[str] = []
        for i in range(ascii_cpu.shape[0]):
            ln = int(lens_cpu[i])
            if ln <= 0 or ln > ascii_cpu.shape[1]:
                raise ValueError("Base58 GPU 產生長度異常")
            out.append(ascii_cpu[i, :ln].tobytes().decode())
        return out
    except Exception:
        body_cpu: np.ndarray = cp.asnumpy(tron21_gpu)
        out: List[str] = []
        for row in body_cpu:
            body = row.tobytes()
            checksum = _sha256d(body)[:4]
            out.append(base58.b58encode(body + checksum).decode())
        return out


@dataclass
class _BatchContext:
    event: cp.cuda.Event
    ascii_gpu: Optional["cp.ndarray"]
    lens_gpu: Optional["cp.ndarray"]
    tron21_gpu: "cp.ndarray"
    sk_gpu: "cp.ndarray"
    hits_idx: Optional["cp.ndarray"]
    fallback_cpu: bool
    size: int
    device_id: int
    cpu_future: Optional[Future] = None
    buffer_owners: Dict[str, Optional["cp.ndarray"]] = field(default_factory=dict)


def _release_context_buffers(ctx: _BatchContext) -> None:
    if not ctx.buffer_owners:
        return
    with cp.cuda.Device(ctx.device_id):
        for owner in ctx.buffer_owners.values():
            _release_gpu_array(owner)
    ctx.buffer_owners.clear()


def _suffix_mod_filter(tron21_gpu: "cp.ndarray", target: int, mod_base: int) -> "cp.ndarray":
    n = tron21_gpu.shape[0]
    checksum_buf, checksum_owner = _acquire_gpu_array(n, 4, cp.uint8)
    try:
        lens_buf, lens_owner = _acquire_gpu_array(n, None, cp.int32)
        hash_stage1, hash1_owner = _acquire_gpu_array(n, 32, cp.uint8)
        hash_stage2, hash2_owner = _acquire_gpu_array(n, 32, cp.uint8)
        try:
            lens_buf.fill(21)
            gpu_sha256_oneblock_batch(tron21_gpu, lens_buf, out=hash_stage1)
            lens_buf.fill(32)
            gpu_sha256_oneblock_batch(hash_stage1, lens_buf, out=hash_stage2)
            checksum_buf[:, :] = hash_stage2[:, :4]
        finally:
            _release_gpu_array(hash2_owner)
            _release_gpu_array(hash1_owner)
            _release_gpu_array(lens_owner)

        mask_buf, mask_owner = _acquire_gpu_array(n, None, cp.uint8)
        try:
            threads = _BASE58_THREADS
            blocks = (n + threads - 1) // threads
            _suffix_filter_kernel(
                (blocks,),
                (threads,),
                (
                    tron21_gpu,
                    checksum_buf,
                    mask_buf,
                    cp.uint64(mod_base),
                    cp.uint64(target),
                    cp.int32(n),
                ),
            )
            mask_bool = mask_buf.view(cp.bool_)
            hits_idx = cp.nonzero(mask_bool)[0]
        finally:
            _release_gpu_array(mask_owner)
    finally:
        _release_gpu_array(checksum_owner)
    return hits_idx


def _launch_batch(
    cur: int,
    stream: "cp.cuda.Stream",
    prefix_gpu: Optional["cp.ndarray"],
    pref_len: int,
    suffix_gpu: Optional["cp.ndarray"],
    suf_len: int,
    use_wnaf_requested: bool,
    suffix_mod_params: Optional[Tuple[int, int]],
) -> _BatchContext:
    ascii_gpu: Optional["cp.ndarray"] = None
    lens_gpu: Optional["cp.ndarray"] = None
    hits_idx: Optional["cp.ndarray"] = None
    fallback_cpu = False
    buffer_map: Dict[str, Optional["cp.ndarray"]] = {}
    global _WNAF_READY, _WNAF_BROKEN, _WNAF_LAST_USED, _WNAF_SIZE_LOGGED, _WNAF_LAST_REASON
    device_id = int(cp.cuda.Device())

    if not use_wnaf_requested and _USE_WNAF_DEFAULT and not _WNAF_SIZE_LOGGED:
        logger.info(
            "[WNAF] 批次 %d 超過門檻 %d，自動改用標準內核",
            cur,
            _WNAF_BATCH_THRESHOLD,
        )
        _WNAF_SIZE_LOGGED = True

    _WNAF_LAST_USED = False

    with stream:
        sk_gpu, sk_owner = _acquire_gpu_array(cur, 32, cp.uint8)
        buffer_map["sk"] = sk_owner
        _fill_random_uint8(sk_gpu)

        use_kernel = (
            use_wnaf_requested
            and not _WNAF_BROKEN
            and secp_gpu_batch_w4 is not None
        )
        if use_kernel and not _WNAF_READY and _warmup_window4_table is not None:
            logger.info("[WNAF] 觸發預熱，準備載入 Window4 常數表")
            try:
                _warmup_window4_table()
                _WNAF_READY = True
                logger.info("[WNAF] 常數表初始化完成，開始使用 Window4 核心")
            except Exception as exc:  # pragma: no cover
                logger.error("[WNAF] 常數表初始化失敗，回退至標準核心：%s", exc)
                _WNAF_BROKEN = True
                use_kernel = False

        if use_kernel:
            if not _WNAF_READY:
                wnaf_reason = "warmup"
            else:
                wnaf_reason = "window4"
        else:
            if not use_wnaf_requested:
                wnaf_reason = "adaptive-threshold"
            elif _WNAF_BROKEN:
                wnaf_reason = "broken-flag"
            elif secp_gpu_batch_w4 is None:
                wnaf_reason = "kernel-missing"
            else:
                wnaf_reason = "fallback"
        logger.debug(
            "[WNAF] 批次 %d 決策：use=%s reason=%s requested=%s limit=%d ready=%s broken=%s kernel=%s",
            cur,
            use_kernel,
            wnaf_reason,
            use_wnaf_requested,
            _WNAF_ADAPTIVE_LIMIT,
            _WNAF_READY,
            _WNAF_BROKEN,
            secp_gpu_batch_w4 is not None,
        )
        _WNAF_LAST_REASON = wnaf_reason

        pub65_gpu, pub_owner = _acquire_gpu_array(cur, 65, cp.uint8)
        buffer_map["pub65"] = pub_owner
        if use_kernel:
            _WNAF_SIZE_LOGGED = False
            try:
                pub65_gpu = secp_gpu_batch_w4(sk_gpu, out=pub65_gpu)
                _WNAF_LAST_USED = True
                _wnaf_record_success(cur)
            except Exception as exc:
                _WNAF_BROKEN = True
                logger.warning("[WNAF] Window4 核心執行異常(%s)，改用標準版", exc, exc_info=True)
                logger.debug("[WNAF] 批次 %d 失敗原因：%s", cur, exc)
                _WNAF_LAST_REASON = f"error:{exc}"
                _wnaf_record_failure(cur)
                if secp_gpu_batch is not None:
                    pub65_gpu = secp_gpu_batch(sk_gpu, out=pub65_gpu)
                else:
                    pub65_gpu = gpu_secp256k1_batch(sk_gpu, out=pub65_gpu)
                _WNAF_LAST_USED = False
        else:
            if secp_gpu_batch is not None:
                pub65_gpu = secp_gpu_batch(sk_gpu, out=pub65_gpu)
            else:
                pub65_gpu = gpu_secp256k1_batch(sk_gpu, out=pub65_gpu)
        logger.debug(
            "[WNAF] 批次 %d 完成：last_used=%s broken=%s", cur, _WNAF_LAST_USED, _WNAF_BROKEN
        )

        xy_gpu = pub65_gpu[:, 1:]
        tron21_gpu, tron_owner = _acquire_gpu_array(cur, 21, cp.uint8)
        buffer_map["tron21"] = tron_owner
        tron21_gpu[:, 0] = 0x41
        addr_slice = tron21_gpu[:, 1:]
        gpu_keccak256_batch(xy_gpu, address_only=True, out=addr_slice)

        if suffix_mod_params is not None and suf_len > 0:
            target_val, mod_base = suffix_mod_params
            hits_idx = _suffix_mod_filter(tron21_gpu, target_val, mod_base)

        if suffix_mod_params is None or (hits_idx is not None and hits_idx.size == 0):
            ascii_buf, ascii_owner = _acquire_gpu_array(cur, 60, cp.uint8)
            buffer_map["ascii"] = ascii_owner
            ascii_buf.fill(0)
            lens_buf, lens_owner = _acquire_gpu_array(cur, None, cp.int32)
            buffer_map["lens"] = lens_owner
            lens_buf.fill(0)

            ascii_gpu, lens_gpu = _gpu_base58check_raw(tron21_gpu, ascii_buf, lens_buf)
            if buffer_map.get("ascii") is not None and ascii_gpu.base is not buffer_map["ascii"]:
                _release_gpu_array(buffer_map.pop("ascii"))
            if buffer_map.get("lens") is not None and lens_gpu.base is not buffer_map["lens"]:
                _release_gpu_array(buffer_map.pop("lens"))

            mask: Optional["cp.ndarray"] = None
            if pref_len > 0 and prefix_gpu is not None:
                try:
                    prefix_mask_buf, prefix_owner = _acquire_gpu_array(cur, None, cp.uint8)
                    try:
                        threads_prefix = _BASE58_THREADS
                        blocks_prefix = (cur + threads_prefix - 1) // threads_prefix
                        _prefix_filter_kernel(
                            (blocks_prefix,),
                            (threads_prefix,),
                            (
                                ascii_gpu,
                                lens_gpu,
                                prefix_gpu,
                                cp.int32(pref_len),
                                cp.int32(ascii_gpu.shape[1]),
                                cp.int32(cur),
                                prefix_mask_buf,
                            ),
                        )
                        prefix_match = prefix_mask_buf.view(cp.bool_)
                        mask = prefix_match if mask is None else cp.logical_and(mask, prefix_match)
                    finally:
                        _release_gpu_array(prefix_owner)
                except Exception:
                    prefix_mask = lens_gpu >= pref_len
                    head = ascii_gpu[:, :pref_len]
                    head_cmp = cp.all(head == prefix_gpu[None, :], axis=1)
                    prefix_match = cp.logical_and(prefix_mask, head_cmp)
                    mask = prefix_match if mask is None else cp.logical_and(mask, prefix_match)
            if suf_len > 0 and suffix_gpu is not None:
                suffix_mask = lens_gpu >= suf_len
                idx = cp.arange(suf_len, dtype=cp.int32)[None, :]
                start_raw = (lens_gpu - suf_len).astype(cp.int32)
                safe_start = cp.where(
                    suffix_mask,
                    start_raw,
                    cp.zeros_like(start_raw, dtype=cp.int32),
                )[:, None]
                gather_idx = safe_start + idx
                tail = cp.take_along_axis(ascii_gpu, gather_idx, axis=1)
                tail_cmp = cp.all(tail == suffix_gpu[None, :], axis=1)
                suffix_match = cp.logical_and(suffix_mask, tail_cmp)
                mask = suffix_match if mask is None else cp.logical_and(mask, suffix_match)
            if mask is not None:
                hits_idx = cp.nonzero(mask)[0]
            elif pref_len == 0 and suf_len == 0:
                hits_idx = cp.arange(cur, dtype=cp.int64)
        else:
            if hits_idx is None:
                hits_idx = cp.empty((0,), dtype=cp.int64)
            if hits_idx.size > 0:
                ascii_buf, ascii_owner = _acquire_gpu_array(cur, 60, cp.uint8)
                buffer_map["ascii"] = ascii_owner
                ascii_buf.fill(0)
                lens_buf, lens_owner = _acquire_gpu_array(cur, None, cp.int32)
                buffer_map["lens"] = lens_owner
                lens_buf.fill(0)

                subset = tron21_gpu[hits_idx]
                ascii_sub, lens_sub = _gpu_base58check_raw(subset)
                ascii_buf[hits_idx, :] = ascii_sub
                lens_buf[hits_idx] = lens_sub
                ascii_gpu = ascii_buf
                lens_gpu = lens_buf
            else:
                ascii_gpu = None
                lens_gpu = None

        event = cp.cuda.Event()
        event.record(stream)

    return _BatchContext(
        event=event,
        ascii_gpu=ascii_gpu,
        lens_gpu=lens_gpu,
        tron21_gpu=tron21_gpu,
        sk_gpu=sk_gpu,
        hits_idx=hits_idx,
        fallback_cpu=fallback_cpu,
        size=cur,
        device_id=device_id,
        buffer_owners=buffer_map,
    )


def _process_context_cpu_sync(
    ctx: _BatchContext,
    prefix_bytes: Optional[bytes],
    prefix_str: Optional[str],
    suffix_bytes: Optional[bytes],
    suffix_str: Optional[str],
) -> Tuple[List[Tuple[str, str]], List[bytes]]:
    results_local: List[Tuple[str, str]] = []
    priv_local: List[bytes] = []
    with cp.cuda.Device(ctx.device_id):
        ctx.event.synchronize()
        try:
            if ctx.fallback_cpu or ctx.ascii_gpu is None or ctx.lens_gpu is None:
                hits_idx = ctx.hits_idx
                if hits_idx is not None:
                    hits_cpu = cp.asnumpy(hits_idx)
                    if hits_cpu.size == 0:
                        return results_local, priv_local
                    tron21_sel = cp.asnumpy(ctx.tron21_gpu[hits_cpu])
                    priv_sel = cp.asnumpy(ctx.sk_gpu[hits_cpu])
                    for body_arr, priv_arr in zip(tron21_sel, priv_sel):
                        body = body_arr.tobytes()
                        checksum = _sha256d(body)[:4]
                        addr_b58 = base58.b58encode(body + checksum).decode()
                        if prefix_bytes is not None and not addr_b58.startswith(prefix_str or ""):
                            continue
                        if suffix_bytes is not None and not addr_b58.endswith(suffix_str or ""):
                            continue
                        results_local.append((body.hex(), addr_b58))
                        priv_local.append(priv_arr.tobytes())
                else:
                    tron21_cpu = cp.asnumpy(ctx.tron21_gpu)
                    priv_cpu = cp.asnumpy(ctx.sk_gpu)
                    for i in range(ctx.size):
                        body = tron21_cpu[i].tobytes()
                        checksum = _sha256d(body)[:4]
                        addr_b58 = base58.b58encode(body + checksum).decode()
                        if prefix_bytes is not None and not addr_b58.startswith(prefix_str or ""):
                            continue
                        if suffix_bytes is not None and not addr_b58.endswith(suffix_str or ""):
                            continue
                        results_local.append((body.hex(), addr_b58))
                        priv_local.append(priv_cpu[i].tobytes())
            elif prefix_bytes is not None or suffix_bytes is not None:
                hits_idx = ctx.hits_idx
                if hits_idx is not None and hits_idx.size > 0:
                    hits_cpu = cp.asnumpy(hits_idx)
                    if hits_cpu.size > 0:
                        tron21_hits = cp.asnumpy(ctx.tron21_gpu[hits_cpu])
                        ascii_hits = cp.asnumpy(ctx.ascii_gpu[hits_cpu])
                        lens_hits = cp.asnumpy(ctx.lens_gpu[hits_cpu])
                        priv_hits = cp.asnumpy(ctx.sk_gpu[hits_cpu])
                        for i in range(tron21_hits.shape[0]):
                            addr_hex = tron21_hits[i].tobytes().hex()
                            ln = int(lens_hits[i])
                            addr_b58 = ascii_hits[i, :ln].tobytes().decode()
                            if prefix_bytes is not None and not addr_b58.startswith(prefix_str or ""):
                                continue
                            if suffix_bytes is not None and not addr_b58.endswith(suffix_str or ""):
                                continue
                            results_local.append((addr_hex, addr_b58))
                            priv_local.append(priv_hits[i].tobytes())
            else:
                tron21_cpu = cp.asnumpy(ctx.tron21_gpu)
                ascii_cpu = cp.asnumpy(ctx.ascii_gpu)
                lens_cpu = cp.asnumpy(ctx.lens_gpu)
                priv_cpu = cp.asnumpy(ctx.sk_gpu)
                index_iter = range(ctx.size)
                if ctx.hits_idx is not None and ctx.hits_idx.size > 0:
                    index_iter = cp.asnumpy(ctx.hits_idx).tolist()
                for i in index_iter:
                    addr_hex = tron21_cpu[i].tobytes().hex()
                    ln = int(lens_cpu[i])
                    addr_b58 = ascii_cpu[i, :ln].tobytes().decode()
                    if suffix_bytes is not None and not addr_b58.endswith(suffix_str or ""):
                        continue
                    results_local.append((addr_hex, addr_b58))
                    priv_local.append(priv_cpu[i].tobytes())
        finally:
            _release_context_buffers(ctx)
            ctx.ascii_gpu = None
            ctx.lens_gpu = None
            ctx.tron21_gpu = None
            ctx.sk_gpu = None
            ctx.hits_idx = None
    return results_local, priv_local


async def _process_context_async(
    ctx: _BatchContext,
    prefix_bytes: Optional[bytes],
    prefix_str: Optional[str],
    suffix_bytes: Optional[bytes],
    suffix_str: Optional[str],
) -> Tuple[List[Tuple[str, str]], List[bytes]]:
    return await asyncio.to_thread(
        _process_context_cpu_sync, ctx, prefix_bytes, prefix_str, suffix_bytes, suffix_str
    )


def _extend_results(
    pairs: List[Tuple[str, str]],
    privs: List[bytes],
    results: List[Tuple[str, str]],
    privkeys_out: List[bytes],
    max_hits: Optional[int],
) -> Tuple[bool, int]:
    limit = max_hits if max_hits is not None else None
    hits_added = 0
    done = False
    for pair, priv in zip(pairs, privs):
        if limit is not None and len(results) >= limit:
            done = True
            break
        results.append(pair)
        privkeys_out.append(priv)
        hits_added += 1
        if limit is not None and len(results) >= limit:
            done = True
            break
    return done, hits_added


def _handle_completed_context(
    ctx: _BatchContext,
    prefix_bytes: Optional[bytes],
    prefix_str: Optional[str],
    suffix_bytes: Optional[bytes],
    suffix_str: Optional[str],
    results: List[Tuple[str, str]],
    privkeys_out: List[bytes],
    max_hits: Optional[int],
) -> Tuple[bool, int]:
    try:
        pairs, privs = (
            ctx.cpu_future.result()
            if ctx.cpu_future is not None
            else _process_context_cpu_sync(ctx, prefix_bytes, prefix_str, suffix_bytes, suffix_str)
        )
    except Exception:
        pairs, privs = _process_context_cpu_sync(ctx, prefix_bytes, prefix_str, suffix_bytes, suffix_str)
    finally:
        ctx.cpu_future = None
    done, hits_added = _extend_results(pairs, privs, results, privkeys_out, max_hits)
    return done, hits_added


def _drain_ready_contexts(
    pending: List[_BatchContext],
    results: List[Tuple[str, str]],
    privkeys_out: List[bytes],
    prefix_bytes: Optional[bytes],
    prefix_str: Optional[str],
    suffix_bytes: Optional[bytes],
    suffix_str: Optional[str],
    max_hits: Optional[int],
) -> Tuple[bool, int, int]:
    done = False
    hits_total = 0
    processed = 0
    while pending:
        future = pending[0].cpu_future
        if future is None or not future.done():
            break
        ctx = pending.pop(0)
        processed += 1
        done_flag, hits_added = _handle_completed_context(
            ctx,
            prefix_bytes,
            prefix_str,
            suffix_bytes,
            suffix_str,
            results,
            privkeys_out,
            max_hits,
        )
        hits_total += hits_added
        if done_flag:
            done = True
            break
    return done, hits_total, processed


def generate_tron_addresses_gpu(
    count: int,
    batch_size: int = 16384,
    prefix: Optional[Union[str, bytes]] = None,
    suffix: Optional[Union[str, bytes]] = None,
    max_hits: Optional[int] = None,
    *,
    dynamic_batches: Optional[Sequence[int]] = None,
    stream_count: Optional[int] = None,
    return_stats: bool = False,
) -> Union[Tuple[List[Tuple[str, str]], List[bytes]], Tuple[List[Tuple[str, str]], List[bytes], Dict[str, Union[int, float, bool]]]]:
    """
    完全在 GPU 記憶體流程的雛形（保留 CPU 後備），回傳 (地址列表, 私鑰列表)：
    - 地址列表：[(hex_addr, base58_addr), ...]
    - 私鑰列表：[privkey_bytes, ...]

    若提供 prefix 或 suffix，則僅回傳符合該 Base58 前/後綴的結果，並盡量在 GPU 端完成篩選。
    新增參數：
    - dynamic_batches：可選的批次候選列表（例如 [16384,32768,65536]），會依 GPU 可用記憶體自動挑選合適批次。
    - stream_count：自訂 CUDA stream 數量（預設依前綴長度自動介於 4~8）。
    """
    if count <= 0:
        return [], []

    if not _AUTO_TUNING:
        try:
            _PERF_TUNER.ensure_calibrated()
        except Exception:  # pragma: no cover - 自動調整失敗時記錄但不中斷
            logger.warning("[TUNER] 自動調整失敗，維持預設配置", exc_info=True)

    results: List[Tuple[str, str]] = []
    privkeys_out: List[bytes] = []
    batches_executed = 0
    launch_time_total = 0.0
    wnaf_used_any = False
    wnaf_success_streak = 0
    global _WNAF_SKIP_STREAK
    wnaf_promotions = 0
    start_time_total = time.perf_counter()

    global _WNAF_ADAPTIVE_LIMIT

    logger.info(
        "[WNAF] generate_tron_addresses_gpu 呼叫：ready=%s broken=%s default=%s",
        _WNAF_READY,
        _WNAF_BROKEN,
        _USE_WNAF_DEFAULT,
    )

    prefix_bytes: Optional[bytes] = None
    prefix_gpu: Optional["cp.ndarray"] = None
    prefix_str: Optional[str] = None
    pref_len = 0
    if prefix is not None:
        prefix_bytes = prefix.encode("ascii") if isinstance(prefix, str) else prefix
        pref_len = len(prefix_bytes)
        if pref_len > 0:
            prefix_gpu = cp.asarray(np.frombuffer(prefix_bytes, dtype=np.uint8))
            prefix_str = prefix_bytes.decode("ascii", errors="ignore")
        else:
            prefix_bytes = None  # 空字串等同於無需篩選

    suffix_bytes: Optional[bytes] = None
    suffix_gpu: Optional["cp.ndarray"] = None
    suffix_str: Optional[str] = None
    suf_len = 0
    if suffix is not None:
        suffix_bytes = suffix.encode("ascii") if isinstance(suffix, str) else suffix
        suf_len = len(suffix_bytes)
        if suf_len > 0:
            suffix_gpu = cp.asarray(np.frombuffer(suffix_bytes, dtype=np.uint8))
            suffix_str = suffix_bytes.decode("ascii", errors="ignore")
        else:
            suffix_bytes = None

    suffix_mod_params: Optional[Tuple[int, int]] = None
    if suf_len > 0 and suffix_bytes is not None:
        try:
            suffix_mod_params = _compute_suffix_mod_params(suffix_bytes)
        except ValueError as exc:
            logger.debug("suffix 模運算不可用：%s", exc)
            suffix_mod_params = None

    processed = 0
    done = False

    batch_plan = _prepare_batch_plan(batch_size, dynamic_batches)
    cap_guess = _dynamic_cap_for_streams(None)
    batch_plan = [min(int(max(1, val)), cap_guess) for val in batch_plan]
    batch_plan = sorted(set(batch_plan))
    if batch_plan and batch_plan[-1] < cap_guess:
        batch_plan.append(cap_guess)
    elif not batch_plan:
        batch_plan = [min(batch_size or 16384, cap_guess)]

    env_stream_override = os.environ.get("VANITY_GPU_STREAMS")
    tuner_ready = getattr(_PERF_TUNER, "completed", False)
    tuned_stream = getattr(_PERF_TUNER, "stream_target", _BASE_STREAM_COUNT)

    if env_stream_override:
        try:
            stream_count = int(env_stream_override)
        except ValueError:
            stream_count = None
    elif stream_count is not None:
        try:
            stream_count = int(stream_count)
        except ValueError:
            stream_count = None

    if stream_count is None:
        stream_count = int(tuned_stream) if tuner_ready and tuned_stream else _BASE_STREAM_COUNT

    stream_count = max(2, int(stream_count))

    max_candidate = batch_plan[-1] if batch_plan else batch_size
    if prefix_bytes is not None:
        target = 6 if pref_len <= 3 else 8
        stream_count = max(stream_count, target)
    elif not tuner_ready:
        if max_candidate >= 393216:
            stream_count = max(stream_count, 12)
        elif max_candidate >= 262144:
            stream_count = max(stream_count, 10)
        elif max_candidate >= 131072:
            stream_count = max(stream_count, 8)

    stream_count = max(2, min(20, stream_count))
    global _CURRENT_STREAM_COUNT
    _CURRENT_STREAM_COUNT = stream_count
    dynamic_cap = _dynamic_cap_for_streams(stream_count)

    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(stream_count)]
    pending: List[_BatchContext] = []
    stream_idx = 0

    use_wnaf_kernel = _USE_WNAF_DEFAULT
    if use_wnaf_kernel:
        logger.info("[WNAF] 預設啟用 Window4 核心，等待預熱完成")
    else:
        logger.info("[WNAF] 已停用 Window4 核心，改用標準 secp256k1 內核")
    if use_wnaf_kernel:
        if _WNAF_ADAPTIVE_LIMIT > 0:
            logger.info("[WNAF] 初始門檻：批次 > %d 將改用標準內核", _WNAF_ADAPTIVE_LIMIT)
        else:
            logger.info("[WNAF] 未設定門檻，將嘗試所有批次使用 Window4")
    async_loop = _ensure_async_loop()
    base_pending_limit = max(stream_count, 1) * _MAX_PENDING_MULTIPLIER_DEFAULT
    pending_limit = base_pending_limit
    pending_block_events = 0
    pending_relax_events = 0
    next_batch = batch_plan[0]
    max_batch_plan = batch_plan[-1]
    no_hit_streak = 0

    while processed < count and not done:
        remain = count - processed
        desired = min(next_batch, remain)
        if desired < batch_plan[0]:
            desired = batch_plan[0] if remain >= batch_plan[0] else remain
        desired = max(1, int(desired))
        desired = min(desired, dynamic_cap)
        plan_with_desired = sorted(set(batch_plan + [desired]))
        cur = _select_batch_size(remain, plan_with_desired, stream_count)
        cur = min(cur, remain)
        cur = max(1, cur)
        cur = min(cur, dynamic_cap)
        processed += cur

        stream = streams[stream_idx % stream_count]
        stream_idx += 1
        adaptive_limit = _WNAF_ADAPTIVE_LIMIT
        use_wnaf_this_batch = use_wnaf_kernel and (adaptive_limit <= 0 or cur <= adaptive_limit)
        threshold_skipped = False
        if (
            use_wnaf_kernel
            and not _WNAF_BROKEN
            and adaptive_limit > 0
            and cur > adaptive_limit
        ):
            threshold_skipped = True
            _WNAF_SKIP_STREAK += 1
            logger.debug(
                "[WNAF] 批次 %d 超出門檻 %d（連續 %d 次）",
                cur,
                adaptive_limit,
                _WNAF_SKIP_STREAK,
            )
            if _WNAF_SKIP_STREAK >= 3:
                new_limit = _snap_batch_to_plan(cur, batch_plan)
                new_limit = min(new_limit, dynamic_cap)
                if new_limit > _WNAF_ADAPTIVE_LIMIT:
                    wnaf_promotions += 1
                    _WNAF_ADAPTIVE_LIMIT = new_limit
                    logger.info(
                        "[WNAF] 長批次連續 %d 次觸發，門檻提升至 %d",
                        _WNAF_SKIP_STREAK,
                        _WNAF_ADAPTIVE_LIMIT,
                    )
                adaptive_limit = _WNAF_ADAPTIVE_LIMIT
                use_wnaf_this_batch = use_wnaf_kernel and (
                    adaptive_limit <= 0 or cur <= adaptive_limit
                )
                next_batch = max(next_batch, _snap_batch_to_plan(adaptive_limit, batch_plan))
                _WNAF_SKIP_STREAK = 0
        if not threshold_skipped:
            if use_wnaf_this_batch or not use_wnaf_kernel or _WNAF_BROKEN:
                _WNAF_SKIP_STREAK = 0

        launch_start = time.perf_counter()
        ctx = _launch_batch(
            cur,
            stream,
            prefix_gpu,
            pref_len,
            suffix_gpu,
            suf_len,
            use_wnaf_this_batch,
            suffix_mod_params,
        )
        launch_time_total += time.perf_counter() - launch_start
        batches_executed += 1
        wnaf_used_any = wnaf_used_any or _WNAF_LAST_USED
        if use_wnaf_this_batch:
            if _WNAF_LAST_USED:
                wnaf_success_streak += 1
            else:
                wnaf_success_streak = 0
        elif _WNAF_BROKEN:
            wnaf_success_streak = 0
        if (
            use_wnaf_kernel
            and not _WNAF_BROKEN
            and _WNAF_LAST_USED
            and wnaf_success_streak >= 3
            and _WNAF_ADAPTIVE_LIMIT > 0
            and _WNAF_ADAPTIVE_LIMIT < max_batch_plan
        ):
            promote_target = max(cur, int(_WNAF_ADAPTIVE_LIMIT * 3 // 2))
            new_limit = _snap_batch_to_plan(promote_target, batch_plan)
            new_limit = min(new_limit, dynamic_cap)
            if new_limit > _WNAF_ADAPTIVE_LIMIT:
                wnaf_promotions += 1
                _WNAF_ADAPTIVE_LIMIT = new_limit
                logger.info(
                    "[WNAF] 連續成功 %d 次，門檻提升至 %d",
                    wnaf_success_streak,
                    _WNAF_ADAPTIVE_LIMIT,
                )
                next_batch = max(next_batch, _snap_batch_to_plan(_WNAF_ADAPTIVE_LIMIT, batch_plan))
            wnaf_success_streak = 0
        ctx.cpu_future = asyncio.run_coroutine_threadsafe(
            _process_context_async(ctx, prefix_bytes, prefix_str, suffix_bytes, suffix_str), async_loop
        )
        pending.append(ctx)

        if len(pending) >= pending_limit:
            pending_block_events += 1
        else:
            pending_block_events = max(0, pending_block_events - 1)

        drain_done, hits_added, processed_ctx = _drain_ready_contexts(
            pending,
            results,
            privkeys_out,
            prefix_bytes,
            prefix_str,
            suffix_bytes,
            suffix_str,
            max_hits,
        )
        if processed_ctx:
            if hits_added == 0:
                no_hit_streak += processed_ctx
                if no_hit_streak >= max(3, stream_count):
                    desired_growth = min(max_batch_plan, int(next_batch * 1.5))
                    next_batch = _snap_batch_to_plan(desired_growth, batch_plan)
                    if _WNAF_ADAPTIVE_LIMIT > 0 and next_batch > _WNAF_ADAPTIVE_LIMIT:
                        _WNAF_ADAPTIVE_LIMIT = min(next_batch, dynamic_cap)
                        logger.info("[WNAF] 自適應門檻提升至 %d (探索較大批次)", _WNAF_ADAPTIVE_LIMIT)
                    no_hit_streak = 0
            else:
                no_hit_streak = 0
                next_batch = max(next_batch, cur)
        if drain_done:
            done = True
            break

        if len(pending) >= pending_limit:
            if pending_block_events >= 8 and pending_limit < stream_count * 8:
                pending_limit += stream_count
                pending_block_events = 0
            ctx_wait = pending.pop(0)
            wait_done, hits_added_wait = _handle_completed_context(
                ctx_wait,
                prefix_bytes,
                prefix_str,
                suffix_bytes,
                suffix_str,
                results,
                privkeys_out,
                max_hits,
            )
            if hits_added_wait > 0:
                no_hit_streak = 0
                next_batch = batch_plan[0]
            else:
                no_hit_streak += 1
                if no_hit_streak >= max(3, stream_count):
                    desired_growth = min(max_batch_plan, int(next_batch * 1.5))
                    next_batch = _snap_batch_to_plan(desired_growth, batch_plan)
                    if _WNAF_ADAPTIVE_LIMIT > 0 and next_batch > _WNAF_ADAPTIVE_LIMIT:
                        _WNAF_ADAPTIVE_LIMIT = min(next_batch, dynamic_cap)
                        logger.info("[WNAF] 自適應門檻提升至 %d (pending flush)", _WNAF_ADAPTIVE_LIMIT)
                    no_hit_streak = 0
            if wait_done:
                done = True
                break
        else:
            if len(pending) <= stream_count and pending_limit > base_pending_limit:
                pending_relax_events += 1
                if pending_relax_events >= 16:
                    pending_limit = max(base_pending_limit, pending_limit - stream_count)
                    pending_relax_events = 0
            else:
                pending_relax_events = 0

        if max_hits is not None and len(results) >= max_hits:
            done = True
            break

    while pending and not done:
        ctx = pending.pop(0)
        done_flag_rest, hits_added_rest = _handle_completed_context(
            ctx,
            prefix_bytes,
            prefix_str,
            suffix_bytes,
            suffix_str,
            results,
            privkeys_out,
            max_hits,
        )
        if hits_added_rest > 0:
            no_hit_streak = 0
            restored = _snap_batch_to_plan(ctx.size, batch_plan)
            next_batch = max(next_batch, restored)
        else:
            no_hit_streak += 1
            if no_hit_streak >= max(3, stream_count):
                desired_growth = min(max_batch_plan, int(next_batch * 1.5))
                next_batch = _snap_batch_to_plan(desired_growth, batch_plan)
                if _WNAF_ADAPTIVE_LIMIT > 0 and next_batch > _WNAF_ADAPTIVE_LIMIT:
                    _WNAF_ADAPTIVE_LIMIT = min(next_batch, dynamic_cap)
                    logger.info("[WNAF] 自適應門檻提升至 %d (flush)", _WNAF_ADAPTIVE_LIMIT)
                no_hit_streak = 0
        if done_flag_rest:
            done = True
            break

    total_elapsed = time.perf_counter() - start_time_total
    throughput_current = processed / max(total_elapsed, 1e-6) if processed > 0 else 0.0
    if not _AUTO_TUNING:
        try:
            _PERF_TUNER.record_run(stream_count, throughput_current)
        except Exception:  # pragma: no cover
            logger.debug("[TUNER] 記錄執行情況時發生例外", exc_info=True)
    if return_stats:
        stats = {
            "requested": count,
            "processed": processed,
            "batches": batches_executed,
            "elapsed_sec": total_elapsed,
            "launch_sec": launch_time_total,
            "throughput": throughput_current,
            "wnaf_used": wnaf_used_any,
            "wnaf_limit": _WNAF_ADAPTIVE_LIMIT,
            "wnaf_base": _WNAF_BASE_THRESHOLD,
            "wnaf_last_used": _WNAF_LAST_USED,
            "wnaf_skip_streak": _WNAF_SKIP_STREAK,
            "wnaf_success_streak": wnaf_success_streak,
            "wnaf_promotions": wnaf_promotions,
            "wnaf_last_reason": _WNAF_LAST_REASON,
            "wnaf_kernel_enabled": use_wnaf_kernel,
            "wnaf_broken": _WNAF_BROKEN,
            "pending_limit": pending_limit,
            "pending_base": base_pending_limit,
            "next_batch": next_batch,
            "streams": stream_count,
            "max_plan": max_batch_plan,
            "last_batch_size": locals().get("cur", 0),
        }
        return results, privkeys_out, stats
    return results, privkeys_out


__all__ = [
    "gpu_secp256k1_batch",
    "gpu_keccak256_batch",
    "gpu_base58_encode_batch",
    "gpu_base58check_batch",
    "generate_tron_addresses_gpu",
    "_WNAF_READY",
    "_WNAF_BROKEN",
    "_WNAF_LAST_USED",
    "_WNAF_BATCH_THRESHOLD",
]
