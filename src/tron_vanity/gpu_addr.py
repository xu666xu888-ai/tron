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
from typing import List, Tuple, Optional, Union, Sequence
from dataclasses import dataclass
from concurrent.futures import Future

logger = logging.getLogger(__name__)

try:
    import cupy as cp  # GPU 陣列/Kernel
except Exception as e:  # pragma: no cover
    raise ImportError("需要安裝 CuPy 才能使用 gpu_addr 模組：pip install cupy-cuda11x/12x") from e

import base58
import sha3
import numpy as np

from .hardware_config import HARDWARE_CONFIG

_HARDWARE_CFG = HARDWARE_CONFIG

_DEVICE_POOL = cp.cuda.MemoryPool()
cp.cuda.set_allocator(_DEVICE_POOL.malloc)
if _HARDWARE_CFG.memory_pool_limit_bytes:
    try:
        _DEVICE_POOL.set_limit(_HARDWARE_CFG.memory_pool_limit_bytes)
    except Exception:  # pragma: no cover
        logger.warning("Memory pool limit 設定失敗，將使用預設值", exc_info=True)
_PINNED_POOL = cp.cuda.PinnedMemoryPool()
cp.cuda.set_pinned_memory_allocator(_PINNED_POOL.malloc)
if _HARDWARE_CFG.memory_pool_limit_bytes:
    try:
        _PINNED_POOL.set_limit(int(_HARDWARE_CFG.memory_pool_limit_bytes * 0.1))
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

_USE_WNAF_DEFAULT = True
if os.environ.get("VANITY_DISABLE_WNAF") == "1":
    _USE_WNAF_DEFAULT = False
elif os.environ.get("VANITY_EXPERIMENTAL_GPU_SECP") == "0":
    _USE_WNAF_DEFAULT = False

_WNAF_READY = False
_WNAF_BROKEN = False
_WNAF_SIZE_LOGGED = False
_WNAF_LAST_USED = False

_DEFAULT_DYNAMIC_BATCHES = _HARDWARE_CFG.default_batches

_WNAF_BATCH_THRESHOLD = int(os.environ.get("VANITY_WNAF_MAX_BATCH", str(_HARDWARE_CFG.wnaf_threshold)))
_BASE_STREAM_COUNT = int(os.environ.get("VANITY_STREAM_COUNT_DEFAULT", str(_HARDWARE_CFG.default_streams)))
_MAX_PENDING_MULTIPLIER_DEFAULT = int(os.environ.get("VANITY_MAX_PENDING_MULTIPLIER", str(_HARDWARE_CFG.max_pending_multiplier)))

_SECP_THREADS = int(os.environ.get("VANITY_SECP_THREADS", str(_HARDWARE_CFG.secp_threads)))
_KECCAK_THREADS = int(os.environ.get("VANITY_KECCAK_THREADS", str(_HARDWARE_CFG.keccak_threads)))
_SHA_THREADS = int(os.environ.get("VANITY_SHA_THREADS", str(_HARDWARE_CFG.sha_threads)))
_BASE58_THREADS = int(os.environ.get("VANITY_BASE58_THREADS", str(_HARDWARE_CFG.base58_threads)))

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
    limit = int(free_mem * 0.8) if free_mem else 0
    per_item = 384  # bytes（估算整體中間緩衝）
    best = batch_plan[0]
    for cand in batch_plan:
        cand = max(256, cand)
        if cand <= 0:
            continue
        mem_need = cand * per_item * max(stream_count, 1)
        if limit and mem_need > limit:
            if best == batch_plan[0]:
                approx = limit // (per_item * max(stream_count, 1))
                if approx > 0:
                    return max(1, min(remain, approx))
            break
        best = cand
        if cand >= remain:
            break
    target = min(best, remain)
    if target <= 0:
        target = min(remain, batch_plan[0])
    return max(1, target)

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


def gpu_sha256_oneblock_batch(msgs_gpu: "cp.ndarray", lens_gpu: "cp.ndarray") -> "cp.ndarray":
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


def gpu_secp256k1_batch(privkeys_gpu: "cp.ndarray") -> "cp.ndarray":
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
    return _gpu_secp256k1(privkeys_gpu)


def gpu_keccak256_batch(pubkey_xy_gpu: "cp.ndarray", address_only: bool = False) -> "cp.ndarray":
    """
    批量 Keccak-256：
    - 若可用 GPU kernel，使用 keccak256_xy_batch
    - 否則退化為 CPU 計算
    """
    if pubkey_xy_gpu.dtype != cp.uint8 or pubkey_xy_gpu.ndim != 2 or pubkey_xy_gpu.shape[1] != 64:
        raise ValueError("pubkey_xy_gpu 需為 uint8 (N,64)")

    if _gpu_keccak256_xy_batch is not None:
        try:
            return _gpu_keccak256_xy_batch(pubkey_xy_gpu, address_only=address_only)
        except Exception:
            pass

    xy_cpu: np.ndarray = cp.asnumpy(pubkey_xy_gpu)
    out_len = 20 if address_only else 32
    out = np.empty((xy_cpu.shape[0], out_len), dtype=np.uint8)
    for i in range(xy_cpu.shape[0]):
        h = _keccak_256(bytes(xy_cpu[i]))
        if address_only:
            out[i, :] = np.frombuffer(h[-20:], dtype=np.uint8)
        else:
            out[i, :] = np.frombuffer(h, dtype=np.uint8)
    return cp.asarray(out)


def _gpu_base58check_raw(tron21_gpu: "cp.ndarray") -> Tuple["cp.ndarray", "cp.ndarray"]:
    """回傳 Base58Check 的 ASCII 緩衝與對應長度。"""
    if tron21_gpu.dtype != cp.uint8 or tron21_gpu.ndim != 2 or tron21_gpu.shape[1] != 21:
        raise ValueError("tron21_gpu 需為 uint8 (N,21)")

    N = tron21_gpu.shape[0]
    tron21_contig = cp.ascontiguousarray(tron21_gpu)
    ascii_gpu = cp.zeros((N, 60), dtype=cp.uint8)
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


def _launch_batch(
    cur: int,
    stream: "cp.cuda.Stream",
    prefix_gpu: Optional["cp.ndarray"],
    pref_len: int,
    suffix_gpu: Optional["cp.ndarray"],
    suf_len: int,
    use_wnaf_requested: bool,
) -> _BatchContext:
    ascii_gpu: Optional["cp.ndarray"] = None
    lens_gpu: Optional["cp.ndarray"] = None
    hits_idx: Optional["cp.ndarray"] = None
    fallback_cpu = False
    global _WNAF_READY, _WNAF_BROKEN, _WNAF_LAST_USED
    device_id = int(cp.cuda.Device())

    global _WNAF_SIZE_LOGGED
    if not use_wnaf_requested and _USE_WNAF_DEFAULT:
        if not _WNAF_SIZE_LOGGED:
            logger.info(
                "[WNAF] 批次 %d 超過門檻 %d，自動改用標準內核",
                cur,
                _WNAF_BATCH_THRESHOLD,
            )
            _WNAF_SIZE_LOGGED = True

    _WNAF_LAST_USED = False

    with stream:
        sk_gpu = cp.random.randint(0, 256, size=(cur, 32), dtype=cp.uint8)

        pub65_gpu: "cp.ndarray"
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
            _WNAF_SIZE_LOGGED = False
            try:
                pub65_gpu = secp_gpu_batch_w4(sk_gpu)
                _WNAF_LAST_USED = True
            except Exception:
                _WNAF_BROKEN = True
                logger.warning("[WNAF] Window4 核心執行異常，改用標準版", exc_info=True)
                pub65_gpu = secp_gpu_batch(sk_gpu) if secp_gpu_batch is not None else gpu_secp256k1_batch(sk_gpu)
                _WNAF_LAST_USED = False
        else:
            if secp_gpu_batch is not None:
                pub65_gpu = secp_gpu_batch(sk_gpu)
            else:
                pub65_gpu = gpu_secp256k1_batch(sk_gpu)
            _WNAF_LAST_USED = False

        xy_gpu = pub65_gpu[:, 1:]
        addr20_gpu = gpu_keccak256_batch(xy_gpu, address_only=True)
        prefix_tron_gpu = cp.full((cur, 1), 0x41, dtype=cp.uint8)
        tron21_gpu = cp.concatenate([prefix_tron_gpu, addr20_gpu], axis=1)

        try:
            ascii_gpu, lens_gpu = _gpu_base58check_raw(tron21_gpu)
            mask: Optional["cp.ndarray"] = None
            if pref_len > 0 and prefix_gpu is not None:
                prefix_mask = lens_gpu >= pref_len
                head = ascii_gpu[:, :pref_len]
                head_cmp = cp.all(head == prefix_gpu[None, :], axis=1)
                mask = cp.logical_and(prefix_mask, head_cmp)
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
        except Exception:
            fallback_cpu = True
            tron21_gpu = tron21_gpu.copy()

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

        if ctx.fallback_cpu or ctx.ascii_gpu is None or ctx.lens_gpu is None:
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
            for i in range(ctx.size):
                addr_hex = tron21_cpu[i].tobytes().hex()
                ln = int(lens_cpu[i])
                addr_b58 = ascii_cpu[i, :ln].tobytes().decode()
                if suffix_bytes is not None and not addr_b58.endswith(suffix_str or ""):
                    continue
                results_local.append((addr_hex, addr_b58))
                priv_local.append(priv_cpu[i].tobytes())

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
) -> bool:
    limit = max_hits if max_hits is not None else None
    for pair, priv in zip(pairs, privs):
        if limit is not None and len(results) >= limit:
            return True
        results.append(pair)
        privkeys_out.append(priv)
    return limit is not None and len(results) >= limit


def _handle_completed_context(
    ctx: _BatchContext,
    prefix_bytes: Optional[bytes],
    prefix_str: Optional[str],
    suffix_bytes: Optional[bytes],
    suffix_str: Optional[str],
    results: List[Tuple[str, str]],
    privkeys_out: List[bytes],
    max_hits: Optional[int],
) -> bool:
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
    return _extend_results(pairs, privs, results, privkeys_out, max_hits)


def _drain_ready_contexts(
    pending: List[_BatchContext],
    results: List[Tuple[str, str]],
    privkeys_out: List[bytes],
    prefix_bytes: Optional[bytes],
    prefix_str: Optional[str],
    suffix_bytes: Optional[bytes],
    suffix_str: Optional[str],
    max_hits: Optional[int],
) -> bool:
    done = False
    while pending:
        future = pending[0].cpu_future
        if future is None or not future.done():
            break
        ctx = pending.pop(0)
        if _handle_completed_context(
            ctx,
            prefix_bytes,
            prefix_str,
            suffix_bytes,
            suffix_str,
            results,
            privkeys_out,
            max_hits,
        ):
            done = True
            break
    return done


def generate_tron_addresses_gpu(
    count: int,
    batch_size: int = 16384,
    prefix: Optional[Union[str, bytes]] = None,
    suffix: Optional[Union[str, bytes]] = None,
    max_hits: Optional[int] = None,
    *,
    dynamic_batches: Optional[Sequence[int]] = None,
    stream_count: Optional[int] = None,
) -> Tuple[List[Tuple[str, str]], List[bytes]]:
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

    results: List[Tuple[str, str]] = []
    privkeys_out: List[bytes] = []

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

    processed = 0
    done = False

    batch_plan = _prepare_batch_plan(batch_size, dynamic_batches)

    env_stream_override = os.environ.get("VANITY_GPU_STREAMS")
    if stream_count is None and env_stream_override:
        try:
            stream_count = int(env_stream_override)
        except ValueError:
            stream_count = None
    if stream_count is None:
        stream_count = _BASE_STREAM_COUNT
        if prefix_bytes is not None:
            target = 6 if pref_len <= 3 else 8
            stream_count = max(stream_count, target)
        else:
            max_candidate = batch_plan[-1] if batch_plan else batch_size
            if max_candidate >= 262144:
                stream_count = max(stream_count, 8)
            elif max_candidate >= 131072:
                stream_count = max(stream_count, 6)
    stream_count = max(2, min(8, int(stream_count)))

    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(stream_count)]
    pending: List[_BatchContext] = []
    stream_idx = 0

    use_wnaf_kernel = _USE_WNAF_DEFAULT
    wnaf_threshold = max(0, _WNAF_BATCH_THRESHOLD)
    if use_wnaf_kernel:
        logger.info("[WNAF] 預設啟用 Window4 核心，等待預熱完成")
    else:
        logger.info("[WNAF] 已停用 Window4 核心，改用標準 secp256k1 內核")
    if use_wnaf_kernel:
        logger.info("[WNAF] 自動切換門檻：批次 > %d 將改用標準內核", _WNAF_BATCH_THRESHOLD)
    async_loop = _ensure_async_loop()
    max_pending = max(stream_count, 1) * _MAX_PENDING_MULTIPLIER_DEFAULT

    while processed < count and not done:
        remain = count - processed
        cur = _select_batch_size(remain, batch_plan, stream_count)
        processed += cur

        stream = streams[stream_idx % stream_count]
        stream_idx += 1
        use_wnaf_this_batch = use_wnaf_kernel and (wnaf_threshold == 0 or cur <= wnaf_threshold)
        ctx = _launch_batch(
            cur,
            stream,
            prefix_gpu,
            pref_len,
            suffix_gpu,
            suf_len,
            use_wnaf_this_batch,
        )
        ctx.cpu_future = asyncio.run_coroutine_threadsafe(
            _process_context_async(ctx, prefix_bytes, prefix_str, suffix_bytes, suffix_str), async_loop
        )
        pending.append(ctx)

        if _drain_ready_contexts(
            pending,
            results,
            privkeys_out,
            prefix_bytes,
            prefix_str,
            suffix_bytes,
            suffix_str,
            max_hits,
        ):
            done = True
            break

        if len(pending) >= max_pending:
            ctx_wait = pending.pop(0)
            if _handle_completed_context(
                ctx_wait,
                prefix_bytes,
                prefix_str,
                suffix_bytes,
                suffix_str,
                results,
                privkeys_out,
                max_hits,
            ):
                done = True
                break

        if max_hits is not None and len(results) >= max_hits:
            done = True
            break

    while pending and not done:
        ctx = pending.pop(0)
        if _handle_completed_context(
            ctx,
            prefix_bytes,
            prefix_str,
            suffix_bytes,
            suffix_str,
            results,
            privkeys_out,
            max_hits,
        ):
            done = True
            break

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
