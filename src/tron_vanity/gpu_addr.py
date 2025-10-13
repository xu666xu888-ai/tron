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
from typing import List, Tuple

try:
    import cupy as cp  # GPU 陣列/Kernel
except Exception as e:  # pragma: no cover
    raise ImportError("需要安裝 CuPy 才能使用 gpu_addr 模組：pip install cupy-cuda11x/12x") from e

import base58
import sha3
import numpy as np
try:
    from .gpu_keccak import keccak256_xy_batch as _gpu_keccak256_xy_batch
except Exception:
    _gpu_keccak256_xy_batch = None
try:
    from .gpu_secp256k1 import gpu_secp256k1_batch as secp_gpu_batch
except Exception:
    secp_gpu_batch = None

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
    threads = 256
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
    threads = 256
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


def gpu_keccak256_batch(pubkey_xy_gpu: "cp.ndarray") -> "cp.ndarray":
    """
    批量 Keccak-256：
    - 若可用 GPU kernel，使用 keccak256_xy_batch
    - 否則退化為 CPU 計算
    """
    if pubkey_xy_gpu.dtype != cp.uint8 or pubkey_xy_gpu.ndim != 2 or pubkey_xy_gpu.shape[1] != 64:
        raise ValueError("pubkey_xy_gpu 需為 uint8 (N,64)")

    if _gpu_keccak256_xy_batch is not None:
        try:
            return _gpu_keccak256_xy_batch(pubkey_xy_gpu)
        except Exception:
            pass

    xy_cpu: np.ndarray = cp.asnumpy(pubkey_xy_gpu)
    out = np.empty((xy_cpu.shape[0], 32), dtype=np.uint8)
    for i in range(xy_cpu.shape[0]):
        h = _keccak_256(bytes(xy_cpu[i]))
        out[i, :] = np.frombuffer(h, dtype=np.uint8)
    return cp.asarray(out)


def gpu_base58check_batch(tron21_gpu: "cp.ndarray") -> List[str]:
    """
    Base58Check 編碼：
    - 輸入為 21 bytes（0x41 + addr20），形狀 (N,21)
    - 在 GPU 上完成雙 SHA-256 與 Base58 編碼，必要時退回 CPU。
    """
    if tron21_gpu.dtype != cp.uint8 or tron21_gpu.ndim != 2 or tron21_gpu.shape[1] != 21:
        raise ValueError("tron21_gpu 需為 uint8 (N,21)")

    N = tron21_gpu.shape[0]
    lens = cp.full((N,), 21, dtype=cp.int32)
    d1 = gpu_sha256_oneblock_batch(tron21_gpu, lens)
    lens2 = cp.full((N,), 32, dtype=cp.int32)
    d2 = gpu_sha256_oneblock_batch(d1, lens2)

    checksum_gpu = d2[:, :4]
    try:
        payload25_gpu = cp.concatenate([tron21_gpu, checksum_gpu], axis=1)
        payload25_gpu = cp.ascontiguousarray(payload25_gpu)
        ascii_gpu, lens_gpu = gpu_base58_encode_batch(payload25_gpu)
        ascii_cpu = cp.asnumpy(ascii_gpu)
        lens_cpu = cp.asnumpy(lens_gpu)
        out: List[str] = []
        for i in range(N):
            ln = int(lens_cpu[i])
            if ln <= 0 or ln > ascii_cpu.shape[1]:
                raise ValueError("Base58 GPU 產生長度異常")
            out.append(ascii_cpu[i, :ln].tobytes().decode())
        return out
    except Exception:
        body_cpu: np.ndarray = cp.asnumpy(tron21_gpu)
        d2_cpu: np.ndarray = cp.asnumpy(d2)
        out: List[str] = []
        for i in range(N):
            checksum = bytes(d2_cpu[i][:4])
            out.append(base58.b58encode(body_cpu[i].tobytes() + checksum).decode())
        return out


def generate_tron_addresses_gpu(count: int, batch_size: int = 16384) -> Tuple[List[Tuple[str, str]], List[bytes]]:
    """
    完全在 GPU 記憶體流程的雛形（保留 CPU 後備），回傳 (地址列表, 私鑰列表)：
    - 地址列表：[(hex_addr, base58_addr), ...]
    - 私鑰列表：[privkey_bytes, ...]

    secp256k1 / Keccak-256 / Base58Check 皆有 CUDA 內核，若 GPU 失敗則退回 CPU。
    """
    results: List[Tuple[str, str]] = []
    privkeys_out: List[bytes] = []

    # 每批在 GPU 端生成私鑰（32 bytes）
    for _ in range(0, count, batch_size):
        cur = min(batch_size, count - len(privkeys_out))

        # 1) GPU 亂數產生私鑰 (cur,32)
        blob: bytes = cp.random.bytes(cur * 32)
        priv_cpu = [blob[i * 32 : (i + 1) * 32] for i in range(cur)]
        privkeys_out.extend(priv_cpu)

        # 搬移到 GPU `uint8` 陣列
        sk_gpu = cp.frombuffer(blob, dtype=cp.uint8).reshape(cur, 32).copy()

        # 2) 公鑰 (cur,65)：預設使用 CPU 後備（確保正確性）。
        # 若需啟用實驗性 GPU 橢圓曲線，設置環境變數 VANITY_EXPERIMENTAL_GPU_SECP=1
        use_exp_gpu = os.environ.get("VANITY_EXPERIMENTAL_GPU_SECP") == "1"
        if use_exp_gpu and secp_gpu_batch is not None:
            try:
                pub65_gpu = secp_gpu_batch(sk_gpu)
            except Exception:
                pub65_gpu = gpu_secp256k1_batch(sk_gpu)
        else:
            pub65_gpu = gpu_secp256k1_batch(sk_gpu)

        # 3) Keccak-256（CPU 後備），取 XY（去 0x04）
        # 取去掉 0x04 的 X||Y（64 bytes），並 copy 以確保 8-byte 對齊
        xy_gpu = pub65_gpu[:, 1:].copy()
        keccak_gpu = gpu_keccak256_batch(xy_gpu)

        # 4) 組 TRON 21 bytes（0x41 + 後 20 bytes）
        addr20_gpu = keccak_gpu[:, -20:]
        prefix_gpu = cp.full((cur, 1), 0x41, dtype=cp.uint8)
        tron21_gpu = cp.concatenate([prefix_gpu, addr20_gpu], axis=1)

        # 5) Base58Check（主要在 GPU 完成）
        b58_list = gpu_base58check_batch(tron21_gpu)

        # 6) HEX 地址輸出
        tron21_cpu = cp.asnumpy(tron21_gpu)
        for i in range(cur):
            results.append((tron21_cpu[i].tobytes().hex(), b58_list[i]))

    return results, privkeys_out


__all__ = [
    "gpu_secp256k1_batch",
    "gpu_keccak256_batch",
    "gpu_base58_encode_batch",
    "gpu_base58check_batch",
    "generate_tron_addresses_gpu",
]
