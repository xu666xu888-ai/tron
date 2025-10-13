# -*- coding: utf-8 -*-
"""
GPU Keccak-256（針對 64 bytes 輸入的批次運算）

用途：
- TRON/Ethereum 風格地址導出時，對未壓縮公鑰去掉 0x04 的 X||Y（64 bytes）做 Keccak-256。
- rate=136 bytes（Keccak-256），本實作使用單區塊吸收：
  將輸入 64 bytes XOR 進前 8 個 64-bit lane；第 9 個 lane XOR 0x01 做 pad10*1，
  第 17 個 lane XOR 0x80<<56 作為尾端 1 bit（整體 136 bytes padding）。

注意：
- 內部以 little-endian lane（uint64）處理；輸出 32 bytes 為前 4 個 lane 之小端串接。
- 若無法初始化 GPU kernel，可於呼叫端退化為 CPU keccak（addr.py 的 helpers）。
"""
from __future__ import annotations

try:
    import cupy as cp
except Exception as e:  # pragma: no cover
    raise ImportError("需要 CuPy 以使用 GPU Keccak：pip install cupy-cuda11x/12x") from e


_KECCAK256_KERNEL = r"""
// 參考 FIPS 202 的 Keccak-f[1600]。以 64-bit lane 形式實作，固定吸收 64 bytes。

__device__ __forceinline__ unsigned long long rotl64(const unsigned long long x, const unsigned int y){
    return (x << y) | (x >> (64 - y));
}

extern "C" __global__ void keccak256_64(
    const unsigned char* __restrict__ in64, // (N,64)
    unsigned char* __restrict__ out32,      // (N,32)
    const int stride_in,                    // 64
    const int stride_out,                   // 32
    const int n
){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    const unsigned char* src = in64 + (size_t)i * (size_t)stride_in;

    unsigned long long st[25];
    #pragma unroll
    for (int t = 0; t < 25; ++t) st[t] = 0ULL;

    #pragma unroll
    for (int j = 0; j < 8; ++j){
        unsigned long long v =
            ((unsigned long long)src[j*8 + 0])       |
            ((unsigned long long)src[j*8 + 1] << 8)  |
            ((unsigned long long)src[j*8 + 2] << 16) |
            ((unsigned long long)src[j*8 + 3] << 24) |
            ((unsigned long long)src[j*8 + 4] << 32) |
            ((unsigned long long)src[j*8 + 5] << 40) |
            ((unsigned long long)src[j*8 + 6] << 48) |
            ((unsigned long long)src[j*8 + 7] << 56);
        st[j] ^= v;
    }

    unsigned char* lanes = reinterpret_cast<unsigned char*>(st);
    lanes[64]  ^= 0x01U;   // pad10*1 起始 1
    lanes[135] ^= 0x80U;   // 最後一個 bit 1（rate=136 bytes）

    const unsigned int KECCAKF_ROTC[24] = {
        1, 3, 6, 10, 15, 21, 28, 36, 45, 55,
        2, 14, 27, 41, 56, 8, 25, 43, 62, 18,
        39, 61, 20, 44
    };
    const unsigned int KECCAKF_PILN[24] = {
        10, 7, 11, 17, 18, 3, 5, 16,
        8, 21, 24, 4, 15, 23, 19, 13,
        12, 2, 20, 14, 22, 9, 6, 1
    };
    const unsigned long long KECCAKF_RNDC[24] = {
        0x0000000000000001ULL, 0x0000000000008082ULL,
        0x800000000000808AULL, 0x8000000080008000ULL,
        0x000000000000808BULL, 0x0000000080000001ULL,
        0x8000000080008081ULL, 0x8000000000008009ULL,
        0x000000000000008AULL, 0x0000000000000088ULL,
        0x0000000080008009ULL, 0x000000008000000AULL,
        0x000000008000808BULL, 0x800000000000008BULL,
        0x8000000000008089ULL, 0x8000000000008003ULL,
        0x8000000000008002ULL, 0x8000000000000080ULL,
        0x000000000000800AULL, 0x800000008000000AULL,
        0x8000000080008081ULL, 0x8000000000008080ULL,
        0x0000000080000001ULL, 0x8000000080008008ULL
    };

    for (int round = 0; round < 24; ++round){
        unsigned long long bc[5];
        for (int x = 0; x < 5; ++x){
            bc[x] = st[x] ^ st[x + 5] ^ st[x + 10] ^ st[x + 15] ^ st[x + 20];
        }

        for (int x = 0; x < 5; ++x){
            unsigned long long t = bc[(x + 4) % 5] ^ rotl64(bc[(x + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5){
                st[j + x] ^= t;
            }
        }

        unsigned long long t = st[1];
        for (int x = 0; x < 24; ++x){
            int j = KECCAKF_PILN[x];
            unsigned long long current = st[j];
            st[j] = rotl64(t, KECCAKF_ROTC[x]);
            t = current;
        }

        for (int j = 0; j < 25; j += 5){
            unsigned long long row[5];
            for (int x = 0; x < 5; ++x) row[x] = st[j + x];
            for (int x = 0; x < 5; ++x){
                st[j + x] = row[x] ^ ((~row[(x + 1) % 5]) & row[(x + 2) % 5]);
            }
        }

        st[0] ^= KECCAKF_RNDC[round];
    }

    unsigned char* dst = out32 + (size_t)i * (size_t)stride_out;
    #pragma unroll
    for (int k = 0; k < 4; ++k){
        unsigned long long v = st[k];
        dst[k*8 + 0] = (unsigned char)(v & 0xFFULL);
        dst[k*8 + 1] = (unsigned char)((v >> 8) & 0xFFULL);
        dst[k*8 + 2] = (unsigned char)((v >> 16) & 0xFFULL);
        dst[k*8 + 3] = (unsigned char)((v >> 24) & 0xFFULL);
        dst[k*8 + 4] = (unsigned char)((v >> 32) & 0xFFULL);
        dst[k*8 + 5] = (unsigned char)((v >> 40) & 0xFFULL);
        dst[k*8 + 6] = (unsigned char)((v >> 48) & 0xFFULL);
        dst[k*8 + 7] = (unsigned char)((v >> 56) & 0xFFULL);
    }
}
"""

_mod = cp.RawModule(code=_KECCAK256_KERNEL, options=("-std=c++11",))
_ker = _mod.get_function("keccak256_64")


def keccak256_xy_batch(pubkey_xy_gpu: "cp.ndarray") -> "cp.ndarray":
    """GPU 計算多筆 Keccak-256（輸入固定 64 bytes）。"""
    if pubkey_xy_gpu.dtype != cp.uint8 or pubkey_xy_gpu.ndim != 2 or pubkey_xy_gpu.shape[1] != 64:
        raise ValueError("pubkey_xy_gpu 需為 uint8 (N,64)")
    try:
        xy_gpu = cp.ascontiguousarray(pubkey_xy_gpu)
        n = xy_gpu.shape[0]
        out = cp.empty((n, 32), dtype=cp.uint8)
        threads = 256
        blocks = (n + threads - 1) // threads
        _ker((blocks,), (threads,), (xy_gpu, out, cp.int32(64), cp.int32(32), cp.int32(n)))
        return out
    except Exception:
        import numpy as np, sha3
        xy_cpu: np.ndarray = cp.asnumpy(pubkey_xy_gpu)
        out = np.empty((xy_cpu.shape[0], 32), dtype=np.uint8)
        for i in range(xy_cpu.shape[0]):
            k = sha3.keccak_256(); k.update(bytes(xy_cpu[i])); out[i,:] = np.frombuffer(k.digest(), dtype=np.uint8)
        return cp.asarray(out)


__all__ = ["keccak256_xy_batch"]
