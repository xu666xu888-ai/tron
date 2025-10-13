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
// 參考 Keccak-f[1600]，以 64-bit lane 實作（little-endian）

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

    unsigned long long A[25];
    #pragma unroll
    for (int t=0;t<25;++t) A[t]=0ULL;

    #pragma unroll
    for (int j=0;j<8;++j){
        unsigned long long v =
            ((unsigned long long)src[j*8+0])       |
            ((unsigned long long)src[j*8+1] << 8)  |
            ((unsigned long long)src[j*8+2] << 16) |
            ((unsigned long long)src[j*8+3] << 24) |
            ((unsigned long long)src[j*8+4] << 32) |
            ((unsigned long long)src[j*8+5] << 40) |
            ((unsigned long long)src[j*8+6] << 48) |
            ((unsigned long long)src[j*8+7] << 56);
        A[j] ^= v;
    }

    A[8]  ^= 0x01ULL;
    A[16] ^= (0x80ULL << 56);

    const int R[25] = {
        0, 1, 62, 28, 27,
        36, 44, 6, 55, 20,
        3, 10, 43, 25, 39,
        41, 45, 15, 21, 8,
        18, 2, 61, 56, 14
    };

    const unsigned long long RC[24] = {
        0x0000000000000001ULL, 0x0000000000008082ULL,
        0x800000000000808aULL, 0x8000000080008000ULL,
        0x000000000000808bULL, 0x0000000080000001ULL,
        0x8000000080008081ULL, 0x8000000000008009ULL,
        0x000000000000008aULL, 0x0000000000000088ULL,
        0x0000000080008009ULL, 0x000000008000000aULL,
        0x000000008000808bULL, 0x800000000000008bULL,
        0x8000000000008089ULL, 0x8000000000008003ULL,
        0x8000000000008002ULL, 0x8000000000000080ULL,
        0x000000000000800aULL, 0x800000008000000aULL,
        0x8000000080008081ULL, 0x8000000000008080ULL,
        0x0000000080000001ULL, 0x8000000080008008ULL
    };

    auto ROTL64 = [](unsigned long long x, int s) -> unsigned long long {
        return (x << s) | (x >> (64 - s));
    };

    for (int round = 0; round < 24; ++round){
        unsigned long long C[5], D[5];
        for (int x=0;x<5;++x){
            C[x] = A[x] ^ A[x+5] ^ A[x+10] ^ A[x+15] ^ A[x+20];
        }
        for (int x=0;x<5;++x){
            D[x] = C[(x+4)%5] ^ ROTL64(C[(x+1)%5], 1);
        }
        for (int y=0;y<5;++y){
            for (int x=0;x<5;++x){
                A[x + 5*y] ^= D[x];
            }
        }

        unsigned long long B[25];
        for (int y=0;y<5;++y){
            for (int x=0;x<5;++x){
                int idx = x + 5*y;
                int newX = y;
                int newY = (2*x + 3*y) % 5;
                B[newX + 5*newY] = ROTL64(A[idx], R[idx]);
            }
        }

        for (int y=0;y<5;++y){
            for (int x=0;x<5;++x){
                A[x + 5*y] = B[x + 5*y] ^ ((~B[(x+1)%5 + 5*y]) & B[(x+2)%5 + 5*y]);
            }
        }

        A[0] ^= RC[round];
    }

    unsigned char* dst = out32 + (size_t)i * (size_t)stride_out;
    #pragma unroll
    for (int k=0;k<4;++k){
        unsigned long long v = A[k];
        dst[k*8+0] = (unsigned char)(v & 0xFFULL);
        dst[k*8+1] = (unsigned char)((v >> 8) & 0xFFULL);
        dst[k*8+2] = (unsigned char)((v >> 16) & 0xFFULL);
        dst[k*8+3] = (unsigned char)((v >> 24) & 0xFFULL);
        dst[k*8+4] = (unsigned char)((v >> 32) & 0xFFULL);
        dst[k*8+5] = (unsigned char)((v >> 40) & 0xFFULL);
        dst[k*8+6] = (unsigned char)((v >> 48) & 0xFFULL);
        dst[k*8+7] = (unsigned char)((v >> 56) & 0xFFULL);
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
