# -*- coding: utf-8 -*-
"""
GPU Incremental secp256k1 Scanner with Fused Keccak + Vanity Match (v4)

Architecture:
- Incremental point addition: P_{i+1} = P_i + stride_G
- Batch inversion (Montgomery's trick, K=128)
- Pseudo-Mersenne direct reduction (schoolbook + reduce_p)
- PTX carry chains for add256/sub256
- Fused Keccak-256: compute address hash inline after affine conversion
- GPU-side vanity matching: only output matching keys (atomic counter)

Pipeline: secp256k1 → affine → Keccak-256 → suffix match → hit buffer
Eliminates ~500MB/batch GPU→CPU transfer of all pubkeys.

Development rule: All changes in experimental first, then copy to stable.
"""
from __future__ import annotations

import os
import logging
import hashlib
import numpy as np

try:
    import cupy as cp
except ImportError as e:
    raise ImportError("CuPy required: pip install cupy-cuda11x/12x") from e

logger = logging.getLogger(__name__)

# --- secp256k1 parameters ---
SECP256K1_P = "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F"
SECP256K1_GX = "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798"
SECP256K1_GY = "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8"

_P_INT = int(SECP256K1_P, 16)
_GX_INT = int(SECP256K1_GX, 16)
_GY_INT = int(SECP256K1_GY, 16)


def _int_to_limbs(v: int) -> list:
    """Convert a 256-bit integer to 8 x 32-bit limbs (little-endian)."""
    limbs = []
    for _ in range(8):
        limbs.append(v & 0xFFFFFFFF)
        v >>= 32
    return limbs


def _point_add_affine(x1: int, y1: int, x2: int, y2: int) -> tuple:
    if x1 == 0 and y1 == 0:
        return x2, y2
    if x2 == 0 and y2 == 0:
        return x1, y1
    if x1 == x2:
        if y1 == y2:
            lam = (3 * x1 * x1 * pow(2 * y1, _P_INT - 2, _P_INT)) % _P_INT
        else:
            return 0, 0
    else:
        lam = ((y2 - y1) * pow(x2 - x1, _P_INT - 2, _P_INT)) % _P_INT
    x3 = (lam * lam - x1 - x2) % _P_INT
    y3 = (lam * (x1 - x3) - y1) % _P_INT
    return x3, y3


def _point_double_affine(x: int, y: int) -> tuple:
    if x == 0 and y == 0:
        return 0, 0
    lam = (3 * x * x * pow(2 * y, _P_INT - 2, _P_INT)) % _P_INT
    x3 = (lam * lam - 2 * x) % _P_INT
    y3 = (lam * (x - x3) - y) % _P_INT
    return x3, y3


def precompute_binary_g_table(max_bits: int = 20) -> np.ndarray:
    table = np.zeros((max_bits, 2, 8), dtype=np.uint32)
    gx, gy = _GX_INT, _GY_INT
    for j in range(max_bits):
        x_limbs = _int_to_limbs(gx)
        y_limbs = _int_to_limbs(gy)
        for k in range(8):
            table[j, 0, k] = x_limbs[k]
            table[j, 1, k] = y_limbs[k]
        gx, gy = _point_double_affine(gx, gy)
    return table


def precompute_stride_g(stride: int) -> tuple:
    gx, gy = _GX_INT, _GY_INT
    rx, ry = 0, 0
    for bit in range(stride.bit_length()):
        if stride & (1 << bit):
            rx, ry = _point_add_affine(rx, ry, gx, gy)
        gx, gy = _point_double_affine(gx, gy)
    x_limbs = np.array(_int_to_limbs(rx), dtype=np.uint32)
    y_limbs = np.array(_int_to_limbs(ry), dtype=np.uint32)
    return x_limbs, y_limbs


BATCH_K = 128
MAX_HITS = 4096  # max matching keys per kernel launch

# Base58 alphabet for TRON addresses
_B58_ALPHABET = b'123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'


def _address_to_suffix_bytes(suffix: str) -> bytes:
    """Convert a TRON address suffix to the raw bytes we need to match.

    TRON address = Base58Check(0x41 || keccak[-20:] || sha256d[:4])
    For suffix matching, we work at the hex level of the 20-byte raw address.
    But Base58Check makes this non-trivial, so we match at the Base58 string level.

    For GPU, we'll match the raw address bytes (the 20-byte keccak suffix)
    against precomputed target patterns. The simplest approach is to match
    the hex digits of the address.

    Returns the suffix as ASCII bytes for matching.
    """
    return suffix.encode('ascii')


# The CUDA kernel with fused Keccak + vanity matching
_INCREMENTAL_CUDA_SOURCE = r"""
__constant__ unsigned int SECP256K1_P[8] = {
    0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

__constant__ unsigned int GTABLE[20][16];
__constant__ unsigned int STRIDE_GX[8];
__constant__ unsigned int STRIDE_GY[8];

// ====================== Field Arithmetic (PTX Carry Chains) =======================

__device__ __forceinline__ int cmp256(const unsigned int* a, const unsigned int* b) {
    for (int i = 7; i >= 0; --i) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}

__device__ __forceinline__ unsigned int add256_ptx(unsigned int* temp, const unsigned int* a, const unsigned int* b) {
    unsigned int carry;
    asm volatile(
        "add.cc.u32   %0,  %9,  %17;\n\t"
        "addc.cc.u32  %1,  %10, %18;\n\t"
        "addc.cc.u32  %2,  %11, %19;\n\t"
        "addc.cc.u32  %3,  %12, %20;\n\t"
        "addc.cc.u32  %4,  %13, %21;\n\t"
        "addc.cc.u32  %5,  %14, %22;\n\t"
        "addc.cc.u32  %6,  %15, %23;\n\t"
        "addc.cc.u32  %7,  %16, %24;\n\t"
        "addc.u32     %8,  0,   0;\n\t"
        : "=r"(temp[0]), "=r"(temp[1]), "=r"(temp[2]), "=r"(temp[3]),
          "=r"(temp[4]), "=r"(temp[5]), "=r"(temp[6]), "=r"(temp[7]),
          "=r"(carry)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7])
    );
    return carry;
}

__device__ __forceinline__ unsigned int sub256_ptx(unsigned int* temp, const unsigned int* a, const unsigned int* b) {
    unsigned int borrow;
    asm volatile(
        "sub.cc.u32   %0,  %9,  %17;\n\t"
        "subc.cc.u32  %1,  %10, %18;\n\t"
        "subc.cc.u32  %2,  %11, %19;\n\t"
        "subc.cc.u32  %3,  %12, %20;\n\t"
        "subc.cc.u32  %4,  %13, %21;\n\t"
        "subc.cc.u32  %5,  %14, %22;\n\t"
        "subc.cc.u32  %6,  %15, %23;\n\t"
        "subc.cc.u32  %7,  %16, %24;\n\t"
        "subc.u32     %8,  0,   0;\n\t"
        : "=r"(temp[0]), "=r"(temp[1]), "=r"(temp[2]), "=r"(temp[3]),
          "=r"(temp[4]), "=r"(temp[5]), "=r"(temp[6]), "=r"(temp[7]),
          "=r"(borrow)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7])
    );
    return borrow;
}

__device__ void add256_mod(unsigned int* r, const unsigned int* a, const unsigned int* b) {
    unsigned int temp[8];
    unsigned int carry = add256_ptx(temp, a, b);
    if (carry || cmp256(temp, SECP256K1_P) >= 0) {
        sub256_ptx(r, temp, SECP256K1_P);
    } else {
        #pragma unroll
        for (int i = 0; i < 8; ++i) r[i] = temp[i];
    }
}

__device__ void sub256_mod(unsigned int* r, const unsigned int* a, const unsigned int* b) {
    unsigned int temp[8];
    unsigned int borrow = sub256_ptx(temp, a, b);
    if (borrow) {
        add256_ptx(r, temp, SECP256K1_P);
    } else {
        #pragma unroll
        for (int i = 0; i < 8; ++i) r[i] = temp[i];
    }
}

__device__ void mul256_full(unsigned int* T, const unsigned int* a, const unsigned int* b) {
    #pragma unroll
    for (int i = 0; i < 16; ++i) T[i] = 0U;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        unsigned long long carry = 0ULL;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            unsigned long long acc = (unsigned long long)T[i+j] + (unsigned long long)a[i] * (unsigned long long)b[j] + carry;
            T[i+j] = (unsigned int)acc;
            carry = acc >> 32;
        }
        T[i+8] += (unsigned int)carry;
    }
}

__device__ void reduce_p(unsigned int* r, const unsigned int* T) {
    unsigned int X[20];
    #pragma unroll
    for (int i = 0; i < 20; ++i) X[i] = 0U;
    #pragma unroll
    for (int i = 0; i < 16; ++i) X[i] = T[i];

    for (int k = 15; k >= 8; --k) {
        unsigned int u = X[k];
        if (!u) continue;
        X[k] = 0U;
        unsigned long long acc = (unsigned long long)X[k-8] + (unsigned long long)u * 977ULL;
        X[k-8] = (unsigned int)acc;
        unsigned long long carry = acc >> 32;
        int idx = k - 7;
        while (carry) {
            unsigned long long a2 = (unsigned long long)X[idx] + carry;
            X[idx] = (unsigned int)a2;
            carry = a2 >> 32;
            ++idx;
        }
        acc = (unsigned long long)X[k-7] + (unsigned long long)u;
        X[k-7] = (unsigned int)acc;
        carry = acc >> 32;
        idx = k - 6;
        while (carry) {
            unsigned long long a3 = (unsigned long long)X[idx] + carry;
            X[idx] = (unsigned int)a3;
            carry = a3 >> 32;
            ++idx;
        }
    }
    int changed = 1;
    while (changed) {
        changed = 0;
        for (int k = 19; k >= 8; --k) {
            unsigned int u = X[k];
            if (!u) continue;
            changed = 1;
            X[k] = 0U;
            unsigned long long acc = (unsigned long long)X[k-8] + (unsigned long long)u * 977ULL;
            X[k-8] = (unsigned int)acc;
            unsigned long long carry = acc >> 32;
            int idx = k - 7;
            while (carry) {
                unsigned long long a2 = (unsigned long long)X[idx] + carry;
                X[idx] = (unsigned int)a2;
                carry = a2 >> 32;
                ++idx;
            }
            acc = (unsigned long long)X[k-7] + (unsigned long long)u;
            X[k-7] = (unsigned int)acc;
            carry = acc >> 32;
            idx = k - 6;
            while (carry) {
                unsigned long long a3 = (unsigned long long)X[idx] + carry;
                X[idx] = (unsigned int)a3;
                carry = a3 >> 32;
                ++idx;
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) r[i] = X[i];
    while (cmp256(r, SECP256K1_P) >= 0) {
        sub256_ptx(r, r, SECP256K1_P);
    }
}

__device__ void mul256_mod(unsigned int* r, const unsigned int* a, const unsigned int* b) {
    unsigned int T[16];
    mul256_full(T, a, b);
    reduce_p(r, T);
}

__device__ void sqr256_mod(unsigned int* r, const unsigned int* a) {
    mul256_mod(r, a, a);
}

__device__ void inv256_mod(unsigned int* r, const unsigned int* a) {
    unsigned int exp[8] = {
        0xFFFFFC2D, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
    };
    unsigned int result[8], base[8];
    result[0] = 1;
    #pragma unroll
    for (int i = 1; i < 8; ++i) result[i] = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) base[i] = a[i];
    for (int i = 0; i < 256; ++i) {
        int word = i >> 5;
        int bit = i & 31;
        if (exp[word] & (1U << bit)) {
            unsigned int tmp[8];
            mul256_mod(tmp, result, base);
            #pragma unroll
            for (int j = 0; j < 8; ++j) result[j] = tmp[j];
        }
        if (i < 255) {
            unsigned int tmp[8];
            sqr256_mod(tmp, base);
            #pragma unroll
            for (int j = 0; j < 8; ++j) base[j] = tmp[j];
        }
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) r[i] = result[i];
}

__device__ void point_add_mixed(
    unsigned int* Px, unsigned int* Py, unsigned int* Pz,
    const unsigned int* qx, const unsigned int* qy
) {
    bool p_inf = true;
    for (int i = 0; i < 8; ++i) if (Pz[i] != 0) { p_inf = false; break; }
    if (p_inf) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) { Px[i] = qx[i]; Py[i] = qy[i]; Pz[i] = 0; }
        Pz[0] = 1;
        return;
    }
    unsigned int h[8], rv[8], t[8];
    mul256_mod(t, Pz, Pz);
    mul256_mod(h, qx, t);
    mul256_mod(t, t, Pz);
    mul256_mod(rv, qy, t);
    sub256_mod(h, h, Px);
    sub256_mod(rv, rv, Py);
    unsigned int nz[8];
    mul256_mod(nz, Pz, h);
    mul256_mod(t, h, h);
    unsigned int h3[8];
    mul256_mod(h3, t, h);
    mul256_mod(h, Px, t);
    unsigned int nx[8];
    mul256_mod(nx, rv, rv);
    sub256_mod(nx, nx, h3);
    sub256_mod(nx, nx, h);
    sub256_mod(nx, nx, h);
    unsigned int ny[8];
    mul256_mod(t, Py, h3);
    sub256_mod(h, h, nx);
    mul256_mod(ny, rv, h);
    sub256_mod(ny, ny, t);
    #pragma unroll
    for (int i = 0; i < 8; ++i) { Px[i] = nx[i]; Py[i] = ny[i]; Pz[i] = nz[i]; }
}

// ====================== Inline Keccak-256 for 64-byte input ======================

__device__ __forceinline__ unsigned long long rotl64(unsigned long long x, unsigned int y) {
    return (x << y) | (x >> (64 - y));
}

__device__ void keccak256_64bytes(const unsigned char* input64, unsigned char* digest32) {
    unsigned long long st[25];
    #pragma unroll
    for (int i = 0; i < 25; ++i) st[i] = 0ULL;

    // Absorb 64 bytes (8 lanes)
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        unsigned long long v =
            ((unsigned long long)input64[j*8 + 0])       |
            ((unsigned long long)input64[j*8 + 1] << 8)  |
            ((unsigned long long)input64[j*8 + 2] << 16) |
            ((unsigned long long)input64[j*8 + 3] << 24) |
            ((unsigned long long)input64[j*8 + 4] << 32) |
            ((unsigned long long)input64[j*8 + 5] << 40) |
            ((unsigned long long)input64[j*8 + 6] << 48) |
            ((unsigned long long)input64[j*8 + 7] << 56);
        st[j] ^= v;
    }

    // Padding: pad10*1 for rate=136
    unsigned char* lanes = reinterpret_cast<unsigned char*>(st);
    lanes[64]  ^= 0x01U;
    lanes[135] ^= 0x80U;

    // Keccak-f[1600] - 24 rounds
    const unsigned long long RNDC[24] = {
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
    const unsigned int ROTC[24] = {
        1, 3, 6, 10, 15, 21, 28, 36, 45, 55,
        2, 14, 27, 41, 56, 8, 25, 43, 62, 18,
        39, 61, 20, 44
    };
    const unsigned int PILN[24] = {
        10, 7, 11, 17, 18, 3, 5, 16,
        8, 21, 24, 4, 15, 23, 19, 13,
        12, 2, 20, 14, 22, 9, 6, 1
    };

    for (int round = 0; round < 24; ++round) {
        unsigned long long bc[5];
        for (int x = 0; x < 5; ++x)
            bc[x] = st[x] ^ st[x+5] ^ st[x+10] ^ st[x+15] ^ st[x+20];
        for (int x = 0; x < 5; ++x) {
            unsigned long long t = bc[(x+4)%5] ^ rotl64(bc[(x+1)%5], 1);
            for (int j = 0; j < 25; j += 5) st[j+x] ^= t;
        }
        unsigned long long t = st[1];
        for (int x = 0; x < 24; ++x) {
            int j = PILN[x];
            unsigned long long cur = st[j];
            st[j] = rotl64(t, ROTC[x]);
            t = cur;
        }
        for (int j = 0; j < 25; j += 5) {
            unsigned long long row[5];
            for (int x = 0; x < 5; ++x) row[x] = st[j+x];
            for (int x = 0; x < 5; ++x)
                st[j+x] = row[x] ^ ((~row[(x+1)%5]) & row[(x+2)%5]);
        }
        st[0] ^= RNDC[round];
    }

    // Extract first 32 bytes (4 lanes, little-endian)
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        unsigned long long v = st[k];
        digest32[k*8+0] = (unsigned char)(v & 0xFF);
        digest32[k*8+1] = (unsigned char)((v >> 8) & 0xFF);
        digest32[k*8+2] = (unsigned char)((v >> 16) & 0xFF);
        digest32[k*8+3] = (unsigned char)((v >> 24) & 0xFF);
        digest32[k*8+4] = (unsigned char)((v >> 32) & 0xFF);
        digest32[k*8+5] = (unsigned char)((v >> 40) & 0xFF);
        digest32[k*8+6] = (unsigned char)((v >> 48) & 0xFF);
        digest32[k*8+7] = (unsigned char)((v >> 56) & 0xFF);
    }
}

// ====================== SHA-256 for Base58Check checksum ======================

__constant__ unsigned int SHA256_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

__device__ __forceinline__ unsigned int rotr32(unsigned int x, unsigned int n) {
    return (x >> n) | (x << (32 - n));
}

// SHA-256 for short messages (up to 55 bytes, single block)
__device__ void sha256_short(const unsigned char* msg, int msg_len, unsigned char* digest) {
    // Prepare message block (single 64-byte block with padding)
    unsigned int W[64];
    unsigned char block[64];
    #pragma unroll
    for (int i = 0; i < 64; ++i) block[i] = 0;
    for (int i = 0; i < msg_len; ++i) block[i] = msg[i];
    block[msg_len] = 0x80;
    // Length in bits (big-endian) at end
    unsigned int bit_len = (unsigned int)msg_len * 8;
    block[60] = (bit_len >> 24) & 0xFF;
    block[61] = (bit_len >> 16) & 0xFF;
    block[62] = (bit_len >> 8) & 0xFF;
    block[63] = bit_len & 0xFF;

    // Parse block into W[0..15] big-endian
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        W[i] = ((unsigned int)block[i*4] << 24) |
               ((unsigned int)block[i*4+1] << 16) |
               ((unsigned int)block[i*4+2] << 8) |
               ((unsigned int)block[i*4+3]);
    }
    // Expand W[16..63]
    for (int i = 16; i < 64; ++i) {
        unsigned int s0 = rotr32(W[i-15], 7) ^ rotr32(W[i-15], 18) ^ (W[i-15] >> 3);
        unsigned int s1 = rotr32(W[i-2], 17) ^ rotr32(W[i-2], 19) ^ (W[i-2] >> 10);
        W[i] = W[i-16] + s0 + W[i-7] + s1;
    }

    // Initialize hash values
    unsigned int h0 = 0x6a09e667, h1 = 0xbb67ae85, h2 = 0x3c6ef372, h3 = 0xa54ff53a;
    unsigned int h4 = 0x510e527f, h5 = 0x9b05688c, h6 = 0x1f83d9ab, h7 = 0x5be0cd19;
    unsigned int a=h0, b=h1, c=h2, d=h3, e=h4, f=h5, g=h6, h=h7;

    // 64 rounds
    for (int i = 0; i < 64; ++i) {
        unsigned int S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        unsigned int ch = (e & f) ^ (~e & g);
        unsigned int temp1 = h + S1 + ch + SHA256_K[i] + W[i];
        unsigned int S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        unsigned int maj = (a & b) ^ (a & c) ^ (b & c);
        unsigned int temp2 = S0 + maj;
        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }

    h0 += a; h1 += b; h2 += c; h3 += d;
    h4 += e; h5 += f; h6 += g; h7 += h;

    // Output digest big-endian
    unsigned int hv[8] = {h0,h1,h2,h3,h4,h5,h6,h7};
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        digest[i*4+0] = (hv[i] >> 24) & 0xFF;
        digest[i*4+1] = (hv[i] >> 16) & 0xFF;
        digest[i*4+2] = (hv[i] >> 8) & 0xFF;
        digest[i*4+3] = hv[i] & 0xFF;
    }
}

// ====================== Base58 Suffix Matching ======================
// TRON address = Base58Check(0x41 || keccak[-20:] || sha256d[:4])
// Total 25 bytes encoded in Base58.
// For suffix matching: compute big_int mod 58^N and extract last N Base58 digits.

__constant__ unsigned char B58_ALPHABET[58] = {
    '1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z',
    'a','b','c','d','e','f','g','h','i','j','k','m','n','o','p','q','r','s','t','u','v','w','x','y','z'
};

// Suffix pattern as Base58 character indices (reversed: last char first)
__constant__ unsigned char B58_SUFFIX[20];  // up to 20 Base58 chars
__constant__ int B58_SUFFIX_LEN;            // 0 = match all

// Match the last N Base58 characters of the encoded address.
// Input: 25-byte raw address (0x41 || addr20 || checksum4)
__device__ int match_base58_suffix(const unsigned char* raw25) {
    int slen = B58_SUFFIX_LEN;
    if (slen <= 0) return 1;

    // Compute big_int mod 58^N by processing 25 bytes big-endian
    // remainder = 0; for each byte: remainder = (remainder * 256 + byte) % (58^slen)

    // First compute 58^slen (fits in 64-bit for slen <= 10)
    unsigned long long modulus = 1;
    for (int i = 0; i < slen; ++i) modulus *= 58ULL;

    unsigned long long remainder = 0;
    for (int i = 0; i < 25; ++i) {
        remainder = (remainder * 256ULL + (unsigned long long)raw25[i]) % modulus;
    }

    // Extract Base58 digits from remainder (least significant first)
    for (int i = 0; i < slen; ++i) {
        unsigned int digit = (unsigned int)(remainder % 58ULL);
        remainder /= 58ULL;
        // Compare with pattern
        if (B58_ALPHABET[digit] != B58_SUFFIX[i]) return 0;
    }
    return 1;
}

// ====================== Full End-to-End Vanity Search Kernel ======================

#define BATCH_K 128

extern "C" __global__
void secp256k1_vanity_search(
    const unsigned int* base_x,
    const unsigned int* base_y,
    unsigned int* hit_keys,        // output: key indices of hits
    unsigned char* hit_addr,       // output: 25-byte raw addresses of hits (for Base58 on CPU)
    int* hit_count,                // atomic counter
    int max_hits,
    int total_threads,
    int iters_per_thread
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) return;

    unsigned int Px[8], Py[8], Pz[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) { Px[i] = base_x[i]; Py[i] = base_y[i]; }
    Pz[0] = 1;
    #pragma unroll
    for (int i = 1; i < 8; ++i) Pz[i] = 0;

    unsigned int offset = (unsigned int)tid;
    for (int bit = 0; bit < 20 && offset; ++bit) {
        if (!(offset & (1u << bit))) continue;
        point_add_mixed(Px, Py, Pz, &GTABLE[bit][0], &GTABLE[bit][8]);
    }

    int out_base = tid;

    for (int batch = 0; batch < iters_per_thread; ++batch) {
        unsigned int stored_X[BATCH_K][8];
        unsigned int stored_Y[BATCH_K][8];
        unsigned int stored_Z[BATCH_K][8];

        for (int k = 0; k < BATCH_K; ++k) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                stored_X[k][i] = Px[i];
                stored_Y[k][i] = Py[i];
                stored_Z[k][i] = Pz[i];
            }
            point_add_mixed(Px, Py, Pz, STRIDE_GX, STRIDE_GY);
        }

        // Batch inversion
        unsigned int prefix[BATCH_K][8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) prefix[0][i] = stored_Z[0][i];
        for (int k = 1; k < BATCH_K; ++k)
            mul256_mod(prefix[k], prefix[k-1], stored_Z[k]);

        unsigned int inv_acc[8];
        inv256_mod(inv_acc, prefix[BATCH_K - 1]);

        unsigned int z_inv[BATCH_K][8];
        for (int k = BATCH_K - 1; k >= 1; --k) {
            mul256_mod(z_inv[k], inv_acc, prefix[k-1]);
            unsigned int tmp[8];
            mul256_mod(tmp, inv_acc, stored_Z[k]);
            #pragma unroll
            for (int i = 0; i < 8; ++i) inv_acc[i] = tmp[i];
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i) z_inv[0][i] = inv_acc[i];

        // Convert to affine, compute Keccak, SHA-256d, Base58 suffix match
        for (int k = 0; k < BATCH_K; ++k) {
            unsigned int z_inv2[8], z_inv3[8];
            sqr256_mod(z_inv2, z_inv[k]);
            mul256_mod(z_inv3, z_inv2, z_inv[k]);

            unsigned int ax[8], ay[8];
            mul256_mod(ax, stored_X[k], z_inv2);
            mul256_mod(ay, stored_Y[k], z_inv3);

            // Encode X||Y as 64 bytes big-endian
            unsigned char pubxy[64];
            for (int i = 0; i < 8; ++i) {
                int off = (7 - i) * 4;
                pubxy[off+0] = (ax[i] >> 24) & 0xFF;
                pubxy[off+1] = (ax[i] >> 16) & 0xFF;
                pubxy[off+2] = (ax[i] >> 8) & 0xFF;
                pubxy[off+3] = ax[i] & 0xFF;
            }
            for (int i = 0; i < 8; ++i) {
                int off = 32 + (7 - i) * 4;
                pubxy[off+0] = (ay[i] >> 24) & 0xFF;
                pubxy[off+1] = (ay[i] >> 16) & 0xFF;
                pubxy[off+2] = (ay[i] >> 8) & 0xFF;
                pubxy[off+3] = ay[i] & 0xFF;
            }

            // 1. Keccak-256(XY) -> take last 20 bytes
            unsigned char keccak_digest[32];
            keccak256_64bytes(pubxy, keccak_digest);

            // 2. Build raw TRON address: 0x41 || addr20
            unsigned char tron21[21];
            tron21[0] = 0x41;
            #pragma unroll
            for (int b = 0; b < 20; ++b) tron21[b+1] = keccak_digest[12+b];

            // 3. SHA-256d(tron21) -> checksum = first 4 bytes
            unsigned char sha1[32], sha2[32];
            sha256_short(tron21, 21, sha1);
            sha256_short(sha1, 32, sha2);

            // 4. Full 25 bytes: tron21 || checksum[:4]
            unsigned char raw25[25];
            #pragma unroll
            for (int b = 0; b < 21; ++b) raw25[b] = tron21[b];
            #pragma unroll
            for (int b = 0; b < 4; ++b) raw25[21+b] = sha2[b];

            // 5. Check Base58 suffix
            if (match_base58_suffix(raw25)) {
                int slot = atomicAdd(hit_count, 1);
                if (slot < max_hits) {
                    int key_idx = out_base + k * total_threads;
                    hit_keys[slot] = (unsigned int)key_idx;

                    // Store raw 25 bytes for CPU to convert to Base58
                    unsigned char* dst = hit_addr + slot * 25;
                    #pragma unroll
                    for (int b = 0; b < 25; ++b) dst[b] = raw25[b];
                }
            }
        }

        out_base += BATCH_K * total_threads;
    }
}

// Non-fused kernel for correctness testing (outputs all pubkeys)
extern "C" __global__
void secp256k1_incremental_scan(
    const unsigned int* base_x,
    const unsigned int* base_y,
    unsigned char* pubkeys,
    int total_threads,
    int iters_per_thread
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) return;

    unsigned int Px[8], Py[8], Pz[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) { Px[i] = base_x[i]; Py[i] = base_y[i]; }
    Pz[0] = 1;
    #pragma unroll
    for (int i = 1; i < 8; ++i) Pz[i] = 0;

    unsigned int offset = (unsigned int)tid;
    for (int bit = 0; bit < 20 && offset; ++bit) {
        if (!(offset & (1u << bit))) continue;
        point_add_mixed(Px, Py, Pz, &GTABLE[bit][0], &GTABLE[bit][8]);
    }

    int out_base = tid;

    for (int batch = 0; batch < iters_per_thread; ++batch) {
        unsigned int stored_X[BATCH_K][8];
        unsigned int stored_Y[BATCH_K][8];
        unsigned int stored_Z[BATCH_K][8];

        for (int k = 0; k < BATCH_K; ++k) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                stored_X[k][i] = Px[i]; stored_Y[k][i] = Py[i]; stored_Z[k][i] = Pz[i];
            }
            point_add_mixed(Px, Py, Pz, STRIDE_GX, STRIDE_GY);
        }

        unsigned int prefix[BATCH_K][8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) prefix[0][i] = stored_Z[0][i];
        for (int k = 1; k < BATCH_K; ++k)
            mul256_mod(prefix[k], prefix[k-1], stored_Z[k]);
        unsigned int inv_acc[8];
        inv256_mod(inv_acc, prefix[BATCH_K - 1]);
        unsigned int z_inv[BATCH_K][8];
        for (int k = BATCH_K - 1; k >= 1; --k) {
            mul256_mod(z_inv[k], inv_acc, prefix[k-1]);
            unsigned int tmp[8];
            mul256_mod(tmp, inv_acc, stored_Z[k]);
            #pragma unroll
            for (int i = 0; i < 8; ++i) inv_acc[i] = tmp[i];
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i) z_inv[0][i] = inv_acc[i];

        for (int k = 0; k < BATCH_K; ++k) {
            unsigned int z_inv2[8], z_inv3[8];
            sqr256_mod(z_inv2, z_inv[k]);
            mul256_mod(z_inv3, z_inv2, z_inv[k]);
            unsigned int ax[8], ay[8];
            mul256_mod(ax, stored_X[k], z_inv2);
            mul256_mod(ay, stored_Y[k], z_inv3);

            int out_idx = out_base + k * total_threads;
            unsigned char* out = pubkeys + (long long)out_idx * 65;
            out[0] = 0x04;
            for (int i = 0; i < 8; ++i) {
                int off = 1 + (7 - i) * 4;
                out[off+0] = (ax[i] >> 24) & 0xFF; out[off+1] = (ax[i] >> 16) & 0xFF;
                out[off+2] = (ax[i] >> 8) & 0xFF;  out[off+3] = ax[i] & 0xFF;
            }
            for (int i = 0; i < 8; ++i) {
                int off = 33 + (7 - i) * 4;
                out[off+0] = (ay[i] >> 24) & 0xFF; out[off+1] = (ay[i] >> 16) & 0xFF;
                out[off+2] = (ay[i] >> 8) & 0xFF;  out[off+3] = ay[i] & 0xFF;
            }
        }
        out_base += BATCH_K * total_threads;
    }
}

// Addr20 output kernel: secp256k1 -> affine -> Keccak-256 -> output 20-byte address for ALL keys
// This is the optimal kernel for vanity search: 20 bytes per key vs 65 bytes for pubkey
extern "C" __global__
void secp256k1_scan_addr20(
    const unsigned int* base_x,
    const unsigned int* base_y,
    unsigned char* out_addr20,     // (total_keys, 20) output buffer
    int total_threads,
    int iters_per_thread
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) return;

    unsigned int Px[8], Py[8], Pz[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) { Px[i] = base_x[i]; Py[i] = base_y[i]; }
    Pz[0] = 1;
    #pragma unroll
    for (int i = 1; i < 8; ++i) Pz[i] = 0;

    unsigned int offset = (unsigned int)tid;
    for (int bit = 0; bit < 20 && offset; ++bit) {
        if (!(offset & (1u << bit))) continue;
        point_add_mixed(Px, Py, Pz, &GTABLE[bit][0], &GTABLE[bit][8]);
    }

    int out_base = tid;

    for (int batch = 0; batch < iters_per_thread; ++batch) {
        unsigned int stored_X[BATCH_K][8];
        unsigned int stored_Y[BATCH_K][8];
        unsigned int stored_Z[BATCH_K][8];

        for (int k = 0; k < BATCH_K; ++k) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                stored_X[k][i] = Px[i]; stored_Y[k][i] = Py[i]; stored_Z[k][i] = Pz[i];
            }
            point_add_mixed(Px, Py, Pz, STRIDE_GX, STRIDE_GY);
        }

        unsigned int prefix[BATCH_K][8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) prefix[0][i] = stored_Z[0][i];
        for (int k = 1; k < BATCH_K; ++k)
            mul256_mod(prefix[k], prefix[k-1], stored_Z[k]);
        unsigned int inv_acc[8];
        inv256_mod(inv_acc, prefix[BATCH_K - 1]);
        unsigned int z_inv[BATCH_K][8];
        for (int k = BATCH_K - 1; k >= 1; --k) {
            mul256_mod(z_inv[k], inv_acc, prefix[k-1]);
            unsigned int tmp[8];
            mul256_mod(tmp, inv_acc, stored_Z[k]);
            #pragma unroll
            for (int i = 0; i < 8; ++i) inv_acc[i] = tmp[i];
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i) z_inv[0][i] = inv_acc[i];

        for (int k = 0; k < BATCH_K; ++k) {
            unsigned int z_inv2[8], z_inv3[8];
            sqr256_mod(z_inv2, z_inv[k]);
            mul256_mod(z_inv3, z_inv2, z_inv[k]);
            unsigned int ax[8], ay[8];
            mul256_mod(ax, stored_X[k], z_inv2);
            mul256_mod(ay, stored_Y[k], z_inv3);

            // Encode X||Y as 64 bytes big-endian for Keccak input
            unsigned char pubxy[64];
            for (int i = 0; i < 8; ++i) {
                int off = (7 - i) * 4;
                pubxy[off+0] = (ax[i] >> 24) & 0xFF;
                pubxy[off+1] = (ax[i] >> 16) & 0xFF;
                pubxy[off+2] = (ax[i] >>  8) & 0xFF;
                pubxy[off+3] = ax[i] & 0xFF;
            }
            for (int i = 0; i < 8; ++i) {
                int off = 32 + (7 - i) * 4;
                pubxy[off+0] = (ay[i] >> 24) & 0xFF;
                pubxy[off+1] = (ay[i] >> 16) & 0xFF;
                pubxy[off+2] = (ay[i] >>  8) & 0xFF;
                pubxy[off+3] = ay[i] & 0xFF;
            }

            // Keccak-256 + extract last 20 bytes = address
            unsigned char digest[32];
            keccak256_64bytes(pubxy, digest);

            int out_idx = out_base + k * total_threads;
            unsigned char* dst = out_addr20 + (long long)out_idx * 20;
            #pragma unroll
            for (int b = 0; b < 20; ++b) dst[b] = digest[12 + b];
        }
        out_base += BATCH_K * total_threads;
    }
}
"""

# --- Module cache ---
_incremental_module = None
_g_table_uploaded = False
_last_total_threads = None


def _get_module():
    global _incremental_module
    if _incremental_module is None:
        _incremental_module = cp.RawModule(code=_INCREMENTAL_CUDA_SOURCE,
                                           options=('--std=c++11',))
    return _incremental_module


def _ensure_constants(total_threads: int):
    global _g_table_uploaded, _last_total_threads
    mod = _get_module()

    table = precompute_binary_g_table(20)
    table_flat = np.zeros((20, 16), dtype=np.uint32)
    table_flat[:, :8] = table[:, 0, :]
    table_flat[:, 8:] = table[:, 1, :]
    gtable_ptr = mod.get_global('GTABLE')
    dest_gtable = cp.ndarray(table_flat.shape, dtype=cp.uint32, memptr=gtable_ptr)
    dest_gtable[...] = cp.asarray(table_flat)

    sx, sy = precompute_stride_g(total_threads)
    stride_gx_ptr = mod.get_global('STRIDE_GX')
    stride_gy_ptr = mod.get_global('STRIDE_GY')
    dest_sx = cp.ndarray(sx.shape, dtype=cp.uint32, memptr=stride_gx_ptr)
    dest_sy = cp.ndarray(sy.shape, dtype=cp.uint32, memptr=stride_gy_ptr)
    dest_sx[...] = cp.asarray(sx)
    dest_sy[...] = cp.asarray(sy)

    _g_table_uploaded = True
    _last_total_threads = total_threads


def _upload_b58_suffix(suffix: str):
    """Upload Base58 vanity suffix pattern to GPU constant memory."""
    mod = _get_module()

    # Store suffix characters reversed (last char first) for matching
    pattern = np.zeros(20, dtype=np.uint8)
    for i, c in enumerate(reversed(suffix)):
        pattern[i] = ord(c)

    suf_ptr = mod.get_global('B58_SUFFIX')
    dest_suf = cp.ndarray(pattern.shape, dtype=cp.uint8, memptr=suf_ptr)
    dest_suf[...] = cp.asarray(pattern)

    slen_ptr = mod.get_global('B58_SUFFIX_LEN')
    dest_slen = cp.ndarray((1,), dtype=cp.int32, memptr=slen_ptr)
    dest_slen[...] = cp.int32(len(suffix))


def gpu_incremental_scan(
    base_privkey: bytes,
    total_threads: int = 4096,
    iters_per_thread: int = 1,
    threads_per_block: int = 128,
) -> tuple:
    """Run incremental scan on GPU (non-fused, outputs all pubkeys).

    Returns: (pubkeys_gpu, base_privkey_int, total_threads)
    """
    import coincurve

    pk_obj = coincurve.PrivateKey(base_privkey)
    pubkey_uncompressed = pk_obj.public_key.format(compressed=False)
    bx_int = int.from_bytes(pubkey_uncompressed[1:33], 'big')
    by_int = int.from_bytes(pubkey_uncompressed[33:65], 'big')

    bx_limbs = np.array(_int_to_limbs(bx_int), dtype=np.uint32)
    by_limbs = np.array(_int_to_limbs(by_int), dtype=np.uint32)

    _ensure_constants(total_threads)

    total_keys = total_threads * iters_per_thread * BATCH_K
    pubkeys_gpu = cp.zeros((total_keys, 65), dtype=cp.uint8)
    base_x_gpu = cp.asarray(bx_limbs)
    base_y_gpu = cp.asarray(by_limbs)

    mod = _get_module()
    kernel = mod.get_function('secp256k1_incremental_scan')
    blocks = (total_threads + threads_per_block - 1) // threads_per_block

    kernel(
        (blocks,), (threads_per_block,),
        (base_x_gpu, base_y_gpu, pubkeys_gpu, total_threads, iters_per_thread)
    )

    base_int = int.from_bytes(base_privkey, 'big')
    return pubkeys_gpu, base_int, total_threads


def gpu_vanity_scan(
    base_privkey: bytes,
    suffix: str,
    total_threads: int = 32768,
    iters_per_thread: int = 2,
    threads_per_block: int = 128,
) -> tuple:
    """Run full end-to-end GPU vanity search.

    Pipeline: secp256k1 → Keccak-256 → SHA-256d → Base58 suffix → hit buffer

    Args:
        base_privkey: 32-byte private key for base point
        suffix: Base58 suffix to match (e.g. '88888')
        total_threads: number of GPU threads
        iters_per_thread: BATCH_K batches per thread
        threads_per_block: CUDA threads per block

    Returns:
        (hits, total_keys_scanned)
        hits: list of (privkey_bytes, addr_hex, addr_base58) tuples
        total_keys_scanned: total number of keys processed
    """
    import coincurve

    pk_obj = coincurve.PrivateKey(base_privkey)
    pubkey_uncompressed = pk_obj.public_key.format(compressed=False)
    bx_int = int.from_bytes(pubkey_uncompressed[1:33], 'big')
    by_int = int.from_bytes(pubkey_uncompressed[33:65], 'big')

    bx_limbs = np.array(_int_to_limbs(bx_int), dtype=np.uint32)
    by_limbs = np.array(_int_to_limbs(by_int), dtype=np.uint32)

    _ensure_constants(total_threads)
    _upload_b58_suffix(suffix)

    total_keys = total_threads * iters_per_thread * BATCH_K
    base_int = int.from_bytes(base_privkey, 'big')

    # Allocate hit buffers
    hit_keys_gpu = cp.zeros(MAX_HITS, dtype=cp.uint32)
    hit_addr_gpu = cp.zeros((MAX_HITS, 25), dtype=cp.uint8)  # 25-byte raw address
    hit_count_gpu = cp.zeros(1, dtype=cp.int32)

    base_x_gpu = cp.asarray(bx_limbs)
    base_y_gpu = cp.asarray(by_limbs)

    mod = _get_module()
    kernel = mod.get_function('secp256k1_vanity_search')
    blocks = (total_threads + threads_per_block - 1) // threads_per_block

    kernel(
        (blocks,), (threads_per_block,),
        (base_x_gpu, base_y_gpu,
         hit_keys_gpu, hit_addr_gpu, hit_count_gpu,
         cp.int32(MAX_HITS),
         cp.int32(total_threads), cp.int32(iters_per_thread))
    )

    cp.cuda.Device().synchronize()

    # Read results
    n_hits = int(cp.asnumpy(hit_count_gpu)[0])
    n_hits = min(n_hits, MAX_HITS)

    hits = []
    if n_hits > 0:
        hit_keys_cpu = cp.asnumpy(hit_keys_gpu[:n_hits])
        hit_addr_cpu = cp.asnumpy(hit_addr_gpu[:n_hits])

        import base58 as b58mod

        for i in range(n_hits):
            key_idx = int(hit_keys_cpu[i])
            pk_int = base_int + key_idx
            pk_bytes = pk_int.to_bytes(32, 'big')

            # raw25 = 0x41 || addr20 || checksum4 — already computed by GPU
            raw25 = bytes(hit_addr_cpu[i])
            addr_b58 = b58mod.b58encode(raw25).decode()
            addr_hex = raw25[:21].hex()  # just the 0x41 || addr20 part

            hits.append((pk_bytes, addr_hex, addr_b58))

    return hits, total_keys


def gpu_scan_addresses(
    base_privkey: bytes,
    total_threads: int = 32768,
    iters_per_thread: int = 2,
    threads_per_block: int = 128,
) -> tuple:
    """Run incremental scan + Keccak, output 20-byte addresses for all keys.

    This is the optimal kernel for vanity search:
    - Outputs 20 bytes per key (vs 65 for pubkey) = 3.25x less data
    - Returns (addr20_gpu, base_privkey_int, total_threads)
    - addr20_gpu: cp.ndarray (total_keys, 20) uint8
    """
    import coincurve

    pk_obj = coincurve.PrivateKey(base_privkey)
    pubkey_uncompressed = pk_obj.public_key.format(compressed=False)
    bx_int = int.from_bytes(pubkey_uncompressed[1:33], 'big')
    by_int = int.from_bytes(pubkey_uncompressed[33:65], 'big')

    bx_limbs = np.array(_int_to_limbs(bx_int), dtype=np.uint32)
    by_limbs = np.array(_int_to_limbs(by_int), dtype=np.uint32)

    _ensure_constants(total_threads)

    total_keys = total_threads * iters_per_thread * BATCH_K
    addr20_gpu = cp.zeros((total_keys, 20), dtype=cp.uint8)
    base_x_gpu = cp.asarray(bx_limbs)
    base_y_gpu = cp.asarray(by_limbs)

    mod = _get_module()
    kernel = mod.get_function('secp256k1_scan_addr20')
    blocks = (total_threads + threads_per_block - 1) // threads_per_block

    kernel(
        (blocks,), (threads_per_block,),
        (base_x_gpu, base_y_gpu, addr20_gpu,
         cp.int32(total_threads), cp.int32(iters_per_thread))
    )

    base_int = int.from_bytes(base_privkey, 'big')
    return addr20_gpu, base_int, total_threads

