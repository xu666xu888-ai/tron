# -*- coding: utf-8 -*-
"""
GPU secp256k1 橢圓曲線點乘 CUDA kernel
完全在 GPU 上實現 secp256k1 私鑰到公鑰的轉換

實現細節：
- 256-bit 大數模運算
- 橢圓曲線點加法和倍點
- 標量乘法（double-and-add 與 windowing）
- 批量處理多個私鑰

參考：
- secp256k1 參數：p = 2^256 - 2^32 - 977
- 生成點 G = (Gx, Gy)
- 曲線方程：y^2 = x^3 + 7 (mod p)
"""
from __future__ import annotations

import os
import logging
import threading
from typing import Dict, Optional, Tuple

try:
    import cupy as cp
except ImportError as e:  # pragma: no cover - 需要手動安裝 CuPy
    raise ImportError("需要安裝 CuPy：pip install cupy-cuda11x/12x") from e

from .hardware_config import HARDWARE_CONFIG

_DEFAULT_SECP_THREADS = min(128, HARDWARE_CONFIG.secp_threads)
_SECP_THREADS = int(os.environ.get("VANITY_SECP_THREADS", str(_DEFAULT_SECP_THREADS)))

_W4_CACHE_LOCK = threading.Lock()
_W6_CACHE_LOCK = threading.Lock()
_W8_CACHE_LOCK = threading.Lock()

_PRECOMP_W4_CPU: Optional[Tuple[object, object]] = None
_PRECOMP_W4_GPU: Dict[int, bool] = {}

_PRECOMP_W6_CPU: Optional[Tuple[object, object]] = None
_PRECOMP_W6_GPU: Dict[int, bool] = {}

_PRECOMP_W8_CPU: Optional[Tuple[object, object]] = None
_PRECOMP_W8_GPU: Dict[int, bool] = {}

# secp256k1 曲線參數（十六進位）
SECP256K1_P = "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F"
SECP256K1_N = "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141"
SECP256K1_GX = "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798"
SECP256K1_GY = "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8"

_SECP256K1_P_INT = int(SECP256K1_P, 16)
_MONT_FACTOR = pow(2, 256, _SECP256K1_P_INT)


def _convert_table_to_montgomery(table):
    flat = table.reshape(-1, table.shape[-1])
    for row in flat:
        value = 0
        for idx in range(7, -1, -1):
            value = (value << 32) | int(row[idx])
        value = (value * _MONT_FACTOR) % _SECP256K1_P_INT
        converted = value
        for idx in range(8):
            row[idx] = converted & 0xFFFFFFFF
            converted >>= 32
    return table


_COMMON_CUDA_SOURCE = r"""
__constant__ unsigned int SECP256K1_P[8] = {
    0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

__constant__ unsigned int SECP256K1_N[8] = {
    0xD0364141, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6,
    0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

__constant__ unsigned int FIELD_ONE[8] = {
    0x00000001, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000
};

__constant__ unsigned int MONT_ONE[8] = {
    0x000003D1, 0x00000001, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000
};

__constant__ unsigned int MONT_R2[8] = {
    0x000E90A1, 0x000007A2, 0x00000001, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000
};

__constant__ unsigned int SECP256K1_GX_MONT[8] = {
    0x487E2097, 0xD7362E5A, 0x29BC66DB, 0x231E2953,
    0x33FD129C, 0x979F48C0, 0xE9089F48, 0x9981E643
};

__constant__ unsigned int SECP256K1_GY_MONT[8] = {
    0xD3DBABE2, 0xB15EA6D2, 0x1F1DC64D, 0x8DFC5D5D,
    0xAC19C136, 0x70B6B59A, 0xD4A582D6, 0xCF3F851F
};

__constant__ unsigned int MONT_NPRIME = 0xD2253531U;

__device__ int cmp256(const unsigned int* a, const unsigned int* b) {
    for (int i = 7; i >= 0; --i) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}

__device__ void add256_mod(unsigned int* r, const unsigned int* a, const unsigned int* b) {
    unsigned long long carry = 0ULL;
    unsigned int temp[8];
    for (int i = 0; i < 8; ++i) {
        carry += (unsigned long long)a[i] + b[i];
        temp[i] = (unsigned int)carry;
        carry >>= 32;
    }
    if (carry || cmp256(temp, SECP256K1_P) >= 0) {
        carry = 0ULL;
        for (int i = 0; i < 8; ++i) {
            unsigned long long sub = (unsigned long long)temp[i] - SECP256K1_P[i] - carry;
            r[i] = (unsigned int)sub;
            carry = (sub >> 32) & 1ULL;
        }
    } else {
        for (int i = 0; i < 8; ++i) {
            r[i] = temp[i];
        }
    }
}

__device__ void sub256_mod(unsigned int* r, const unsigned int* a, const unsigned int* b) {
    unsigned long long borrow = 0ULL;
    unsigned int temp[8];
    for (int i = 0; i < 8; ++i) {
        unsigned long long sub = (unsigned long long)a[i] - b[i] - borrow;
        temp[i] = (unsigned int)sub;
        borrow = (sub >> 32) & 1ULL;
    }
    if (borrow) {
        unsigned long long carry = 0ULL;
        for (int i = 0; i < 8; ++i) {
            carry += (unsigned long long)temp[i] + SECP256K1_P[i];
            r[i] = (unsigned int)carry;
            carry >>= 32;
        }
    } else {
        for (int i = 0; i < 8; ++i) {
            r[i] = temp[i];
        }
    }
}

__device__ void montgomery_mul(unsigned int* r, const unsigned int* a, const unsigned int* b) {
    unsigned long long t[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) t[i] = 0ULL;

    for (int i = 0; i < 8; ++i) {
        unsigned long long carry = 0ULL;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            unsigned long long prod = t[i + j] + (unsigned long long)a[j] * b[i] + carry;
            t[i + j] = (unsigned int)prod;
            carry = prod >> 32;
        }
        t[i + 8] += carry;

        unsigned int m = (unsigned int)((t[i] & 0xFFFFFFFFULL) * (unsigned long long)MONT_NPRIME);
        carry = 0ULL;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            unsigned long long prod = t[i + j] + (unsigned long long)m * SECP256K1_P[j] + carry;
            t[i + j] = (unsigned int)prod;
            carry = prod >> 32;
        }
        t[i + 8] += carry;
    }

    unsigned long long carry = 0ULL;
    unsigned int res[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        unsigned long long sum = t[i + 8] + carry;
        res[i] = (unsigned int)sum;
        carry = sum >> 32;
    }

    if (carry || cmp256(res, SECP256K1_P) >= 0) {
        unsigned long long borrow = 0ULL;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            unsigned long long sub = (unsigned long long)res[i] - SECP256K1_P[i] - borrow;
            res[i] = (unsigned int)sub;
            borrow = (sub >> 32) & 1ULL;
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        r[i] = res[i];
    }
}

__device__ void to_montgomery(unsigned int* r, const unsigned int* a) {
    montgomery_mul(r, a, MONT_R2);
}

__device__ void from_montgomery(unsigned int* r, const unsigned int* a) {
    montgomery_mul(r, a, FIELD_ONE);
}

__device__ void mul256_mod(unsigned int* r, const unsigned int* a, const unsigned int* b) {
    montgomery_mul(r, a, b);
}

__device__ void sqr256_mod(unsigned int* r, const unsigned int* a) {
    mul256_mod(r, a, a);
}

__device__ void inv256_mod(unsigned int* r, const unsigned int* a) {
    unsigned int exp[8] = {
        0xFFFFFC2D, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
    };

    unsigned int result[8];
    unsigned int base[8];
    for (int i = 0; i < 8; ++i) {
        result[i] = MONT_ONE[i];
        base[i] = a[i];
    }

    for (int i = 0; i < 256; ++i) {
        int word = i >> 5;
        int bit = i & 31;
        if (exp[word] & (1U << bit)) {
            unsigned int temp[8];
            mul256_mod(temp, result, base);
            for (int j = 0; j < 8; ++j) result[j] = temp[j];
        }
        if (i < 255) {
            unsigned int temp[8];
            mul256_mod(temp, base, base);
            for (int j = 0; j < 8; ++j) base[j] = temp[j];
        }
    }
    for (int i = 0; i < 8; ++i) r[i] = result[i];
}

struct Point {
    unsigned int x[8];
    unsigned int y[8];
    unsigned int z[8];
};

__device__ __forceinline__ void point_copy(Point* dst, const Point* src) {
    for (int i = 0; i < 8; ++i) {
        dst->x[i] = src->x[i];
        dst->y[i] = src->y[i];
        dst->z[i] = src->z[i];
    }
}

__device__ void point_set_infinity(Point* p) {
    for (int i = 0; i < 8; ++i) {
        p->x[i] = 0;
        p->y[i] = 0;
        p->z[i] = 0;
    }
}

__device__ bool point_is_infinity(const Point* p) {
    for (int i = 0; i < 8; ++i) {
        if (p->z[i] != 0) return false;
    }
    return true;
}

__device__ void point_double(Point* r, const Point* p) {
    if (point_is_infinity(p)) {
        point_set_infinity(r);
        return;
    }
    unsigned int s[8], m[8], t[8], u[8];
    mul256_mod(t, p->y, p->y);
    mul256_mod(u, p->x, t);
    add256_mod(s, u, u);
    add256_mod(s, s, s);
    mul256_mod(t, p->x, p->x);
    add256_mod(m, t, t);
    add256_mod(m, m, t);
    mul256_mod(r->x, m, m);
    sub256_mod(r->x, r->x, s);
    sub256_mod(r->x, r->x, s);
    sub256_mod(t, s, r->x);
    mul256_mod(r->y, m, t);
    mul256_mod(t, p->y, p->y);
    mul256_mod(t, t, t);
    add256_mod(u, t, t);
    add256_mod(u, u, u);
    add256_mod(u, u, u);
    sub256_mod(r->y, r->y, u);
    mul256_mod(t, p->y, p->z);
    add256_mod(r->z, t, t);
}

__device__ void point_add(Point* r, const Point* p, const Point* q) {
    if (point_is_infinity(p)) {
        point_copy(r, q);
        return;
    }
    if (point_is_infinity(q)) {
        point_copy(r, p);
        return;
    }
    unsigned int u1[8], u2[8], s1[8], s2[8], h[8], rv[8], t[8];
    mul256_mod(t, q->z, q->z);
    mul256_mod(u1, p->x, t);
    mul256_mod(t, p->z, p->z);
    mul256_mod(u2, q->x, t);
    mul256_mod(t, q->z, q->z);
    mul256_mod(t, t, q->z);
    mul256_mod(s1, p->y, t);
    mul256_mod(t, p->z, p->z);
    mul256_mod(t, t, p->z);
    mul256_mod(s2, q->y, t);
    sub256_mod(h, u2, u1);
    sub256_mod(rv, s2, s1);
    bool h0 = true, r0 = true;
    for (int i = 0; i < 8; ++i) {
        if (h[i] != 0) h0 = false;
        if (rv[i] != 0) r0 = false;
    }
    if (h0) {
        if (r0) {
            point_double(r, p);
        } else {
            point_set_infinity(r);
        }
        return;
    }
    unsigned int h2[8], h3[8], u1h2[8], s1h3[8];
    mul256_mod(h2, h, h);
    mul256_mod(h3, h2, h);
    mul256_mod(u1h2, u1, h2);
    mul256_mod(s1h3, s1, h3);
    unsigned int x3[8], y3[8], z3[8];
    mul256_mod(x3, rv, rv);
    sub256_mod(x3, x3, h3);
    sub256_mod(x3, x3, u1h2);
    sub256_mod(x3, x3, u1h2);
    sub256_mod(y3, u1h2, x3);
    mul256_mod(y3, y3, rv);
    sub256_mod(y3, y3, s1h3);
    mul256_mod(z3, p->z, q->z);
    mul256_mod(z3, z3, h);
    for (int i = 0; i < 8; ++i) {
        r->x[i] = x3[i];
        r->y[i] = y3[i];
        r->z[i] = z3[i];
    }
}

__device__ void point_to_affine(unsigned int* x, unsigned int* y, const Point* p) {
    if (point_is_infinity(p)) {
        for (int i = 0; i < 8; ++i) {
            x[i] = 0;
            y[i] = 0;
        }
        return;
    }
    unsigned int z_inv[8], z_inv2[8], z_inv3[8];
    inv256_mod(z_inv, p->z);
    mul256_mod(z_inv2, z_inv, z_inv);
    mul256_mod(z_inv3, z_inv2, z_inv);
    mul256_mod(x, p->x, z_inv2);
    mul256_mod(y, p->y, z_inv3);
    from_montgomery(x, x);
    from_montgomery(y, y);
}
"""

_SECP256K1_KERNEL = _COMMON_CUDA_SOURCE + r"""
extern "C" __global__
void secp256k1_pubkey_batch(
    const unsigned char* privkeys,
    unsigned char* pubkeys,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const unsigned char* priv = privkeys + idx * 32;

    Point result;
    point_set_infinity(&result);

    Point g;
    for (int i = 0; i < 8; ++i) {
        g.x[i] = SECP256K1_GX_MONT[i];
        g.y[i] = SECP256K1_GY_MONT[i];
        g.z[i] = MONT_ONE[i];
    }

    for (int bit_index = 0; bit_index < 256; ++bit_index) {
        Point temp;
        point_double(&temp, &result);
        point_copy(&result, &temp);

        int byte_pos = bit_index >> 3;
        int bit_pos = 7 - (bit_index & 7);
        unsigned int bit = (unsigned int)((priv[byte_pos] >> bit_pos) & 1U);
        if (bit) {
            point_add(&temp, &result, &g);
            point_copy(&result, &temp);
        }
    }

    unsigned int qx[8], qy[8];
    point_to_affine(qx, qy, &result);

    unsigned char* out = pubkeys + idx * 65;
    out[0] = 0x04;
    for (int i = 0; i < 8; ++i) {
        int offset = 1 + (7 - i) * 4;
        out[offset + 0] = (qx[i] >> 24) & 0xFF;
        out[offset + 1] = (qx[i] >> 16) & 0xFF;
        out[offset + 2] = (qx[i] >> 8) & 0xFF;
        out[offset + 3] = qx[i] & 0xFF;
    }
    for (int i = 0; i < 8; ++i) {
        int offset = 33 + (7 - i) * 4;
        out[offset + 0] = (qy[i] >> 24) & 0xFF;
        out[offset + 1] = (qy[i] >> 16) & 0xFF;
        out[offset + 2] = (qy[i] >> 8) & 0xFF;
        out[offset + 3] = qy[i] & 0xFF;
    }
}
"""

_SECP256K1_KERNEL_W4 = _COMMON_CUDA_SOURCE + r"""
__constant__ unsigned int W4_PRECOMP_X[16][8];
__constant__ unsigned int W4_PRECOMP_Y[16][8];

__device__ __forceinline__ bool u256_is_zero(const unsigned long long* limbs){
    return (limbs[0] | limbs[1] | limbs[2] | limbs[3]) == 0ULL;
}

__device__ __forceinline__ void u256_shr1(unsigned long long* limbs){
    unsigned long long carry = 0ULL;
    for(int i=3;i>=0;--i){
        unsigned long long next = limbs[i] & 1ULL;
        limbs[i] = (limbs[i] >> 1) | (carry << 63);
        carry = next;
    }
}

__device__ __forceinline__ void u256_sub_small(unsigned long long* limbs, unsigned int value){
    unsigned long long borrow = value;
    for(int i=0;i<4 && borrow;i++){
        unsigned long long cur = limbs[i];
        unsigned long long res = cur - borrow;
        limbs[i] = res;
        borrow = (cur < borrow) ? 1ULL : 0ULL;
    }
}

__device__ __forceinline__ void u256_add_small(unsigned long long* limbs, unsigned int value){
    unsigned long long carry = value;
    for(int i=0;i<4 && carry;i++){
        unsigned long long cur = limbs[i];
        unsigned long long res = cur + carry;
        limbs[i] = res;
        carry = (res < cur) ? 1ULL : 0ULL;
    }
}

__device__ int wnaf_4(signed char* digits, const unsigned char* scalar_be){
    unsigned long long limbs[4];
    #pragma unroll
    for(int i=0;i<4;++i){
        int base = 24 - i*8;
        unsigned long long v = 0ULL;
        #pragma unroll
        for(int j=0;j<8;++j){
            v = (v << 8) | (unsigned long long)scalar_be[base + j];
        }
        limbs[i] = v;
    }
    int pos = 0;
    while(!u256_is_zero(limbs)){
        int digit = 0;
        if(limbs[0] & 1ULL){
            unsigned int mod = (unsigned int)(limbs[0] & 0xFULL);
            if(mod > 8U) mod -= 16U;
            digit = (int)mod;
            if(digit > 0){
                u256_sub_small(limbs, (unsigned int)digit);
            } else {
                u256_add_small(limbs, (unsigned int)(-digit));
            }
        }
        digits[pos++] = (signed char)digit;
        u256_shr1(limbs);
    }
    return pos;
}

extern "C" __global__
void secp256k1_pubkey_batch_w4(
    const unsigned char* privkeys,
    unsigned char* pubkeys,
    int n
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const unsigned char* sk = privkeys + idx*32;
    unsigned char scalar_be[32];
    #pragma unroll
    for (int b=0;b<32;++b){
        scalar_be[b] = sk[b];
    }

    signed char digits[260];
    int wlen = wnaf_4(digits, scalar_be);

    if (wlen == 0){
        unsigned char* out_zero = pubkeys + idx*65;
        for(int i=0;i<65;++i) out_zero[i] = 0;
        return;
    }

    Point R;
    point_set_infinity(&R);
    for (int i = wlen - 1; i >= 0; --i){
        Point tmp;
        point_double(&tmp, &R);
        point_copy(&R, &tmp);

        int digit = (int)digits[i];
        if (!digit) continue;

        int idx_tbl = digit > 0 ? digit : -digit;
        const unsigned int* px = W4_PRECOMP_X[idx_tbl];
        const unsigned int* py = W4_PRECOMP_Y[idx_tbl];

        Point T;
        for(int limb=0; limb<8; ++limb){
            T.x[limb] = px[limb];
            T.y[limb] = py[limb];
            T.z[limb] = MONT_ONE[limb];
        }

        if (digit < 0){
            unsigned int negy[8];
            sub256_mod(negy, SECP256K1_P, T.y);
            for(int limb=0; limb<8; ++limb){
                T.y[limb] = negy[limb];
            }
        }

        Point summed;
        point_add(&summed, &R, &T);
        point_copy(&R, &summed);
    }

    unsigned int qx[8], qy[8];
    point_to_affine(qx, qy, &R);
    unsigned char* out = pubkeys + idx*65; out[0]=0x04;
    for (int i=0;i<8;++i){ int off=1+(7-i)*4; out[off+0]=(qx[i]>>24)&0xFF; out[off+1]=(qx[i]>>16)&0xFF; out[off+2]=(qx[i]>>8)&0xFF; out[off+3]=qx[i]&0xFF; }
    for (int i=0;i<8;++i){ int off=33+(7-i)*4; out[off+0]=(qy[i]>>24)&0xFF; out[off+1]=(qy[i]>>16)&0xFF; out[off+2]=(qy[i]>>8)&0xFF; out[off+3]=qy[i]&0xFF; }
}
"""

_SECP256K1_KERNEL_W6 = _COMMON_CUDA_SOURCE + r"""
__constant__ unsigned int W6_PRECOMP_X[64][8];
__constant__ unsigned int W6_PRECOMP_Y[64][8];

__device__ __forceinline__ bool u256_is_zero(const unsigned long long* limbs){
    return (limbs[0] | limbs[1] | limbs[2] | limbs[3]) == 0ULL;
}

__device__ __forceinline__ void u256_shr1(unsigned long long* limbs){
    unsigned long long carry = 0ULL;
    for(int i=3;i>=0;--i){
        unsigned long long next = limbs[i] & 1ULL;
        limbs[i] = (limbs[i] >> 1) | (carry << 63);
        carry = next;
    }
}

__device__ __forceinline__ void u256_sub_small(unsigned long long* limbs, unsigned int value){
    unsigned long long borrow = value;
    for(int i=0;i<4 && borrow;i++){
        unsigned long long cur = limbs[i];
        unsigned long long res = cur - borrow;
        limbs[i] = res;
        borrow = (cur < borrow) ? 1ULL : 0ULL;
    }
}

__device__ __forceinline__ void u256_add_small(unsigned long long* limbs, unsigned int value){
    unsigned long long carry = value;
    for(int i=0;i<4 && carry;i++){
        unsigned long long cur = limbs[i];
        unsigned long long res = cur + carry;
        limbs[i] = res;
        carry = (res < cur) ? 1ULL : 0ULL;
    }
}

__device__ int wnaf_6(signed char* digits, const unsigned char* scalar_be){
    unsigned long long limbs[4];
    #pragma unroll
    for(int i=0;i<4;++i){
        int base = 24 - i*8;
        unsigned long long v = 0ULL;
        #pragma unroll
        for(int j=0;j<8;++j){
            v = (v << 8) | (unsigned long long)scalar_be[base + j];
        }
        limbs[i] = v;
    }
    int pos = 0;
    while(!u256_is_zero(limbs)){
        int digit = 0;
        if(limbs[0] & 1ULL){
            unsigned int mod = (unsigned int)(limbs[0] & 0x3FULL);
            if(mod > 32U) mod -= 64U;
            digit = (int)mod;
            if(digit > 0){
                u256_sub_small(limbs, (unsigned int)digit);
            } else {
                u256_add_small(limbs, (unsigned int)(-digit));
            }
        }
        digits[pos++] = (signed char)digit;
        u256_shr1(limbs);
    }
    return pos;
}

extern "C" __global__
void secp256k1_pubkey_batch_w6(
    const unsigned char* privkeys,
    unsigned char* pubkeys,
    int n
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const unsigned char* sk = privkeys + idx*32;
    unsigned char scalar_be[32];
    #pragma unroll
    for (int b=0;b<32;++b){
        scalar_be[b] = sk[b];
    }

    signed char digits[260];
    int wlen = wnaf_6(digits, scalar_be);

    if (wlen == 0){
        unsigned char* out_zero = pubkeys + idx*65;
        for(int i=0;i<65;++i) out_zero[i] = 0;
        return;
    }

    Point R;
    point_set_infinity(&R);
    for (int i = wlen - 1; i >= 0; --i){
        Point tmp;
        point_double(&tmp, &R);
        point_copy(&R, &tmp);

        int digit = (int)digits[i];
        if (!digit) continue;

        int idx_tbl = digit > 0 ? digit : -digit;
        const unsigned int* px = W6_PRECOMP_X[idx_tbl];
        const unsigned int* py = W6_PRECOMP_Y[idx_tbl];

        Point T;
        for(int limb=0; limb<8; ++limb){
            T.x[limb] = px[limb];
            T.y[limb] = py[limb];
            T.z[limb] = MONT_ONE[limb];
        }

        if (digit < 0){
            unsigned int negy[8];
            sub256_mod(negy, SECP256K1_P, T.y);
            for(int limb=0; limb<8; ++limb){
                T.y[limb] = negy[limb];
            }
        }

        Point summed;
        point_add(&summed, &R, &T);
        point_copy(&R, &summed);
    }

    unsigned int qx[8], qy[8];
    point_to_affine(qx, qy, &R);
    unsigned char* out = pubkeys + idx*65; out[0]=0x04;
    for (int i=0;i<8;++i){ int off=1+(7-i)*4; out[off+0]=(qx[i]>>24)&0xFF; out[off+1]=(qx[i]>>16)&0xFF; out[off+2]=(qx[i]>>8)&0xFF; out[off+3]=qx[i]&0xFF; }
    for (int i=0;i<8;++i){ int off=33+(7-i)*4; out[off+0]=(qy[i]>>24)&0xFF; out[off+1]=(qy[i]>>16)&0xFF; out[off+2]=(qy[i]>>8)&0xFF; out[off+3]=qy[i]&0xFF; }
}
"""

_SECP256K1_KERNEL_W8 = _COMMON_CUDA_SOURCE + r"""
__constant__ unsigned int W8_PRECOMP_X[256][8];
__constant__ unsigned int W8_PRECOMP_Y[256][8];

__device__ __forceinline__ bool u256_is_zero(const unsigned long long* limbs){
    return (limbs[0] | limbs[1] | limbs[2] | limbs[3]) == 0ULL;
}

__device__ __forceinline__ void u256_shr1(unsigned long long* limbs){
    unsigned long long carry = 0ULL;
    for(int i=3;i>=0;--i){
        unsigned long long next = limbs[i] & 1ULL;
        limbs[i] = (limbs[i] >> 1) | (carry << 63);
        carry = next;
    }
}

__device__ __forceinline__ void u256_sub_small(unsigned long long* limbs, unsigned int value){
    unsigned long long borrow = value;
    for(int i=0;i<4 && borrow;i++){
        unsigned long long cur = limbs[i];
        unsigned long long res = cur - borrow;
        limbs[i] = res;
        borrow = (cur < borrow) ? 1ULL : 0ULL;
    }
}

__device__ __forceinline__ void u256_add_small(unsigned long long* limbs, unsigned int value){
    unsigned long long carry = value;
    for(int i=0;i<4 && carry;i++){
        unsigned long long cur = limbs[i];
        unsigned long long res = cur + carry;
        limbs[i] = res;
        carry = (res < cur) ? 1ULL : 0ULL;
    }
}

__device__ int wnaf_8(signed char* digits, const unsigned char* scalar_be){
    unsigned long long limbs[4];
    #pragma unroll
    for(int i=0;i<4;++i){
        int base = 24 - i*8;
        unsigned long long v = 0ULL;
        #pragma unroll
        for(int j=0;j<8;++j){
            v = (v << 8) | (unsigned long long)scalar_be[base + j];
        }
        limbs[i] = v;
    }
    int pos = 0;
    while(!u256_is_zero(limbs)){
        int digit = 0;
        if(limbs[0] & 1ULL){
            unsigned int mod = (unsigned int)(limbs[0] & 0xFFULL);
            if(mod > 128U) mod -= 256U;
            digit = (int)mod;
            if(digit > 0){
                u256_sub_small(limbs, (unsigned int)digit);
            } else {
                u256_add_small(limbs, (unsigned int)(-digit));
            }
        }
        digits[pos++] = (signed char)digit;
        u256_shr1(limbs);
    }
    return pos;
}

extern "C" __global__
void secp256k1_pubkey_batch_w8(
    const unsigned char* privkeys,
    unsigned char* pubkeys,
    int n
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const unsigned char* sk = privkeys + idx*32;
    unsigned char scalar_be[32];
    #pragma unroll
    for (int b=0;b<32;++b){
        scalar_be[b] = sk[b];
    }

    signed char digits[260];
    int wlen = wnaf_8(digits, scalar_be);

    if (wlen == 0){
        unsigned char* out_zero = pubkeys + idx*65;
        for(int i=0;i<65;++i) out_zero[i] = 0;
        return;
    }

    Point R;
    point_set_infinity(&R);
    for (int i = wlen - 1; i >= 0; --i){
        Point tmp;
        point_double(&tmp, &R);
        point_copy(&R, &tmp);

        int digit = (int)digits[i];
        if (!digit) continue;

        int idx_tbl = digit > 0 ? digit : -digit;
        const unsigned int* px = W8_PRECOMP_X[idx_tbl];
        const unsigned int* py = W8_PRECOMP_Y[idx_tbl];

        Point T;
        for(int limb=0; limb<8; ++limb){
            T.x[limb] = px[limb];
            T.y[limb] = py[limb];
            T.z[limb] = MONT_ONE[limb];
        }

        if (digit < 0){
            unsigned int negy[8];
            sub256_mod(negy, SECP256K1_P, T.y);
            for(int limb=0; limb<8; ++limb){
                T.y[limb] = negy[limb];
            }
        }

        Point summed;
        point_add(&summed, &R, &T);
        point_copy(&R, &summed);
    }

    unsigned int qx[8], qy[8];
    point_to_affine(qx, qy, &R);
    unsigned char* out = pubkeys + idx*65; out[0]=0x04;
    for (int i=0;i<8;++i){ int off=1+(7-i)*4; out[off+0]=(qx[i]>>24)&0xFF; out[off+1]=(qx[i]>>16)&0xFF; out[off+2]=(qx[i]>>8)&0xFF; out[off+3]=qx[i]&0xFF; }
    for (int i=0;i<8;++i){ int off=33+(7-i)*4; out[off+0]=(qy[i]>>24)&0xFF; out[off+1]=(qy[i]>>16)&0xFF; out[off+2]=(qy[i]>>8)&0xFF; out[off+3]=qy[i]&0xFF; }
}
"""

_secp256k1_module = cp.RawModule(code=_SECP256K1_KERNEL, options=("-std=c++11",))
_secp256k1_kernel = _secp256k1_module.get_function("secp256k1_pubkey_batch")

_secp256k1_module_w4 = cp.RawModule(code=_SECP256K1_KERNEL_W4, options=("-std=c++11",))
_secp256k1_kernel_w4 = _secp256k1_module_w4.get_function("secp256k1_pubkey_batch_w4")

_secp256k1_module_w6 = cp.RawModule(code=_SECP256K1_KERNEL_W6, options=("-std=c++11",))
_secp256k1_kernel_w6 = _secp256k1_module_w6.get_function("secp256k1_pubkey_batch_w6")

_secp256k1_module_w8 = cp.RawModule(code=_SECP256K1_KERNEL_W8, options=("-std=c++11",))
_secp256k1_kernel_w8 = _secp256k1_module_w8.get_function("secp256k1_pubkey_batch_w8")

_LOGGER = logging.getLogger(__name__)


def gpu_secp256k1_batch(privkeys_gpu: "cp.ndarray", out: Optional["cp.ndarray"] = None) -> "cp.ndarray":
    """
    在 GPU 上批量計算 secp256k1 公鑰（未壓縮 65 bytes）。
    """
    if privkeys_gpu.dtype != cp.uint8 or privkeys_gpu.ndim != 2 or privkeys_gpu.shape[1] != 32:
        raise ValueError("privkeys_gpu 需為 uint8 (N,32)")

    n = privkeys_gpu.shape[0]
    if out is None:
        pubkeys_gpu = cp.empty((n, 65), dtype=cp.uint8)
    else:
        if out.dtype != cp.uint8 or out.ndim != 2 or out.shape[0] != n or out.shape[1] != 65:
            raise ValueError("out 需為 uint8 (N,65)")
        if out.strides[1] != 1:
            raise ValueError("out 必須為連續記憶體陣列")
        pubkeys_gpu = out

    threads = _SECP_THREADS
    blocks = (n + threads - 1) // threads

    _secp256k1_kernel(
        (blocks,), (threads,),
        (privkeys_gpu, pubkeys_gpu, cp.int32(n))
    )

    return pubkeys_gpu


def _build_precomp_table_w4_cpu():
    global _PRECOMP_W4_CPU
    if _PRECOMP_W4_CPU is not None:
        return _PRECOMP_W4_CPU

    import numpy as np
    import coincurve

    xs = np.zeros((16, 8), dtype=np.uint32)
    ys = np.zeros((16, 8), dtype=np.uint32)
    for k in range(1, 16):
        sk = (k).to_bytes(32, "big")
        pk = coincurve.PrivateKey(sk).public_key.format(compressed=False)
        x = pk[1:33]
        y = pk[33:65]
        for i in range(8):
            xs[k, i] = int.from_bytes(x[28 - 4 * i : 32 - 4 * i], "big")
            ys[k, i] = int.from_bytes(y[28 - 4 * i : 32 - 4 * i], "big")

    _convert_table_to_montgomery(xs)
    _convert_table_to_montgomery(ys)

    _PRECOMP_W4_CPU = (xs, ys)
    return _PRECOMP_W4_CPU


def _build_precomp_table_w6_cpu():
    global _PRECOMP_W6_CPU
    if _PRECOMP_W6_CPU is not None:
        return _PRECOMP_W6_CPU

    import numpy as np
    import coincurve

    xs = np.zeros((64, 8), dtype=np.uint32)
    ys = np.zeros((64, 8), dtype=np.uint32)
    for k in range(1, 64):
        sk = (k).to_bytes(32, "big")
        pk = coincurve.PrivateKey(sk).public_key.format(compressed=False)
        x = pk[1:33]
        y = pk[33:65]
        for i in range(8):
            xs[k, i] = int.from_bytes(x[28 - 4 * i : 32 - 4 * i], "big")
            ys[k, i] = int.from_bytes(y[28 - 4 * i : 32 - 4 * i], "big")

    _convert_table_to_montgomery(xs)
    _convert_table_to_montgomery(ys)

    _PRECOMP_W6_CPU = (xs, ys)
    return _PRECOMP_W6_CPU


def _build_precomp_table_w8_cpu():
    global _PRECOMP_W8_CPU
    if _PRECOMP_W8_CPU is not None:
        return _PRECOMP_W8_CPU

    import numpy as np
    import coincurve

    size = 1 << 8
    xs = np.zeros((size, 8), dtype=np.uint32)
    ys = np.zeros((size, 8), dtype=np.uint32)
    for k in range(1, size):
        sk = (k).to_bytes(32, "big")
        pk = coincurve.PrivateKey(sk).public_key.format(compressed=False)
        x = pk[1:33]
        y = pk[33:65]
        for i in range(8):
            xs[k, i] = int.from_bytes(x[28 - 4 * i : 32 - 4 * i], "big")
            ys[k, i] = int.from_bytes(y[28 - 4 * i : 32 - 4 * i], "big")

    _convert_table_to_montgomery(xs)
    _convert_table_to_montgomery(ys)

    _PRECOMP_W8_CPU = (xs, ys)
    return _PRECOMP_W8_CPU


def _ensure_precomp_table_w4() -> None:
    dev_id = int(cp.cuda.Device())
    with _W4_CACHE_LOCK:
        if _PRECOMP_W4_GPU.get(dev_id):
            return
        cpu_xs, cpu_ys = _build_precomp_table_w4_cpu()
        module = _secp256k1_module_w4
        ptr_x = module.get_global("W4_PRECOMP_X")
        ptr_y = module.get_global("W4_PRECOMP_Y")
        dest_x = cp.ndarray(cpu_xs.shape, dtype=cp.uint32, memptr=ptr_x)
        dest_y = cp.ndarray(cpu_ys.shape, dtype=cp.uint32, memptr=ptr_y)
        dest_x[...] = cp.asarray(cpu_xs)
        dest_y[...] = cp.asarray(cpu_ys)
        _PRECOMP_W4_GPU[dev_id] = True


def _ensure_precomp_table_w6() -> None:
    dev_id = int(cp.cuda.Device())
    with _W6_CACHE_LOCK:
        if _PRECOMP_W6_GPU.get(dev_id):
            return
        cpu_xs, cpu_ys = _build_precomp_table_w6_cpu()
        module = _secp256k1_module_w6
        ptr_x = module.get_global("W6_PRECOMP_X")
        ptr_y = module.get_global("W6_PRECOMP_Y")
        dest_x = cp.ndarray(cpu_xs.shape, dtype=cp.uint32, memptr=ptr_x)
        dest_y = cp.ndarray(cpu_ys.shape, dtype=cp.uint32, memptr=ptr_y)
        dest_x[...] = cp.asarray(cpu_xs)
        dest_y[...] = cp.asarray(cpu_ys)
        _PRECOMP_W6_GPU[dev_id] = True


def _ensure_precomp_table_w8() -> None:
    dev_id = int(cp.cuda.Device())
    with _W8_CACHE_LOCK:
        if _PRECOMP_W8_GPU.get(dev_id):
            return
        cpu_xs, cpu_ys = _build_precomp_table_w8_cpu()
        module = _secp256k1_module_w8
        ptr_x = module.get_global("W8_PRECOMP_X")
        ptr_y = module.get_global("W8_PRECOMP_Y")
        dest_x = cp.ndarray(cpu_xs.shape, dtype=cp.uint32, memptr=ptr_x)
        dest_y = cp.ndarray(cpu_ys.shape, dtype=cp.uint32, memptr=ptr_y)
        dest_x[...] = cp.asarray(cpu_xs)
        dest_y[...] = cp.asarray(cpu_ys)
        _PRECOMP_W8_GPU[dev_id] = True


def warmup_window4_table(force: bool = False) -> None:
    dev_id = int(cp.cuda.Device())
    if force:
        with _W4_CACHE_LOCK:
            _PRECOMP_W4_GPU.pop(dev_id, None)
    _ensure_precomp_table_w4()


def warmup_window6_table(force: bool = False) -> None:
    dev_id = int(cp.cuda.Device())
    if force:
        with _W6_CACHE_LOCK:
            _PRECOMP_W6_GPU.pop(dev_id, None)
    _ensure_precomp_table_w6()


def warmup_window8_table(force: bool = False) -> None:
    dev_id = int(cp.cuda.Device())
    if force:
        with _W8_CACHE_LOCK:
            _PRECOMP_W8_GPU.pop(dev_id, None)
    _ensure_precomp_table_w8()


def gpu_secp256k1_batch_window4(privkeys_gpu: "cp.ndarray", out: Optional["cp.ndarray"] = None) -> "cp.ndarray":
    if privkeys_gpu.dtype != cp.uint8 or privkeys_gpu.ndim != 2 or privkeys_gpu.shape[1] != 32:
        raise ValueError("privkeys_gpu 需為 uint8 (N,32)")
    n = privkeys_gpu.shape[0]
    if out is None:
        pub = cp.empty((n, 65), dtype=cp.uint8)
    else:
        if out.dtype != cp.uint8 or out.ndim != 2 or out.shape[0] != n or out.shape[1] != 65:
            raise ValueError("out 需為 uint8 (N,65)")
        if out.strides[1] != 1:
            raise ValueError("out 必須為連續記憶體陣列")
        pub = out
    _ensure_precomp_table_w4()
    threads = _SECP_THREADS
    blocks = (n + threads - 1)//threads
    _secp256k1_kernel_w4((blocks,), (threads,), (privkeys_gpu, pub, cp.int32(n)))
    return pub


def gpu_secp256k1_batch_window6(privkeys_gpu: "cp.ndarray", out: Optional["cp.ndarray"] = None) -> "cp.ndarray":
    if privkeys_gpu.dtype != cp.uint8 or privkeys_gpu.ndim != 2 or privkeys_gpu.shape[1] != 32:
        raise ValueError("privkeys_gpu 需為 uint8 (N,32)")
    n = privkeys_gpu.shape[0]
    if out is None:
        pub = cp.empty((n, 65), dtype=cp.uint8)
    else:
        if out.dtype != cp.uint8 or out.ndim != 2 or out.shape[0] != n or out.shape[1] != 65:
            raise ValueError("out 需為 uint8 (N,65)")
        if out.strides[1] != 1:
            raise ValueError("out 必須為連續記憶體陣列")
        pub = out
    _ensure_precomp_table_w6()
    threads = _SECP_THREADS
    blocks = (n + threads - 1)//threads
    _secp256k1_kernel_w6((blocks,), (threads,), (privkeys_gpu, pub, cp.int32(n)))
    return pub


def gpu_secp256k1_batch_window8(privkeys_gpu: "cp.ndarray", out: Optional["cp.ndarray"] = None) -> "cp.ndarray":
    if privkeys_gpu.dtype != cp.uint8 or privkeys_gpu.ndim != 2 or privkeys_gpu.shape[1] != 32:
        raise ValueError("privkeys_gpu 需為 uint8 (N,32)")
    n = privkeys_gpu.shape[0]
    if out is None:
        pub = cp.empty((n, 65), dtype=cp.uint8)
    else:
        if out.dtype != cp.uint8 or out.ndim != 2 or out.shape[0] != n or out.shape[1] != 65:
            raise ValueError("out 需為 uint8 (N,65)")
        if out.strides[1] != 1:
            raise ValueError("out 必須為連續記憶體陣列")
        pub = out
    _ensure_precomp_table_w8()
    threads = _SECP_THREADS
    blocks = (n + threads - 1)//threads
    _secp256k1_kernel_w8((blocks,), (threads,), (privkeys_gpu, pub, cp.int32(n)))
    return pub


__all__ = [
    "gpu_secp256k1_batch",
    "gpu_secp256k1_batch_window4",
    "gpu_secp256k1_batch_window6",
    "gpu_secp256k1_batch_window8",
    "warmup_window4_table",
    "warmup_window6_table",
    "warmup_window8_table",
]
