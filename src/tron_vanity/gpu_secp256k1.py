# -*- coding: utf-8 -*-
"""
GPU secp256k1 橢圓曲線點乘 CUDA kernel
完全在 GPU 上實現 secp256k1 私鑰到公鑰的轉換

實現細節：
- 256-bit 大數模運算（Montgomery 形式）
- 橢圓曲線點加法和倍點
- 標量乘法（double-and-add with windowing）
- 批量處理多個私鑰

參考：
- secp256k1 參數：p = 2^256 - 2^32 - 977
- 生成點 G = (Gx, Gy)
- 曲線方程：y^2 = x^3 + 7 (mod p)
"""
from __future__ import annotations

try:
    import cupy as cp
except ImportError as e:
    raise ImportError("需要安裝 CuPy：pip install cupy-cuda11x/12x") from e

from typing import Dict, Optional, Tuple

import threading

_W4_CACHE_LOCK = threading.Lock()

_PRECOMP_W4_CPU: Optional[Tuple[object, object]] = None
_PRECOMP_W4_GPU: Dict[int, Tuple[cp.ndarray, cp.ndarray]] = {}

# secp256k1 曲線參數（十六進位）
SECP256K1_P = "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F"
SECP256K1_N = "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141"
SECP256K1_GX = "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798"
SECP256K1_GY = "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8"

# CUDA kernel 代碼
_SECP256K1_KERNEL = r"""
// secp256k1 參數（256-bit，以 8 個 uint32 表示，小端序）
__constant__ unsigned int SECP256K1_P[8] = {
    0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

__constant__ unsigned int SECP256K1_N[8] = {
    0xD0364141, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6,
    0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

// 生成點 G
__constant__ unsigned int SECP256K1_GX[8] = {
    0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB,
    0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E
};

__constant__ unsigned int SECP256K1_GY[8] = {
    0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448,
    0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77
};

// ===== 256-bit 大數運算 =====

// 比較：a > b 返回 1，a == b 返回 0，a < b 返回 -1
__device__ int cmp256(const unsigned int* a, const unsigned int* b) {
    for (int i = 7; i >= 0; --i) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}

// 加法：r = a + b (mod p)
__device__ void add256_mod(unsigned int* r, const unsigned int* a, const unsigned int* b) {
    unsigned long long carry = 0;
    unsigned int temp[8];
    
    // 先做普通加法
    for (int i = 0; i < 8; ++i) {
        carry += (unsigned long long)a[i] + b[i];
        temp[i] = (unsigned int)carry;
        carry >>= 32;
    }
    
    // 如果溢出或 >= p，則減去 p
    if (carry || cmp256(temp, SECP256K1_P) >= 0) {
        carry = 0;
        for (int i = 0; i < 8; ++i) {
            unsigned long long sub = (unsigned long long)temp[i] - SECP256K1_P[i] - carry;
            r[i] = (unsigned int)sub;
            carry = (sub >> 32) & 1;
        }
    } else {
        for (int i = 0; i < 8; ++i) {
            r[i] = temp[i];
        }
    }
}

// 減法：r = a - b (mod p)
__device__ void sub256_mod(unsigned int* r, const unsigned int* a, const unsigned int* b) {
    unsigned long long borrow = 0;
    unsigned int temp[8];
    
    // 先做普通減法
    for (int i = 0; i < 8; ++i) {
        unsigned long long sub = (unsigned long long)a[i] - b[i] - borrow;
        temp[i] = (unsigned int)sub;
        borrow = (sub >> 32) & 1;
    }
    
    // 如果借位，則加上 p
    if (borrow) {
        unsigned long long carry = 0;
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

// 輔助：將 8x32 位元整數與 (H << 32) 相加，輸出 9x32（第 9 limb 為最終進位）
__device__ void add_shift32_9(unsigned int* out9, const unsigned int* L, const unsigned int* H) {
    unsigned long long carry = 0ULL;
    unsigned long long acc = (unsigned long long)L[0] + 0ULL;
    out9[0] = (unsigned int)acc;
    carry = acc >> 32;
    for (int i = 1; i < 8; ++i) {
        acc = (unsigned long long)L[i] + (unsigned long long)H[i-1] + carry;
        out9[i] = (unsigned int)acc;
        carry = acc >> 32;
    }
    out9[8] = (unsigned int)carry;
}

// 輔助：將 8x32 乘以常數 977，輸出 9x32（第 9 limb 為最終進位）
__device__ void mul_const_977_9(unsigned int* out9, const unsigned int* H) {
    unsigned long long carry = 0ULL;
    for (int i = 0; i < 8; ++i) {
        unsigned long long acc = (unsigned long long)H[i] * 977ULL + carry;
        out9[i] = (unsigned int)acc;
        carry = acc >> 32;
    }
    out9[8] = (unsigned int)carry;
}

// 特殊素數 p=2^256-2^32-977 快速約簡（迭代摺疊法）
// 對第 k>=8 limb：折疊回 k-8 與 k-7（乘 977 與移位 32 bits）
__device__ void reduce_p(unsigned int* r, const unsigned int* T) {
    unsigned int X[20];
    for (int i=0;i<20;++i) X[i]=0U;
    for (int i=0;i<16;++i) X[i] = T[i];

    for (int k = 15; k >= 8; --k) {
        unsigned int u = X[k];
        if (!u) continue;
        X[k] = 0U;
        // 折疊到 k-8：乘以 977
        unsigned long long acc = (unsigned long long)X[k-8] + (unsigned long long)u * 977ULL;
        X[k-8] = (unsigned int)acc;
        unsigned long long carry = acc >> 32;
        int idx = k-7;
        while (carry) {
            unsigned long long a2 = (unsigned long long)X[idx] + carry;
            X[idx] = (unsigned int)a2;
            carry = a2 >> 32;
            ++idx;
        }
        // 折疊到 k-7：加上 u（對應 2^32）
        acc = (unsigned long long)X[k-7] + (unsigned long long)u;
        X[k-7] = (unsigned int)acc;
        carry = acc >> 32;
        idx = k-6;
        while (carry) {
            unsigned long long a3 = (unsigned long long)X[idx] + carry;
            X[idx] = (unsigned int)a3;
            carry = a3 >> 32;
            ++idx;
        }
    }

    // 若仍有更高 limb（因進位擴散到 16..19），持續折疊直到 8..19 皆為 0
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
            int idx = k-7;
            while (carry) {
                unsigned long long a2 = (unsigned long long)X[idx] + carry;
                X[idx] = (unsigned int)a2;
                carry = a2 >> 32;
                ++idx;
            }
            acc = (unsigned long long)X[k-7] + (unsigned long long)u;
            X[k-7] = (unsigned int)acc;
            carry = acc >> 32;
            idx = k-6;
            while (carry) {
                unsigned long long a3 = (unsigned long long)X[idx] + carry;
                X[idx] = (unsigned int)a3;
                carry = a3 >> 32;
                ++idx;
            }
        }
    }

    for (int i=0;i<8;++i) r[i] = X[i];
    // 最後確保 < p
    while (cmp256(r, SECP256K1_P) >= 0) {
        unsigned long long br=0ULL; for (int i=0;i<8;++i){ unsigned long long sub=(unsigned long long)r[i]-SECP256K1_P[i]-br; r[i]=(unsigned int)sub; br=(sub>>32)&1ULL; }
    }
}

// 乘法：r = a * b (mod p) - 使用 secp256k1 特殊素數快速約簡
__device__ void mul256_mod(unsigned int* r, const unsigned int* a, const unsigned int* b) {
    unsigned int T[16];
    // 乘積（schoolbook，小端 32-bit）
    for (int i=0;i<16;++i) T[i]=0U;
    for (int i = 0; i < 8; ++i) {
        unsigned long long carry = 0ULL;
        for (int j = 0; j < 8; ++j) {
            unsigned long long acc = (unsigned long long)T[i+j] + (unsigned long long)a[i]*(unsigned long long)b[j] + carry;
            T[i+j] = (unsigned int)acc;
            carry = acc >> 32;
        }
        // propagate carry
        unsigned long long acc2 = (unsigned long long)T[i+8] + carry;
        T[i+8] = (unsigned int)acc2;
        unsigned long long c2 = acc2 >> 32;
        int k = i+9;
        while (c2 && k < 16) {
            unsigned long long acc3 = (unsigned long long)T[k] + c2;
            T[k] = (unsigned int)acc3;
            c2 = acc3 >> 32;
            ++k;
        }
    }
    reduce_p(r, T);
}

// 模逆：r = a^(-1) (mod p) - 使用費馬小定理：a^(p-2) mod p
__device__ void inv256_mod(unsigned int* r, const unsigned int* a) {
    // p - 2 = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
    unsigned int exp[8] = {
        0xFFFFFC2D, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
    };
    
    unsigned int result[8] = {1, 0, 0, 0, 0, 0, 0, 0};
    unsigned int base[8];
    for (int i = 0; i < 8; ++i) base[i] = a[i];
    
    // 二進位快速冪
    for (int i = 0; i < 256; ++i) {
        int word = i / 32;
        int bit = i % 32;
        
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

// ===== 橢圓曲線點運算 =====

// 點結構（Jacobian 座標：X, Y, Z）
struct Point {
    unsigned int x[8];
    unsigned int y[8];
    unsigned int z[8];
};

// 無窮遠點
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

// 點倍增：R = 2P (Jacobian 座標)
__device__ void point_double(Point* r, const Point* p) {
    if (point_is_infinity(p)) {
        point_set_infinity(r);
        return;
    }
    
    unsigned int s[8], m[8], t[8], u[8];
    
    // S = 4*X*Y^2
    mul256_mod(t, p->y, p->y);  // Y^2
    mul256_mod(u, p->x, t);     // X*Y^2
    add256_mod(s, u, u);        // 2*X*Y^2
    add256_mod(s, s, s);        // 4*X*Y^2
    
    // M = 3*X^2 (因為 a=0)
    mul256_mod(t, p->x, p->x);  // X^2
    add256_mod(m, t, t);        // 2*X^2
    add256_mod(m, m, t);        // 3*X^2
    
    // X' = M^2 - 2*S
    mul256_mod(r->x, m, m);     // M^2
    sub256_mod(r->x, r->x, s);  // M^2 - S
    sub256_mod(r->x, r->x, s);  // M^2 - 2*S
    
    // Y' = M*(S - X') - 8*Y^4
    sub256_mod(t, s, r->x);     // S - X'
    mul256_mod(r->y, m, t);     // M*(S - X')
    mul256_mod(t, p->y, p->y);  // Y^2
    mul256_mod(t, t, t);        // Y^4
    add256_mod(u, t, t);        // 2*Y^4
    add256_mod(u, u, u);        // 4*Y^4
    add256_mod(u, u, u);        // 8*Y^4
    sub256_mod(r->y, r->y, u);  // M*(S - X') - 8*Y^4
    
    // Z' = 2*Y*Z
    mul256_mod(t, p->y, p->z);  // Y*Z
    add256_mod(r->z, t, t);     // 2*Y*Z
}

// 點加法：R = P + Q (Jacobian 座標)
__device__ void point_add(Point* r, const Point* p, const Point* q) {
    if (point_is_infinity(p)) {
        *r = *q;
        return;
    }
    if (point_is_infinity(q)) {
        *r = *p;
        return;
    }
    
    unsigned int u1[8], u2[8], s1[8], s2[8], h[8], r_val[8], t[8];
    
    // U1 = X1*Z2^2
    mul256_mod(t, q->z, q->z);
    mul256_mod(u1, p->x, t);
    
    // U2 = X2*Z1^2
    mul256_mod(t, p->z, p->z);
    mul256_mod(u2, q->x, t);
    
    // S1 = Y1*Z2^3
    mul256_mod(t, q->z, q->z);
    mul256_mod(t, t, q->z);
    mul256_mod(s1, p->y, t);
    
    // S2 = Y2*Z1^3
    mul256_mod(t, p->z, p->z);
    mul256_mod(t, t, p->z);
    mul256_mod(s2, q->y, t);
    
    // H = U2 - U1
    sub256_mod(h, u2, u1);
    
    // R = S2 - S1
    sub256_mod(r_val, s2, s1);
    
    // 檢查是否為同一點
    bool h_zero = true, r_zero = true;
    for (int i = 0; i < 8; ++i) {
        if (h[i] != 0) h_zero = false;
        if (r_val[i] != 0) r_zero = false;
    }
    
    if (h_zero) {
        if (r_zero) {
            // P == Q，使用倍點
            point_double(r, p);
        } else {
            // P == -Q
            point_set_infinity(r);
        }
        return;
    }
    
    // X3 = R^2 - H^3 - 2*U1*H^2
    unsigned int h2[8], h3[8], u1h2[8];
    mul256_mod(h2, h, h);           // H^2
    mul256_mod(h3, h2, h);          // H^3
    mul256_mod(u1h2, u1, h2);       // U1*H^2
    mul256_mod(r->x, r_val, r_val); // R^2
    sub256_mod(r->x, r->x, h3);     // R^2 - H^3
    sub256_mod(r->x, r->x, u1h2);   // R^2 - H^3 - U1*H^2
    sub256_mod(r->x, r->x, u1h2);   // R^2 - H^3 - 2*U1*H^2
    
    // Y3 = R*(U1*H^2 - X3) - S1*H^3
    sub256_mod(t, u1h2, r->x);      // U1*H^2 - X3
    mul256_mod(r->y, r_val, t);     // R*(U1*H^2 - X3)
    mul256_mod(t, s1, h3);          // S1*H^3
    sub256_mod(r->y, r->y, t);      // R*(U1*H^2 - X3) - S1*H^3
    
    // Z3 = H*Z1*Z2
    mul256_mod(t, p->z, q->z);      // Z1*Z2
    mul256_mod(r->z, h, t);         // H*Z1*Z2
}

// 標量乘法：R = k*G (使用 double-and-add)
__device__ void point_mul(Point* r, const unsigned int* k) {
    Point result, temp, g;
    
    // 初始化生成點 G（Affine -> Jacobian）
    for (int i = 0; i < 8; ++i) {
        g.x[i] = SECP256K1_GX[i];
        g.y[i] = SECP256K1_GY[i];
    }
    g.z[0] = 1;
    for (int i = 1; i < 8; ++i) g.z[i] = 0;
    
    point_set_infinity(&result);
    
    // 從最高位開始掃描
    for (int i = 255; i >= 0; --i) {
        int word = i / 32;
        int bit = i % 32;
        
        // 倍點
        point_double(&temp, &result);
        result = temp;
        
        // 如果該位為 1，則加上 G
        if (k[word] & (1U << bit)) {
            point_add(&temp, &result, &g);
            result = temp;
        }
    }
    
    *r = result;
}

// Jacobian 轉 Affine 座標
__device__ void point_to_affine(unsigned int* x, unsigned int* y, const Point* p) {
    if (point_is_infinity(p)) {
        for (int i = 0; i < 8; ++i) {
            x[i] = 0;
            y[i] = 0;
        }
        return;
    }
    // 快速路徑：若 Z == 1（[1,0,...,0]），則已是 Affine
    bool z_is_one = (p->z[0] == 1);
    for (int i = 1; i < 8; ++i) z_is_one = z_is_one && (p->z[i] == 0);
    if (z_is_one) {
        for (int i=0;i<8;++i){ x[i]=p->x[i]; y[i]=p->y[i]; }
        return;
    }
    
    unsigned int z_inv[8], z_inv2[8], z_inv3[8];
    
    // Z^(-1)
    inv256_mod(z_inv, p->z);
    
    // Z^(-2)
    mul256_mod(z_inv2, z_inv, z_inv);
    
    // Z^(-3)
    mul256_mod(z_inv3, z_inv2, z_inv);
    
    // X = X / Z^2
    mul256_mod(x, p->x, z_inv2);
    
    // Y = Y / Z^3
    mul256_mod(y, p->y, z_inv3);
}

// ===== 主 kernel =====

extern "C" __global__
void secp256k1_pubkey_batch(
    const unsigned char* privkeys,  // (N, 32) 私鑰
    unsigned char* pubkeys,          // (N, 65) 公鑰輸出
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // 指向該筆私鑰（按位由 MSB->LSB 掃描）
    const unsigned char* priv = privkeys + idx * 32;

    // 計算公鑰點 Q = k*G（左至右二進位法，MSB 首位）
    Point q;
    Point result, temp, g;
    for (int i = 0; i < 8; ++i) { g.x[i] = SECP256K1_GX[i]; g.y[i] = SECP256K1_GY[i]; }
    g.z[0] = 1; for (int i = 1; i < 8; ++i) g.z[i] = 0;
    point_set_infinity(&result);

    for (int i = 255; i >= 0; --i) {
        // 先倍點
        point_double(&temp, &result);
        result = temp;
        // 抽取第 i 位（以大端序掃描）：定位位元所在之 byte 與偏移
        int bj = (255 - i) >> 3;                  // 0..31（從高位字節到低位字節）
        int s  = 7 - ((255 - i) & 7);             // 該 byte 中的位（MSB=7..LSB=0）
        unsigned int bit = (unsigned int)((priv[bj] >> s) & 1U);
        if (bit) {
            point_add(&temp, &result, &g);
            result = temp;
        }
    }
    q = result;
    
    // 轉換為 Affine 座標
    unsigned int qx[8], qy[8];
    point_to_affine(qx, qy, &q);
    
    // 輸出未壓縮公鑰（0x04 + X + Y，大端序）
    unsigned char* pub = pubkeys + idx * 65;
    pub[0] = 0x04;
    
    // X 座標（大端序）
    for (int i = 0; i < 8; ++i) {
        int offset = 1 + (7 - i) * 4;
        pub[offset+0] = (qx[i] >> 24) & 0xFF;
        pub[offset+1] = (qx[i] >> 16) & 0xFF;
        pub[offset+2] = (qx[i] >> 8) & 0xFF;
        pub[offset+3] = qx[i] & 0xFF;
    }
    
    // Y 座標（大端序）
    for (int i = 0; i < 8; ++i) {
        int offset = 33 + (7 - i) * 4;
        pub[offset+0] = (qy[i] >> 24) & 0xFF;
        pub[offset+1] = (qy[i] >> 16) & 0xFF;
        pub[offset+2] = (qy[i] >> 8) & 0xFF;
        pub[offset+3] = qy[i] & 0xFF;
    }
}
"""

# 編譯 kernel
_secp256k1_module = cp.RawModule(code=_SECP256K1_KERNEL, options=("-std=c++11",))
_secp256k1_kernel = _secp256k1_module.get_function("secp256k1_pubkey_batch")

# 以 4-bit window 的標量乘法（需要預先提供 16 條 G 的倍點表，含 0）
_SECP256K1_KERNEL_W4 = r"""
__constant__ unsigned int SECP256K1_P[8] = {
    0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

// 比較、加減、乘、逆 與點運算，與主 kernel 一致的實作（節選複用）
__device__ int cmp256(const unsigned int* a, const unsigned int* b) {
    for (int i = 7; i >= 0; --i) { if (a[i] > b[i]) return 1; if (a[i] < b[i]) return -1; } return 0;
}
__device__ void add256_mod(unsigned int* r, const unsigned int* a, const unsigned int* b){ unsigned long long c=0; unsigned int t[8]; for(int i=0;i<8;++i){ c+=(unsigned long long)a[i]+b[i]; t[i]=(unsigned int)c; c>>=32;} if(c||cmp256(t,SECP256K1_P)>=0){ c=0; for(int i=0;i<8;++i){ unsigned long long s=(unsigned long long)t[i]-SECP256K1_P[i]-c; r[i]=(unsigned int)s; c=(s>>32)&1; } } else { for(int i=0;i<8;++i) r[i]=t[i]; } }
__device__ void sub256_mod(unsigned int* r, const unsigned int* a, const unsigned int* b){ unsigned long long br=0; unsigned int t[8]; for(int i=0;i<8;++i){ unsigned long long s=(unsigned long long)a[i]-b[i]-br; t[i]=(unsigned int)s; br=(s>>32)&1; } if(br){ unsigned long long c=0; for(int i=0;i<8;++i){ c+=(unsigned long long)t[i]+SECP256K1_P[i]; r[i]=(unsigned int)c; c>>=32; } } else { for(int i=0;i<8;++i) r[i]=t[i]; } }

// 乘法與約簡（與上方主 kernel相同的 reduce_p/mul256_mod 實現，為簡潔此處內嵌簡版）
__device__ void reduce_p(unsigned int* r, const unsigned int* T){ unsigned int X[20]; for(int i=0;i<20;++i) X[i]=0U; for(int i=0;i<16;++i) X[i]=T[i]; for(int k=15;k>=8;--k){ unsigned int u=X[k]; if(!u) continue; X[k]=0U; unsigned long long acc=(unsigned long long)X[k-8]+(unsigned long long)u*977ULL; X[k-8]=(unsigned int)acc; unsigned long long carry=acc>>32; int idx=k-7; while(carry){ unsigned long long a2=(unsigned long long)X[idx]+carry; X[idx]=(unsigned int)a2; carry=a2>>32; ++idx; } acc=(unsigned long long)X[k-7]+(unsigned long long)u; X[k-7]=(unsigned int)acc; carry=acc>>32; idx=k-6; while(carry){ unsigned long long a3=(unsigned long long)X[idx]+carry; X[idx]=(unsigned int)a3; carry=a3>>32; ++idx; } } int changed=1; while(changed){ changed=0; for(int k=19;k>=8;--k){ unsigned int u=X[k]; if(!u) continue; changed=1; X[k]=0U; unsigned long long acc=(unsigned long long)X[k-8]+(unsigned long long)u*977ULL; X[k-8]=(unsigned int)acc; unsigned long long carry=acc>>32; int idx=k-7; while(carry){ unsigned long long a2=(unsigned long long)X[idx]+carry; X[idx]=(unsigned int)a2; carry=a2>>32; ++idx; } acc=(unsigned long long)X[k-7]+(unsigned long long)u; X[k-7]=(unsigned int)acc; carry=acc>>32; idx=k-6; while(carry){ unsigned long long a3=(unsigned long long)X[idx]+carry; X[idx]=(unsigned int)a3; carry=a3>>32; ++idx; } } } for(int i=0;i<8;++i) r[i]=X[i]; while(cmp256(r,SECP256K1_P)>=0){ unsigned long long br=0ULL; for(int i=0;i<8;++i){ unsigned long long s=(unsigned long long)r[i]-SECP256K1_P[i]-br; r[i]=(unsigned int)s; br=(s>>32)&1ULL; } } }
__device__ void mul256_mod(unsigned int* r, const unsigned int* a, const unsigned int* b){ unsigned int T[16]; for(int i=0;i<16;++i) T[i]=0U; for(int i=0;i<8;++i){ unsigned long long c=0ULL; for(int j=0;j<8;++j){ unsigned long long acc=(unsigned long long)T[i+j]+(unsigned long long)a[i]*(unsigned long long)b[j]+c; T[i+j]=(unsigned int)acc; c=acc>>32; } unsigned long long acc2=(unsigned long long)T[i+8]+c; T[i+8]=(unsigned int)acc2; unsigned long long c2=acc2>>32; int k=i+9; while(c2 && k<16){ unsigned long long acc3=(unsigned long long)T[k]+c2; T[k]=(unsigned int)acc3; c2=acc3>>32; ++k; } } reduce_p(r,T); }

struct Point { unsigned int x[8]; unsigned int y[8]; unsigned int z[8]; };
__device__ void point_set_infinity(Point* p){ for(int i=0;i<8;++i){ p->x[i]=0; p->y[i]=0; p->z[i]=0; } }
__device__ bool point_is_infinity(const Point* p){ for(int i=0;i<8;++i){ if(p->z[i]!=0) return false; } return true; }
__device__ void point_double(Point* r, const Point* p){ if(point_is_infinity(p)){ point_set_infinity(r); return; } unsigned int s[8], m[8], t[8], u[8]; mul256_mod(t,p->y,p->y); mul256_mod(u,p->x,t); add256_mod(s,u,u); add256_mod(s,s,s); mul256_mod(t,p->x,p->x); add256_mod(m,t,t); add256_mod(m,m,t); mul256_mod(r->x,m,m); sub256_mod(r->x,r->x,s); sub256_mod(r->x,r->x,s); sub256_mod(t,s,r->x); mul256_mod(r->y,m,t); mul256_mod(t,p->y,p->y); mul256_mod(t,t,t); add256_mod(u,t,t); add256_mod(u,u,u); add256_mod(u,u,u); sub256_mod(r->y,r->y,u); mul256_mod(t,p->y,p->z); add256_mod(r->z,t,t); }
__device__ void point_add(Point* r, const Point* p, const Point* q){ if(point_is_infinity(p)){ *r=*q; return;} if(point_is_infinity(q)){ *r=*p; return;} unsigned int u1[8],u2[8],s1[8],s2[8],h[8],rv[8],t[8]; mul256_mod(t,q->z,q->z); mul256_mod(u1,p->x,t); mul256_mod(t,p->z,p->z); mul256_mod(u2,q->x,t); mul256_mod(t,q->z,q->z); mul256_mod(t,t,q->z); mul256_mod(s1,p->y,t); mul256_mod(t,p->z,p->z); mul256_mod(t,t,p->z); mul256_mod(s2,q->y,t); sub256_mod(h,u2,u1); sub256_mod(rv,s2,s1); bool h0=true,r0=true; for(int i=0;i<8;++i){ if(h[i]!=0) h0=false; if(rv[i]!=0) r0=false; } if(h0){ if(r0){ point_double(r,p);} else { point_set_infinity(r);} return;} unsigned int h2[8],h3[8],u1h2[8]; mul256_mod(h2,h,h); mul256_mod(h3,h2,h); mul256_mod(u1h2,u1,h2); mul256_mod(r->x,rv,rv); sub256_mod(r->x,r->x,h3); sub256_mod(r->x,r->x,u1h2); sub256_mod(r->x,r->x,u1h2); sub256_mod(t,u1h2,r->x); mul256_mod(r->y,rv,t); mul256_mod(t,s1,h3); sub256_mod(r->y,r->y,t); mul256_mod(t,p->z,q->z); mul256_mod(r->z,h,t); }
// 逆元：r = a^(p-2) mod p（費馬）
__device__ void inv256_mod(unsigned int* r, const unsigned int* a) {
    unsigned int exp[8] = {
        0xFFFFFC2D, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
    };
    unsigned int result[8] = {1,0,0,0,0,0,0,0};
    unsigned int base[8]; for(int i=0;i<8;++i) base[i]=a[i];
    for (int i=0;i<256;++i){
        int word=i/32, bit=i%32;
        if (exp[word] & (1U<<bit)){
            unsigned int tmp[8]; mul256_mod(tmp, result, base); for(int j=0;j<8;++j) result[j]=tmp[j];
        }
        if (i<255){ unsigned int tmp[8]; mul256_mod(tmp, base, base); for(int j=0;j<8;++j) base[j]=tmp[j]; }
    }
    for(int i=0;i<8;++i) r[i]=result[i];
}

__device__ void point_to_affine(unsigned int* x, unsigned int* y, const Point* p){
    bool z1=(p->z[0]==1);
    for(int i=1;i<8;++i) z1 = z1 && (p->z[i]==0);
    if(z1){ for(int i=0;i<8;++i){ x[i]=p->x[i]; y[i]=p->y[i]; } return; }
    unsigned int z_inv[8], z_inv2[8], z_inv3[8];
    inv256_mod(z_inv, p->z);
    mul256_mod(z_inv2, z_inv, z_inv);
    mul256_mod(z_inv3, z_inv2, z_inv);
    mul256_mod(x, p->x, z_inv2);
    mul256_mod(y, p->y, z_inv3);
}

extern "C" __global__
void secp256k1_pubkey_batch_w4(
    const unsigned char* privkeys,  // (N,32)
    const unsigned int* preX,       // (16,8) uint32 LE limbs
    const unsigned int* preY,       // (16,8)
    unsigned char* pubkeys,         // (N,65)
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

    Point R; point_set_infinity(&R);
    bool started = false;
    for (int w = 0; w < 64; ++w){
        if (started){
            Point tmp;
            point_double(&tmp,&R); R = tmp;
            point_double(&tmp,&R); R = tmp;
            point_double(&tmp,&R); R = tmp;
            point_double(&tmp,&R); R = tmp;
        }

        unsigned char byte = scalar_be[w >> 1];
        unsigned int nibble = (w & 1) ? (unsigned int)(byte & 0x0FU)
                                      : (unsigned int)(byte >> 4);
        if (!nibble) continue;

        Point T;
        for(int i=0;i<8;++i){ T.x[i]=preX[nibble*8 + i]; T.y[i]=preY[nibble*8 + i]; T.z[i]=0; }
        T.z[0]=1;
        if (!started){
            R = T;
            started = true;
            continue;
        }
        Point tmp; point_add(&tmp,&R,&T); R = tmp;
    }

    // 輸出未壓縮公鑰（0x04 + X + Y，大端序）
    unsigned int qx[8], qy[8];
    point_to_affine(qx, qy, &R);
    unsigned char* out = pubkeys + idx*65; out[0]=0x04;
    for (int i=0;i<8;++i){ int off=1+(7-i)*4; out[off+0]=(qx[i]>>24)&0xFF; out[off+1]=(qx[i]>>16)&0xFF; out[off+2]=(qx[i]>>8)&0xFF; out[off+3]=qx[i]&0xFF; }
    for (int i=0;i<8;++i){ int off=33+(7-i)*4; out[off+0]=(qy[i]>>24)&0xFF; out[off+1]=(qy[i]>>16)&0xFF; out[off+2]=(qy[i]>>8)&0xFF; out[off+3]=qy[i]&0xFF; }
}
"""

_secp256k1_module_w4 = cp.RawModule(code=_SECP256K1_KERNEL_W4, options=("-std=c++11",))
_secp256k1_kernel_w4 = _secp256k1_module_w4.get_function("secp256k1_pubkey_batch_w4")


def gpu_secp256k1_batch(privkeys_gpu: "cp.ndarray") -> "cp.ndarray":
    """
    在 GPU 上批量計算 secp256k1 公鑰（未壓縮 65 bytes）。
    
    參數：
    - privkeys_gpu: uint8 形狀為 (N, 32) 的 GPU 陣列
    
    回傳：
    - uint8 形狀為 (N, 65) 的 GPU 陣列（未壓縮公鑰：0x04 + X(32) + Y(32)）
    """
    if privkeys_gpu.dtype != cp.uint8 or privkeys_gpu.ndim != 2 or privkeys_gpu.shape[1] != 32:
        raise ValueError("privkeys_gpu 需為 uint8 (N,32)")
    
    n = privkeys_gpu.shape[0]
    pubkeys_gpu = cp.zeros((n, 65), dtype=cp.uint8)
    
    threads = 256
    blocks = (n + threads - 1) // threads
    
    _secp256k1_kernel(
        (blocks,), (threads,),
        (privkeys_gpu, pubkeys_gpu, cp.int32(n))
    )
    
    return pubkeys_gpu


def _build_precomp_table_w4_cpu():
    """建立 4-bit 視窗預計算表（CPU 端），結果快取供後續 GPU 重用。"""
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

    _PRECOMP_W4_CPU = (xs, ys)
    return _PRECOMP_W4_CPU


def _ensure_precomp_table_w4() -> Tuple[cp.ndarray, cp.ndarray]:
    """回傳目前 device 的 Window4 預計算表（GPU 端），必要時自動建立。"""
    dev_id = int(cp.cuda.Device())
    with _W4_CACHE_LOCK:
        cached = _PRECOMP_W4_GPU.get(dev_id)
        if cached is not None:
            return cached

        cpu_xs, cpu_ys = _build_precomp_table_w4_cpu()
        pre_x = cp.asarray(cpu_xs, dtype=cp.uint32)
        pre_y = cp.asarray(cpu_ys, dtype=cp.uint32)
        _PRECOMP_W4_GPU[dev_id] = (pre_x, pre_y)
        return pre_x, pre_y


def warmup_window4_table(force: bool = False) -> None:
    """預先建立或重新建立目前 device 的 Window4 預計算表。"""
    dev_id = int(cp.cuda.Device())
    if force:
        with _W4_CACHE_LOCK:
            _PRECOMP_W4_GPU.pop(dev_id, None)
    _ensure_precomp_table_w4()


def gpu_secp256k1_batch_window4(privkeys_gpu: "cp.ndarray") -> "cp.ndarray":
    """實驗性：4-bit 視窗的 GPU 標量乘法。
    已輸出未壓縮公鑰；與主內核一致。若發生錯誤會回退至主內核實作。
    """
    if privkeys_gpu.dtype != cp.uint8 or privkeys_gpu.ndim != 2 or privkeys_gpu.shape[1] != 32:
        raise ValueError("privkeys_gpu 需為 uint8 (N,32)")
    n = privkeys_gpu.shape[0]
    pub = cp.zeros((n,65), dtype=cp.uint8)
    try:
        preX, preY = _ensure_precomp_table_w4()
        threads = 256; blocks = (n + threads - 1)//threads
        _secp256k1_kernel_w4((blocks,), (threads,), (privkeys_gpu, preX, preY, pub, cp.int32(n)))
        return pub
    except Exception:
        return gpu_secp256k1_batch(privkeys_gpu)


__all__ = ["gpu_secp256k1_batch", "gpu_secp256k1_batch_window4", "warmup_window4_table"]
