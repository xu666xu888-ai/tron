# -*- coding: utf-8 -*-
"""
GPU 模運算單元測試（針對 secp256k1 素數模）
- 驗證 GPU 端 mul256_mod/reduce_p 的正確性（隨機樣本）

執行：
  PYTHONPATH=src python -m tron_vanity.test_mod_arith_gpu --n 64
"""
from __future__ import annotations

import argparse
import secrets
from typing import Tuple

import cupy as cp

P_HEX = int("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F", 16)


KERNEL = r"""
__constant__ unsigned int SECP256K1_P[8] = {
    0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

__device__ int cmp256(const unsigned int* a, const unsigned int* b) {
    for (int i = 7; i >= 0; --i) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}

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

__device__ void mul_const_977_9(unsigned int* out9, const unsigned int* H) {
    unsigned long long carry = 0ULL;
    for (int i = 0; i < 8; ++i) {
        unsigned long long acc = (unsigned long long)H[i] * 977ULL + carry;
        out9[i] = (unsigned int)acc;
        carry = acc >> 32;
    }
    out9[8] = (unsigned int)carry;
}

__device__ void reduce_p(unsigned int* r, const unsigned int* T) {
    unsigned int X[20];
    for (int i=0;i<20;++i) X[i]=0U;
    for (int i=0;i<16;++i) X[i] = T[i];
    for (int k = 15; k >= 8; --k) {
        unsigned int u = X[k];
        if (!u) continue;
        X[k] = 0U;
        unsigned long long acc = (unsigned long long)X[k-8] + (unsigned long long)u * 977ULL;
        X[k-8] = (unsigned int)acc;
        unsigned long long carry = acc >> 32;
        int idx = k-7;
        while (carry) { unsigned long long a2 = (unsigned long long)X[idx] + carry; X[idx] = (unsigned int)a2; carry = a2 >> 32; ++idx; }
        acc = (unsigned long long)X[k-7] + (unsigned long long)u;
        X[k-7] = (unsigned int)acc;
        carry = acc >> 32; idx = k-6;
        while (carry) { unsigned long long a3 = (unsigned long long)X[idx] + carry; X[idx] = (unsigned int)a3; carry = a3 >> 32; ++idx; }
    }
    int changed = 1;
    while (changed) {
        changed = 0;
        for (int k = 19; k >= 8; --k) {
            unsigned int u = X[k];
            if (!u) continue;
            changed = 1; X[k]=0U;
            unsigned long long acc = (unsigned long long)X[k-8] + (unsigned long long)u * 977ULL;
            X[k-8] = (unsigned int)acc;
            unsigned long long carry = acc >> 32;
            int idx = k-7;
            while (carry) { unsigned long long a2 = (unsigned long long)X[idx] + carry; X[idx] = (unsigned int)a2; carry = a2 >> 32; ++idx; }
            acc = (unsigned long long)X[k-7] + (unsigned long long)u;
            X[k-7] = (unsigned int)acc;
            carry = acc >> 32; idx = k-6;
            while (carry) { unsigned long long a3 = (unsigned long long)X[idx] + carry; X[idx] = (unsigned int)a3; carry = a3 >> 32; ++idx; }
        }
    }
    for (int i=0;i<8;++i) r[i] = X[i];
    while (cmp256(r, SECP256K1_P) >= 0) {
        unsigned long long br=0ULL; for (int i=0;i<8;++i){ unsigned long long sub=(unsigned long long)r[i]-SECP256K1_P[i]-br; r[i]=(unsigned int)sub; br=(sub>>32)&1ULL; }
    }
}

__device__ void mul256_mod(unsigned int* r, const unsigned int* a, const unsigned int* b) {
    unsigned int T[16];
    for (int i=0;i<16;++i) T[i]=0U;
    for (int i = 0; i < 8; ++i) {
        unsigned long long carry = 0ULL;
        for (int j = 0; j < 8; ++j) {
            unsigned long long acc = (unsigned long long)T[i+j] + (unsigned long long)a[i]*(unsigned long long)b[j] + carry;
            T[i+j] = (unsigned int)acc;
            carry = acc >> 32;
        }
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

extern "C" __global__
void mul_test(const unsigned int* A, const unsigned int* B, unsigned int* R, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const unsigned int* a = A + idx*8;
    const unsigned int* b = B + idx*8;
    unsigned int* r = R + idx*8;
    mul256_mod(r, a, b);
}

extern "C" __global__
void reduce_test(const unsigned int* T, unsigned int* R, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const unsigned int* t = T + idx*16;
    unsigned int* r = R + idx*8;
    reduce_p(r, t);
}
"""


def le_bytes_to_limbs_le32(x: int) -> bytes:
    b = x.to_bytes(32, 'big')  # big-endian bytes
    # Convert to 8 little-endian 32-bit limbs as bytes in little-endian order
    limbs = []
    for i in range(8):
        limb = int.from_bytes(b[28-4*i:32-4*i], 'big')
        limbs.append(limb)
    return b"".join(l.to_bytes(4, 'little') for l in limbs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=32)
    args = ap.parse_args()

    mod = cp.RawModule(code=KERNEL, options=("-std=c++11",))
    ker = mod.get_function('mul_test')
    rker = mod.get_function('reduce_test')

    A = []
    B = []
    gold = []
    for _ in range(args.n):
        a = secrets.randbelow(P_HEX)
        b = secrets.randbelow(P_HEX)
        r = (a*b) % P_HEX
        A.append(le_bytes_to_limbs_le32(a))
        B.append(le_bytes_to_limbs_le32(b))
        gold.append(r.to_bytes(32, 'big'))

    A_gpu = cp.frombuffer(b"".join(A), dtype=cp.uint32).reshape(args.n, 8)
    B_gpu = cp.frombuffer(b"".join(B), dtype=cp.uint32).reshape(args.n, 8)
    R_gpu = cp.zeros((args.n, 8), dtype=cp.uint32)

    threads = 256
    blocks = (args.n + threads - 1)//threads
    ker((blocks,), (threads,), (A_gpu, B_gpu, R_gpu, cp.int32(args.n)))

    R = bytes(cp.asnumpy(R_gpu).astype('<u4').tobytes())

    ok = True
    for i in range(args.n):
        got_le = R[i*32:(i+1)*32]
        got_be = got_le[::-1]
        if got_be != gold[i]:
            print(f"[X] mul mod mismatch at {i}")
            # 輸出 a,b 及結果便於排查
            a_be = int.from_bytes(A[i], 'big')
            b_be = int.from_bytes(B[i], 'big')
            print('a =', hex(a_be))
            print('b =', hex(b_be))
            print('cpu =', gold[i].hex())
            print('gpu =', got_be.hex())
            ok = False
            break
    if ok:
        print(f"[OK] GPU mul256_mod 一致（n={args.n}）")
        return 0
    # 若失敗，進一步測試 reduce_p 對同一樣本
    # 構造單筆 T=a*b 的 16 個 limb（小端 32-bit）
    a = int.from_bytes(A[0], 'big')
    b = int.from_bytes(B[0], 'big')
    t = a*b
    t_le = t.to_bytes(64, 'little')
    T_gpu = cp.frombuffer(t_le, dtype=cp.uint32).reshape(1,16)
    R2_gpu = cp.zeros((1,8), dtype=cp.uint32)
    rker((1,), (1,), (T_gpu, R2_gpu, cp.int32(1)))
    r2_le = bytes(cp.asnumpy(R2_gpu).astype('<u4').tobytes())
    r2_be = r2_le[::-1]
    print('[reduce_test] cpu=', (t % P_HEX).to_bytes(32,'big').hex())
    print('[reduce_test] gpu=', r2_be.hex())
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
