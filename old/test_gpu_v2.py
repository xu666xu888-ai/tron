# -*- coding: utf-8 -*-
"""
GPU secp256k1 V2（PTX 內聯彙編）快速測試
- 預設僅測試私鑰=1（目前 V2 僅針對該用例驗證）
- 可選 --random N 進行隨機測試（多數將失敗，因 V2 尚未完成一般 k*G 實作）

使用方式：
  python3 scripts/test_gpu_v2.py              # 測試 k=1
  python3 scripts/test_gpu_v2.py --random 8   # 額外嘗試 N 筆隨機（預期多數失敗）
"""
from __future__ import annotations

import argparse
import secrets
import cupy as cp
import coincurve

from tron_vanity.gpu_secp256k1_v2 import gpu_secp256k1_batch_v2


def test_k_eq_1() -> bool:
    test_priv = bytes.fromhex('00' * 31 + '01')
    priv_gpu = cp.array([list(test_priv)], dtype=cp.uint8)
    print('測試 GPU secp256k1 V2 (PTX) 私鑰=1…')
    pub_gpu = gpu_secp256k1_batch_v2(priv_gpu)
    pub_result = bytes(cp.asnumpy(pub_gpu[0]))

    pk = coincurve.PrivateKey(test_priv)
    pub_expected = pk.public_key.format(compressed=False)

    print('GPU 公鑰:', pub_result.hex())
    print('CPU 公鑰:', pub_expected.hex())
    ok = (pub_result == pub_expected)
    print('✅ 驗證通過！' if ok else '❌ 驗證失敗！')
    if not ok:
        print('差異:')
        print('  預期 Gx:', pub_expected[1:33].hex())
        print('  實際 Gx:', pub_result[1:33].hex())
        print('  預期 Gy:', pub_expected[33:65].hex())
        print('  實際 Gy:', pub_result[33:65].hex())
    return ok


def test_random(n: int) -> None:
    print(f'隨機測試 {n} 筆（V2 尚未完成一般 k*G，預期多數失敗）…')
    privs = [secrets.token_bytes(32) for _ in range(n)]
    priv_gpu = cp.array([list(p) for p in privs], dtype=cp.uint8)
    pubs_gpu = gpu_secp256k1_batch_v2(priv_gpu)
    pubs_gpu = cp.asnumpy(pubs_gpu)
    fail = 0
    for i, pk in enumerate(privs):
        cpu = coincurve.PrivateKey(pk).public_key.format(compressed=False)
        gpu = bytes(pubs_gpu[i])
        if cpu != gpu:
            fail += 1
    print(f'隨機用例不一致: {fail}/{n}')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--random', type=int, default=0, help='額外隨機測試數量（可選）')
    args = ap.parse_args()

    ok = test_k_eq_1()
    if args.random > 0:
        test_random(args.random)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())

