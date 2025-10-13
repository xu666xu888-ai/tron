# -*- coding: utf-8 -*-
"""
調試 GPU secp256k1 實現
逐步驗證每個組件
"""
import cupy as cp
import coincurve

# 測試私鑰 = 1，應該得到生成點 G
test_priv = bytes.fromhex('0000000000000000000000000000000000000000000000000000000000000001')

# 預期結果（生成點 G）
expected_gx = bytes.fromhex('79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798')
expected_gy = bytes.fromhex('483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8')

print("預期生成點 G:")
print(f"Gx = {expected_gx.hex()}")
print(f"Gy = {expected_gy.hex()}")
print()

# 用 coincurve 驗證
pk = coincurve.PrivateKey(test_priv)
pub = pk.public_key.format(compressed=False)
print("coincurve 計算結果:")
print(f"公鑰 = {pub.hex()}")
print(f"Gx = {pub[1:33].hex()}")
print(f"Gy = {pub[33:65].hex()}")
print()

# 檢查字節序
print("字節序檢查:")
print(f"Gx 大端序: {expected_gx.hex()}")
print(f"Gx 小端序: {expected_gx[::-1].hex()}")
print()

# 測試我們的 GPU 實現
from tron_vanity.gpu_secp256k1 import gpu_secp256k1_batch

priv_gpu = cp.array([list(test_priv)], dtype=cp.uint8)
pub_gpu = gpu_secp256k1_batch(priv_gpu)
pub_result = bytes(cp.asnumpy(pub_gpu[0]))

print("GPU 計算結果:")
print(f"公鑰 = {pub_result.hex()}")
print(f"Gx = {pub_result[1:33].hex()}")
print(f"Gy = {pub_result[33:65].hex()}")
print()

# 比較
if pub_result == pub:
    print("✅ 完全匹配！")
else:
    print("❌ 不匹配")
    print()
    print("詳細比較:")
    print(f"預期 Gx: {pub[1:33].hex()}")
    print(f"實際 Gx: {pub_result[1:33].hex()}")
    print(f"預期 Gy: {pub[33:65].hex()}")
    print(f"實際 Gy: {pub_result[33:65].hex()}")