# 舊文件歸檔目錄

本目錄包含開發過程中產生的測試文件和實驗性代碼，這些文件對於運行完整 CUDA 方案並非必需，但保留作為參考。

## 📁 文件說明

### 測試文件（6 個）

1. **test_base58_gpu_vs_cpu.py**
   - Base58Check GPU vs CPU 一致性測試
   - 驗證 GPU Base58 編碼的正確性

2. **test_ecc_w4_vs_ref.py**
   - Window4 ECC 優化測試
   - 比對 Window4 與參考實現的一致性

3. **test_gpu_vs_cpu.py**
   - GPU vs CPU 公鑰生成一致性測試
   - 驗證 GPU secp256k1 的正確性

4. **test_keccak_gpu_vs_cpu.py**
   - Keccak-256 GPU vs CPU 一致性測試
   - 驗證 GPU Keccak-256 的正確性

5. **test_mod_arith_gpu.py**
   - GPU 模運算測試
   - 驗證 256-bit 大數模運算的正確性

6. **test_secp256k1_debug.py**
   - secp256k1 調試工具
   - 用於調試橢圓曲線運算

### 實驗性文件（2 個）

7. **gpu_secp256k1_v2.py**
   - Window4 優化的實驗版本
   - 尚未完成驗證，保留作為參考

8. **test_gpu_v2.py**
   - GPU 測試腳本
   - 已被 `scripts/benchmark.py` 取代

### 演示文件（1 個）

9. **v1_demo.py**
   - V1 演示模式
   - 單地址生成與驗證
   - 已被 `v2_vanity.py` 取代

## 🔄 如何使用這些文件

如果需要運行測試或參考實驗性代碼，可以將文件複製回原位置：

```bash
# 複製測試文件回 src/tron_vanity/
cp old/test_*.py src/tron_vanity/

# 複製實驗性文件
cp old/gpu_secp256k1_v2.py src/tron_vanity/

# 複製演示文件
cp old/v1_demo.py src/tron_vanity/
```

## ⚠️ 注意事項

- 這些文件僅供參考和測試使用
- 不影響生產環境的完整 CUDA 方案運行
- 如需刪除，可直接刪除整個 `old/` 目錄

## 📊 核心運行文件

完整 CUDA 方案僅需以下核心文件：

```
requirements.txt
.gitignore
scripts/check_env.py
scripts/benchmark.py
src/tron_vanity/__init__.py
src/tron_vanity/addr.py
src/tron_vanity/gpu_random.py
src/tron_vanity/gpu_secp256k1.py
src/tron_vanity/gpu_keccak.py
src/tron_vanity/gpu_addr.py
src/tron_vanity/v2_vanity.py
src/tron_vanity/validate.py
README.md
```

---

最後更新：2025-10-13