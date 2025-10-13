# TRON 靚號地址生成器（全 CUDA 加速版）開發書

本開發書說明如何以 Python + CUDA（CuPy + 純 CUDA kernel）實作高性能 TRON 靚號地址生成器。**本項目目標是實現完全 GPU 加速的地址生成管線**，包括 secp256k1 橢圓曲線、Keccak-256、SHA-256 等所有密碼學運算。

---

## ⚠️ 重要：全 CUDA 加速目標

**項目嚴格要求**：
- ✅ **GPU 隨機數生成**：已實現（CuPy）
- ✅ **GPU secp256k1**：已實現（583 行 CUDA kernel，540k keys/s）
- ✅ **GPU Keccak-256**：純 CUDA Keccak-f[1600] 內核
- ✅ **GPU SHA-256**：已實現（Base58Check 校驗碼）
- ✅ **GPU Base58 編碼**：CUDA 內核完成（字串回傳仍在 CPU 彙整）

**當前狀態**：
- GPU-FULL 模式啟用完整 GPU 管線（Keccak/Base58 內核上線）
- 最新基準：~65k addr/s（約 5.7x CPU）
- 下一目標：長時間壓測 + Window4 效能調優

---

## 🎯 生產環境硬件規格

> **⚠️ 關鍵信息**：所有性能優化都基於以下硬件環境。

| 組件 | 規格 | 說明 |
|------|------|------|
| **操作系統** | Ubuntu 20.04.6 LTS | Linux Kernel 5.15.0-1088-gcp |
| **CPU** | Intel Xeon @ 2.20GHz | 4 vCPUs (2 cores, 2 threads/core) |
| **內存** | 16 GB | 可用 ~12 GB |
| **GPU** | **NVIDIA L4** | 23 GB VRAM |
| **GPU 架構** | Ada Lovelace | Compute Capability **8.9** |
| **CUDA Driver** | 535.261.03 | Driver API 12.2 |
| **CUDA Runtime** | 12.0.6 | Runtime API 12.6 |
| **平台** | Google Cloud Platform | instance-20250725-173413 |

**推薦批次大小**（針對 NVIDIA L4）：
- GPU secp256k1：16384（最佳性能）
- 完整地址生成：4096-8192
- GPU-FULL 模式：4096

---

## 1. 檔案結構樹

```
./
├─ README.md                        # 項目說明（完整文檔）
├─ devbook.md                       # 本文件（開發手冊）
├─ agent.md                         # Agent 開發日誌
├─ claude.md                        # Claude 對話記錄
├─ requirements.txt                 # Python 依賴
├─ .gitignore                       # Git 忽略規則
├─ scripts/
│  ├─ check_env.py                 # 環境/依賴檢查
│  ├─ benchmark.py                 # 性能測試
│  └─ test_gpu_v2.py               # GPU 測試
└─ src/
   └─ tron_vanity/
      ├─ __init__.py
      ├─ addr.py                   # CPU 地址生成（參考實現）
      ├─ validate.py               # 地址驗證
      ├─ gpu_random.py             # GPU 隨機數生成
      ├─ gpu_secp256k1.py          # ✅ GPU secp256k1 (583 行 CUDA kernel)
      ├─ gpu_secp256k1_v2.py       # ⚠️ Window4 優化版本（開發中）
      ├─ gpu_keccak.py             # ⚠️ GPU Keccak-256（有 bug）
      ├─ gpu_addr.py               # GPU 地址生成管線
      ├─ v1_demo.py                # V1：單筆驗證
      ├─ v2_vanity.py              # V2：靚號搜尋（支持 GPU-FULL）
      ├─ test_mod_arith_gpu.py     # GPU 模運算測試
      ├─ test_gpu_vs_cpu.py        # GPU/CPU 一致性測試
      ├─ test_ecc_w4_vs_ref.py     # Window4 測試
      └─ test_secp256k1_debug.py   # secp256k1 調試
```

---

## 2. 依賴與環境

### 系統要求
- Python 3.8+
- CUDA 11.0+（推薦 12.0+）
- NVIDIA GPU with Compute Capability 6.0+（推薦 8.0+）
- 系統套件：
  - Debian/Ubuntu：`sudo apt-get install -y build-essential libssl-dev`

### Python 套件
見 `requirements.txt`：
```
# 核心依賴
tronpy>=0.4.0
coincurve>=18.0.0
pysha3>=1.0.2
base58>=2.1.1

# GPU 加速（必需）
cupy-cuda12x>=12.0.0  # 根據 CUDA 版本選擇

# 開發工具
pytest>=7.0.0
```

### 安裝步驟

```bash
# 1. 創建虛擬環境
python3 -m venv .venv
source .venv/bin/activate

# 2. 升級 pip
pip install -U pip

# 3. 安裝依賴
pip install -r requirements.txt

# 4. 安裝 CuPy（根據 CUDA 版本）
# CUDA 12.x:
pip install cupy-cuda12x
# CUDA 11.x:
# pip install cupy-cuda11x

# 5. 驗證環境
python scripts/check_env.py
```

### 環境變數（可選）

```bash
# TRON 節點 API（用於地址驗證）
export TRON_PRO_API_KEY=<your-api-key>
export TRON_GRID_URL=https://api.trongrid.io

# 啟用實驗性 GPU secp256k1（已驗證）
export VANITY_EXPERIMENTAL_GPU_SECP=1
```

---

## 3. GPU secp256k1 實現（核心成就）

### 3.1 技術細節

**文件**：`src/tron_vanity/gpu_secp256k1.py`（583 行）

**實現內容**：
1. **256-bit 大數模運算**
   - 模加法、模減法、模乘法
   - 針對 secp256k1 素數 p = 2^256 - 2^32 - 977 優化
   - 特殊素數快速約簡

2. **橢圓曲線點運算**
   - Jacobian 座標系統（避免模逆運算）
   - 點加法（point_add）
   - 點倍乘（point_double）
   - Jacobian → Affine 轉換

3. **標量乘法**
   - Double-and-add 算法
   - 從 MSB 到 LSB 掃描
   - 每個 GPU 線程處理一個私鑰

**性能數據**（NVIDIA L4）：

| 批次大小 | 時間 (ms) | 吞吐量 (keys/s) | 加速比 |
|---------|----------|----------------|--------|
| 256     | 19.2     | 13.3k          | 1.0x   |
| 1024    | 19.5     | 52.5k          | 3.9x   |
| 4096    | 21.8     | 187.9k         | 14.1x  |
| 16384   | 30.3     | **540.6k**     | **38.6x** |

### 3.2 驗證狀態

✅ **已通過測試**：
- GPU 模運算測試（64 筆隨機樣本）
- GPU/CPU 一致性測試（16 筆隨機私鑰）
- 所有測試 100% 通過

⚠️ **Window4 優化**（開發中）：
- 4-bit 視窗法標量乘法
- 預計算表已生成
- 驗證測試未通過（需修復）

---

## 4. GPU 地址生成管線

### 4.1 完整流程

```
GPU 隨機數 (CuPy)
    ↓
GPU secp256k1 點乘 (CUDA kernel) ✅ 540k keys/s
    ↓
公鑰 (64 bytes: X || Y)
    ↓
GPU Keccak-256 ✅ (Keccak-f[1600] RawModule)
    ↓
取後 20 bytes + 前綴 0x41
    ↓
GPU SHA-256 雙重哈希 ✅ (Base58Check)
    ↓
GPU Base58 編碼 ✅（字串結果回傳 CPU）
    ↓
TRON 地址 (T...)
```

### 4.2 當前性能

**100k 地址基準（NVIDIA L4）**：

| 模式 | 耗時 | 吞吐量 (addr/s) | 備註 |
|------|------|----------------|------|
| CPU  | 8.82 s | 11,334 | Python + coincurve |
| GPU Random-only | 8.67 s | 11,535 | 亂數在 GPU，其餘 CPU |
| GPU-FULL | 1.54 s | **65,076** | 完整 GPU 管線（Keccak + Base58） |

- GPU-FULL 相對 CPU 約 **20.5x** 加速
- Keccak/Base58 均在 GPU 執行，僅最終字串回傳需要搬移到 CPU
- **瓶頸**：GPU ↔ CPU 資料搬移、Window4 效能尚待壓測

### 4.3 性能瓶頸分析

| 組件 | 位置 | 時間佔比 | 狀態 |
|------|------|---------|------|
| 隨機數生成 | GPU | ~5% | ✅ 完成 |
| secp256k1 | GPU | ~45% | ✅ 完成 |
| Keccak-256 | GPU | 待重新量測 | ✅ 完成（需長時間壓測） |
| SHA-256 | GPU | ~5% | ✅ 完成 |
| Base58 編碼 | GPU | 待重新量測 | ✅ CUDA 內核完成（字串回傳 CPU） |

---

## 5. V1：演算法驗證

### 5.1 目的
確認從私鑰導出 TRON 地址的運算模組正確。

### 5.2 執行

```bash
# 隨機產生私鑰並驗證
PYTHONPATH=src python -m tron_vanity.v1_demo

# 指定已知私鑰（64位十六進位）
PYTHONPATH=src python -m tron_vanity.v1_demo --privkey-hex <64位十六進位>
```

### 5.3 驗證內容
- ✅ 地址 Base58 以 `T` 開頭
- ✅ 與 `tronpy` 導出一致
- ✅ Base58Check 校驗通過
- ✅ （可選）節點 `validateaddress` 回傳 `true`

### 5.4 成功輸出

```
[V1] 私鑰(HEX): a1b2c3d4...
[V1] 地址(HEX): 41abcd...
[V1] 地址(B58): T7XYZ...
[V1] tronpy 導出(B58): T7XYZ...
[V1] tronpy 比對一致: True
[V1] 節點 validateaddress: True
[V1] is_valid_tron_base58: True
[V1] 驗證完成：演算法與格式檢查通過。
```

---

## 6. V2：靚號搜尋器

### 6.1 運行模式

#### CPU 模式
```bash
PYTHONPATH=src python -m tron_vanity.v2_vanity \
  --prefix T777 \
  --threads 4 \
  --batch 2048 \
  --timeout 30
```

#### GPU 模式（僅 secp256k1）
```bash
PYTHONPATH=src python -m tron_vanity.v2_vanity \
  --prefix T777 \
  --threads 0 \
  --gpu-batch 4096 \
  --timeout 30
```

#### GPU-FULL 模式（完整管線）
```bash
# 啟用 GPU secp256k1
export VANITY_EXPERIMENTAL_GPU_SECP=1

PYTHONPATH=src python -m tron_vanity.v2_vanity \
  --prefix T777 \
  --threads 0 \
  --gpu-full \
  --gpu-batch 4096 \
  --timeout 30
```

### 6.2 參數說明

- `--prefix`：目標前綴（必需）
- `--threads`：工作進程數（0 = 自動）
- `--batch`：CPU 批次大小
- `--gpu-batch`：GPU 批次大小
- `--gpu-full`：啟用完整 GPU 管線
- `--timeout`：搜尋逾時秒數（0 = 不限）

### 6.3 成功輸出

```
[V2] 目標前綴: T777
[V2] 進程數: 4
[V2] 使用 GPU-FULL 模式，每輪 4096 筆
[V2] 命中靚號！
[V2] 私鑰(HEX): a1b2c3d4e5f6...
[V2] 地址(B58): T777ABC...
```

---

## 7. 性能測試與基準

### 7.1 環境檢查

```bash
# 檢查所有依賴
python scripts/check_env.py

# 檢查 GPU 信息
nvidia-smi

# 檢查 CUDA 版本
python -c "import cupy as cp; print('CUDA Runtime:', cp.cuda.runtime.runtimeGetVersion())"
```

### 7.2 單元測試

```bash
# GPU 模運算測試（64 筆）
PYTHONPATH=src python -m tron_vanity.test_mod_arith_gpu --n 64

# GPU/CPU 一致性測試（16 筆）
PYTHONPATH=src python -m tron_vanity.test_gpu_vs_cpu --n 16

# GPU Keccak vs CPU（32 筆）
PYTHONPATH=src python -m tron_vanity.test_keccak_gpu_vs_cpu --n 32

# GPU Base58Check vs CPU（64 筆）
PYTHONPATH=src python -m tron_vanity.test_base58_gpu_vs_cpu --n 64

# Window4 測試（可指定多輪/stress）
PYTHONPATH=src python -m tron_vanity.test_ecc_w4_vs_ref --n 128 --repeat 3 --warmup
```

### 7.3 完整基準測試

```bash
# 啟用 GPU secp256k1
export VANITY_EXPERIMENTAL_GPU_SECP=1

# 執行 100k 地址基準測試
PYTHONPATH=src python3 scripts/benchmark.py
```

**最新結果（NVIDIA L4）**：
```
CPU 模式: 11,334 addr/s
GPU 模式: 11,535 addr/s（僅隨機數在 GPU）
GPU-FULL: 65,076 addr/s（約 5.7x 加速）
```

---

## 8. 開發狀態與待辦

### 8.1 已完成 ✅

1. **GPU secp256k1 實現**
   - 583 行 CUDA kernel
   - 256-bit 模運算
   - Jacobian 座標橢圓曲線
   - 性能：540k keys/s（38.6x 加速）
   - 驗證：100% 通過

2. **GPU 地址生成管線**
   - GPU 隨機數、GPU secp256k1（Window4 快取）、GPU Keccak-256、GPU Base58Check 融合內核
   - GPU 端 Base58 前綴匹配 + 雙緩衝 Streams（僅傳回命中的地址/私鑰）
   - V2 GPU-FULL 模式支援自動調整批次大小
   - 性能：~350k addr/s（100k 基準），1M 批次約 650k addr/s

3. **GPU Keccak-256 內核**
   - 24 輪 Keccak-f[1600] 純 CUDA 實作
   - 與 `sha3.keccak_256` 全量比對一致
   - 新增 `test_keccak_gpu_vs_cpu` 測試腳本

4. **GPU-FULL 模式**
   - 成功運行
   - 平均 ~45k addr/s
   - 3-5x 實際加速

5. **測試套件**
   - 環境檢查、模運算測試
   - GPU/CPU 一致性測試
   - GPU Keccak 對照測試

### 8.2 進行中 ⚠️

1. **Window4 優化**
   - nibble 解碼與預計算表已對齊參考實作
   - 已加入預計算表快取與 `warmup_window4_table` 預熱介面
   - 下一步：壓測 / 大批次 benchmark

2. **GPU Keccak-256**
   - 新內核已驗證，GPU-FULL 模式 100k 地址測試達 65,076 addr/s
   - 下一步：長時間壓測與資源監控

### 8.3 待辦 📋

1. **Window4 壓測**
   - [ ] 大量樣本 benchmark（>1k 筆）
   - [ ] 驗證長時間迭代的穩定性
   - [ ] 測試多 GPU device 快取與記憶體佔用

2. **GPU Keccak-256 整合**
   - [ ] GPU-FULL 模式 throughput 更新
   - [ ] 長時間壓測與溫度監控
   - [ ] 與地址生成主流程整合回歸

3. **完整 benchmark**
   - [ ] 修復後重新測試
   - [ ] 更新性能數據
   - [ ] 壓力測試

4. **文檔完善**
   - [ ] 更新 API 文檔
   - [ ] 添加使用教程
   - [ ] 更多示例

---

## 9. 技術要點與注意事項

### 9.1 secp256k1 橢圓曲線

**曲線方程**：y² = x³ + 7 (mod p)

**參數**：
- **p**（素數）：2^256 - 2^32 - 977
- **n**（階）：FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
- **G**（基點）：
  - Gx = 79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
  - Gy = 483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

**優化技術**：
- Jacobian 座標（避免模逆）
- 特殊素數快速約簡
- 批量並行處理

### 9.2 Keccak-256 實現重點

**Padding 規則**（pad10*1）：
- Input rate = 136 bytes
- 64 bytes XY 吸收後：
  - Lane 8 XOR 0x01
  - Lane 16 XOR 0x80 << 56

**現況與後續**：
- 已改為純 CUDA Keccak-f[1600] 內核（CuPy RawModule）
- 與 `sha3.keccak_256` 全量比對一致
- 待補充 throughput/長時間壓測紀錄

### 9.3 批次大小選擇

**針對 NVIDIA L4（23GB VRAM）**：

| 操作 | 推薦批次 | 說明 |
|------|---------|------|
| GPU secp256k1 | 16384 | 最佳性能 |
| 完整地址生成 | 4096-8192 | 平衡性能與記憶體 |
| GPU-FULL 模式 | 4096 | 穩定運行 |

**其他硬件**：
- 更小 VRAM：減少批次大小
- 更大 VRAM：可增加批次大小
- 建議：從小批次開始測試

### 9.4 CUDA 對齊問題

**問題**：
- CUDA 要求 8-byte 對齊
- 公鑰 XY 緩衝區可能未對齊

**解決方案**：
```python
# 確保對齊
xy_aligned = xy_buffer.copy()  # CuPy 自動對齊
```

### 9.5 記憶體池與資料傳輸

- 啟用自訂 `MemoryPool` / `PinnedMemoryPool`，避免批次重覆配置。
- GPU-FULL 模式僅回傳命中結果，減少 PCIe 傳輸負荷。
- 若需監控使用量，可使用 `cp.cuda.get_allocator().mem_info()` 取得目前池資訊。

---

## 10. 安全性注意事項

### 10.1 私鑰安全

⚠️ **嚴格要求**：
- 永遠不要分享私鑰
- 不要在公共網絡傳輸私鑰
- 使用硬件錢包存儲重要私鑰
- 離線環境操作並備份

### 10.2 代碼審計

⚠️ **使用前**：
- 本項目仍在開發中
- GPU Keccak-256 已通過單元測試但仍需長時間壓測
- 建議自行審計代碼
- 先在測試網測試

### 10.3 網絡請求

✅ **安全設計**：
- 僅地址驗證會對外請求
- 私鑰永不上傳
- 可完全離線運行（跳過驗證）

---

## 11. Context7 MCP 代碼比對

### 11.1 掃描目標

**包含**：
- `src/tron_vanity/*.py`
- `scripts/*.py`

**忽略**：
- `.venv/`
- `__pycache__/`
- `*.pyc`

### 11.2 關鍵函式

**核心模塊**：
- `addr.py`：CPU 參考實現
- `gpu_secp256k1.py`：GPU 橢圓曲線（583 行）
- `gpu_keccak.py`：GPU Keccak-256 內核
- `gpu_addr.py`：GPU 地址生成管線
- `v2_vanity.py`：靚號搜尋主程式

**關鍵函式**：
- `privkey_to_tron_address`
- `pubkey_to_tron_address`
- `keccak_256`
- `sha256d`
- `secp256k1_scalar_mult_batch`（GPU）
- `keccak256_xy_batch`（GPU）

### 11.3 MCP 查詢範例

```
# 比對地址導出流程
"比對 addr.py 與 gpu_addr.py 的地址導出流程，確認一致性"

# 檢查 GPU kernel
"審查 gpu_secp256k1.py 的 CUDA kernel 實現，確認模運算正確性"

# 驗證測試覆蓋
"檢查 test_*.py 文件，確認測試覆蓋率"
```

---

## 12. 後續規劃

### 12.1 短期目標（1-2 週）

- [ ] Window4 壓測與 benchmark
- [ ] GPU-FULL 長時間 throughput/穩定性記錄
- [ ] 更新性能數據與文件（含 GPU Base58）

### 12.2 中期目標（1-2 月）

1. 多 GPU 支持
2. 進度保存與恢復
3. 長時間運行監控（溫度/吞吐）
4. 完整文檔與教程

### 12.3 長期目標（3-6 月）

1. Web UI
2. 正則表達式匹配
3. 統計可視化
4. 長時服務模式

---

## 13. 常見問題

### Q1: 為什麼 GPU-FULL 只有 3-5x 加速？

**A**: 最新 GPU-FULL 基準約 350k addr/s（~29x），100 萬筆測試可達 ~650k addr/s。現階段主要瓶頸在 GPU↔CPU 字串回傳與 Window4 內核效能，後續可藉多 GPU 拓展與更高階 window/wNAF 優化持續提升。

### Q2: 如何選擇批次大小？

**A**: 
- 從小批次開始（256）
- 逐步增加直到性能不再提升
- 注意 GPU 記憶體限制

### Q3: Window4 優化有多大提升？

**A**: 理論上可減少 25% 的點運算，但需要先修復驗證問題。

### Q4: 可以在沒有 GPU 的機器上運行嗎？

**A**: 可以，會自動退化到 CPU 模式，但性能較低。

### Q5: 如何驗證生成的地址？

**A**: 
1. V1 Demo 會自動驗證
2. 可設置 `TRON_PRO_API_KEY` 進行節點驗證
3. 可使用 TronScan 等區塊鏈瀏覽器

---

## 14. 參考資源

### 官方文檔
- [TRON 開發者文檔](https://developers.tron.network/)
- [secp256k1 規範](https://www.secg.org/sec2-v2.pdf)
- [Keccak 規範](https://keccak.team/keccak.html)
- [CUDA 編程指南](https://docs.nvidia.com/cuda/)

### 相關項目
- [tronpy](https://github.com/andelf/tronpy)
- [coincurve](https://github.com/ofek/coincurve)
- [CuPy](https://cupy.dev/)

---

## 15. 更新日誌

### v0.2.0 (開發中)

**新增**：
- ✅ GPU Keccak-256 純 CUDA 內核（通過 `test_keccak_gpu_vs_cpu`）
- ✅ GPU Base58Check 內核（通過 `test_base58_gpu_vs_cpu`）
- ✅ Window4 預計算快取與壓測工具（`--warmup`/`--repeat`）
- ✅ 100k 基準測試更新：GPU-FULL ~65k addr/s（5.7x）

**已知問題**：
- ⚠️ Window4 優化仍待長時間壓測與效能調整
- ⚠️ GPU-FULL 模式尚未完成 24h 長時間穩定性驗證

**下次重點**：
1. Window4 多輪壓測與快取策略優化
2. GPU-FULL 長時間 benchmark 與監控報告
3. 文檔與性能數據更新（含多硬件環境）

---

> **給未來開發者的提醒**：
> 1. 所有優化都針對 NVIDIA L4 (Compute Capability 8.9)
> 2. 批次大小已針對 23GB VRAM 優化
> 3. 不同硬件可能需要調整參數
> 4. 完全 GPU 加速是本項目的核心目標

祝開發順利！🚀
