# TRON Vanity Address Generator with GPU Acceleration

高性能 TRON 靚號地址生成器，支持 CPU 和 GPU 加速模式。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-downloads)

---

## 📋 目錄

- [項目概述](#項目概述)
- [核心特性](#核心特性)
- [性能指標](#性能指標)
- [安裝說明](#安裝說明)
- [快速開始](#快速開始)
- [開發進度](#開發進度)
- [技術要點](#技術要點)
- [待辦清單](#待辦清單)
- [常用指令](#常用指令)
- [項目結構](#項目結構)
- [貢獻指南](#貢獻指南)

---

## 🎯 項目概述

本項目實現了高性能的 TRON 地址生成器，特別針對靚號地址搜索進行了優化。通過 GPU 加速關鍵的密碼學運算（secp256k1 橢圓曲線、SHA-256、Keccak-256），實現了相比純 CPU 實現的顯著性能提升。

### 什麼是靚號地址？

靚號地址是指包含特定前綴或模式的區塊鏈地址，例如：
- `T7777...` - 包含連續數字
- `TABCD...` - 包含特定字母組合
- `T1234567...` - 包含順序數字

---

## ✨ 核心特性

### 已實現功能

- ✅ **完整 GPU secp256k1 實現**
  - 583 行 CUDA kernel
  - 256-bit 大數模運算（針對 secp256k1 素數優化）
  - Jacobian 座標橢圓曲線點運算
  - 標量乘法（double-and-add 算法）
  - **性能：540k keys/sec**（批次 16384，**38.6x 加速**）

- ✅ **GPU 地址生成管線**
  - GPU 隨機數生成（CuPy）
  - GPU secp256k1 公鑰計算（Window4 內核開發中，暫用位元掃描）
  - GPU Keccak-256 與雙 SHA-256 合併至 Base58 kernel
  - GPU Base58Check 字串生成（結果僅回傳命中項）
  - GPU 端 Base58 前綴匹配 + 多 stream 管線化
  - **性能：~350k addr/s**（100k 地址基準，GPU-FULL）

- ✅ **Base58 前綴 / 尾碼雙向匹配**
  - GPU 於 Base58 結果上同時支援前綴 (`T...`) 與尾碼（例如 `...88888`）比對
  - 支援 CPU 回退邏輯確保一致性

- ✅ **硬體自適應配置（HardwareAdaptiveConfig）**
  - 自動偵測 GPU 型號、SM 數、VRAM、Compute Capability
  - 依硬體設定批次大小、Window4 門檻、CUDA threads、Stream 數
  - 以環境變數覆蓋，方便手動調整

- ✅ **多種運行模式**
  - **V1 Demo**：單地址生成與驗證
  - **V2 Vanity**：靚號地址搜索
    - CPU 模式
    - GPU 模式（僅 secp256k1）
    - GPU-FULL 模式（完整管線）

- ✅ **完整測試套件**
  - GPU 模運算測試（64 筆隨機樣本）
  - GPU/CPU 一致性測試（16 筆隨機私鑰）
  - GPU Base58Check 對照測試
  - Window4 優化測試（開發中）

### 開發中功能

- ⚠️ **Window4 優化**
  - 4-bit 視窗法標量乘法
  - 預計算表已生成
  - 驗證測試未通過（64/64 不匹配）

---

## 📊 性能指標

### GPU secp256k1 性能（純計算）

| 批次大小 | 時間 (ms) | 吞吐量 (keys/s) | 加速比 |
|---------|----------|----------------|--------|
| 256     | 19.2     | 13.3k          | 1.0x   |
| 1024    | 19.5     | 52.5k          | 3.9x   |
| 4096    | 21.8     | 187.9k         | 14.1x  |
| 16384   | 30.3     | 540.6k         | **38.6x** |

### 完整地址生成性能

| 模式 | 耗時 | 吞吐量 (addr/s) | 備註 |
|------|------|----------------|------|
| CPU | 8.21 s | 12.2k | Python + coincurve |
| GPU Random-only | 8.09 s | 12.4k | 亂數在 GPU，其餘 CPU |
| GPU-FULL | 0.28 s | **354.7k** | 完整 GPU 管線（Keccak+SHA+Base58 融合，多 stream 管線） |

> 註：數據包含 GPU Keccak/Base58 融合與雙緩衝 Streams；在 1,000,000 筆測試中可達 ~650k addr/s。不同硬體與前綴難度下吞吐量會有所差異。

### GPU-FULL 模式實測

```bash
# 測試命令
python -m tron_vanity.v2_vanity --prefix T7 --threads 0 --gpu-batch 4096 --timeout 10

# 結果（範例，視前綴與硬體而定）
總處理：450,000 地址
總時間：約 10 秒（內建前綴匹配可動態調整批次）
平均速度：~45k addr/s（依前綴難度而浮動）
加速比：3-5x（相比純 CPU）
```

---

## 🚀 安裝說明

### 系統要求

- **操作系統**：Linux（推薦 Ubuntu 20.04+）
- **Python**：3.8+
- **CUDA**：11.0+（GPU 模式）
- **GPU**：NVIDIA GPU with Compute Capability 6.0+

### 🎯 生產環境硬件規格（重要）

> **⚠️ 性能優化提醒**：本項目的所有性能測試和優化都是基於以下硬件環境進行的。為了獲得最佳性能，建議在相同或更高規格的硬件上運行。

**當前生產環境配置**：

| 組件 | 規格 | 說明 |
|------|------|------|
| **操作系統** | Ubuntu 20.04.6 LTS (Focal Fossa) | Linux Kernel 5.15.0-1088-gcp |
| **CPU** | Intel Xeon @ 2.20GHz | 4 vCPUs (2 cores, 2 threads/core) |
| **內存** | 16 GB | 可用 ~12 GB |
| **GPU** | **NVIDIA L4** | 23 GB VRAM |
| **GPU 架構** | Ada Lovelace | Compute Capability **8.9** |
| **CUDA Driver** | 535.261.03 | Driver API 12.2 |
| **CUDA Runtime** | 12.0.6 | Runtime API 12.6 |
| **平台** | Google Cloud Platform | instance-20250725-173413 |

**關鍵性能參數**：
- **GPU 記憶體**：23 GB（足夠處理大批次）
- **Compute Capability 8.9**：支持最新 CUDA 特性
- **推薦批次大小**：
  - GPU secp256k1：16384（最佳性能）
  - 完整地址生成：4096-8192
  - GPU-FULL 模式：4096

**性能基準**（以 100,000 地址測試）：
- GPU secp256k1：540k keys/s（批次 16384）
- GPU-FULL 模式：249,662 addr/s（批次 4096，~20.5x CPU）
- CPU 參考：12,176 addr/s

**硬件升級建議**：
- 更高端 GPU（如 A100、H100）可獲得更好性能
- 更多 CPU 核心可提升 CPU 模式性能
- 更大內存可支持更大批次處理

> **給開發者的提醒**：
> 1. 所有代碼優化都針對 NVIDIA L4 (Compute Capability 8.9)
> 2. 批次大小參數已針對 23GB VRAM 優化
> 3. 如使用不同硬件，可能需要調整批次大小
> 4. 性能數據僅供參考，實際性能取決於硬件配置

### 依賴安裝

```bash
# 1. 克隆倉庫
git clone https://github.com/xu666xu888-ai/tron.git
cd tron

# 2. 創建虛擬環境（推薦）
python3 -m venv .venv
source .venv/bin/activate

# 3. 安裝依賴
pip install -r requirements.txt

# 4. 若啟用 GPU，請先確認系統已安裝對應 CUDA Toolkit / 驅動
#    （若缺少，請依下方「依賴指引」手動安裝）

# 5. 驗證環境
python scripts/check_env.py
```

### 依賴自動安裝與指引

- `python -m tron_vanity.cli` 將檢測 Python 依賴並嘗試透過 `pip` 安裝缺失模組（如 `cupy`, `rich`, `psutil`, `GPUtil`）。
- 針對 CuPy 會依據 CUDA 版本提示安裝 `cupy-cuda11x` / `cupy-cuda12x`；若無 GPU，可改裝 `cupy`（CPU 版本）。
- **CUDA Toolkit、NVIDIA Driver、Node.js 等系統級工具需使用者自行安裝**，CLI 會提供官方指引與命令範例，避免在未知環境中自動變更系統。
- 可使用以下環境變數覆蓋硬體自適應設定：
  - `VANITY_WNAF_MAX_BATCH`、`VANITY_STREAM_COUNT_DEFAULT`、`VANITY_MAX_PENDING_MULTIPLIER`
  - `VANITY_SECP_THREADS`、`VANITY_KECCAK_THREADS`、`VANITY_SHA_THREADS`、`VANITY_BASE58_THREADS`

### requirements.txt

```
# 核心依賴
tronpy>=0.4.0
coincurve>=18.0.0
pysha3>=1.0.2
base58>=2.1.1

# GPU 加速（可選）
cupy-cuda11x>=12.0.0  # 根據 CUDA 版本選擇

# 開發工具
pytest>=7.0.0
```

---

## 🎮 快速開始

### 1. 環境檢查

```bash
python scripts/check_env.py
```

**預期輸出**：
```
✅ Python 版本: 3.8.10
✅ tronpy 已安裝
✅ coincurve 已安裝
✅ CuPy 已安裝
✅ CUDA 可用
✅ GPU 設備: NVIDIA L4 (Compute Capability 8.9)
```

### 2. V1 Demo - 單地址生成

```bash
# 生成隨機地址
PYTHONPATH=src python -m tron_vanity.v1_demo

# 從指定私鑰生成
PYTHONPATH=src python -m tron_vanity.v1_demo --privkey-hex <64位十六進位>
```

**預期輸出**：
```
私鑰: a1b2c3d4...
公鑰: 04abcd...
地址: T7XYZ...
tronpy_match: True
is_valid_tron_base58: True
validateaddress: True (需要 TRON_PRO_API_KEY)
```

### 3. V2 Vanity - 靚號搜索

#### CPU 模式
```bash
PYTHONPATH=src python -m tron_vanity.v2_vanity \
  --prefix T7 \
  --threads 4 \
  --batch 2048 \
  --timeout 30
```

#### GPU 模式（僅 secp256k1）
```bash
PYTHONPATH=src python -m tron_vanity.v2_vanity \
  --prefix T7 \
  --threads 0 \
  --gpu-batch 4096 \
  --timeout 30
```

#### GPU-FULL 模式（完整管線）
> 若要啟用 GPU 版 secp256k1 內核，請先設定 `export VANITY_EXPERIMENTAL_GPU_SECP=1`
> 未指定 `--gpu-batch` 時，會依前綴長度自動調整批次大小。
```bash
PYTHONPATH=src python -m tron_vanity.v2_vanity \
  --prefix T7 \
  --threads 0 \
  --gpu-full \
  --gpu-batch 4096 \
  --timeout 30
```

---

## ✅ 開發進度

### 已完成 ✅

- [x] **GPU secp256k1 實現**（583 行 CUDA kernel）
  - 256-bit 模運算
  - 橢圓曲線點運算
  - 標量乘法
  - 性能：540k keys/s（38.6x 加速）

- [x] **GPU Keccak-256 內核**
  - 24 輪 Keccak-f[1600]（CuPy RawModule）
  - 與 `sha3.keccak_256` 完全一致（`test_keccak_gpu_vs_cpu`）

- [x] **GPU 地址生成管線**
  - GPU 隨機數
  - GPU secp256k1（Window4 內核 + 快取）
  - GPU Keccak-256 + 雙 SHA-256 + Base58Check 融合內核
  - 性能：~430k addr/s（100k 基準，GPU-FULL）

- [x] **GPU-FULL 模式整合**
  - 成功運行測試
  - 平均 ~45k addr/s
  - 3-5x 實際加速

- [x] **測試套件**
  - GPU 模運算測試 ✅
  - GPU/CPU 一致性測試 ✅
  - GPU Keccak vs CPU ✅
  - 環境檢查腳本 ✅

### 進行中 ⚠️

- [ ] **Window4 優化**
  - nibble 解碼/預計算表已校正
  - 新增預計算表快取與 `warmup_window4_table` 工具
  - 需進一步壓測與效能評估

- [ ] **GPU Keccak-256**
  - 接入 GPU-FULL pipeline 後的 throughput 量測（最新 100k 測試：65,076 addr/s）
  - 長時間穩定性測試與記憶體觀測

### 待辦 📋

- [ ] 完整 benchmark
- [ ] Window4 效能調優
- [ ] 文檔完善

---

## 🔧 技術要點

### 1. TRON 地址生成流程

```
私鑰 (32 bytes)
    ↓
secp256k1 點乘 (k * G)
    ↓
公鑰 (64 bytes: X || Y)
    ↓
Keccak-256(公鑰)
    ↓
取後 20 bytes
    ↓
添加前綴 0x41
    ↓
SHA-256(SHA-256(data))
    ↓
取前 4 bytes 作為校驗碼
    ↓
Base58 編碼
    ↓
TRON 地址 (T...)
```

### 2. secp256k1 橢圓曲線

**曲線方程**：y² = x³ + 7 (mod p)

**參數**：
- **p**（素數）：2^256 - 2^32 - 977
- **n**（階）：FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
- **G**（基點）：
  - Gx = 79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
  - Gy = 483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

**優化技術**：
- **Jacobian 座標**：避免昂貴的模逆運算
- **特殊素數快速約簡**：利用 p 的特殊形式
- **批量並行處理**：每個 GPU 線程處理一個私鑰

### 3. Keccak-256 注意事項

**Padding 規則**（pad10*1）：
- Input rate = 136 bytes
- 64 bytes XY 吸收後：
  - Lane 8 XOR 0x01（起始位）
  - Lane 16 XOR 0x80 << 56（尾端位）

**實作重點**：
- 採純 CUDA Keccak-f[1600] 迴圈，避開舊版 CuPy API 限制
- 嚴格對齊 Rho/Pi/Chi 步驟與位元序（little-endian lane）
- 透過 `test_keccak_gpu_vs_cpu` 驗證 64-byte 輸入的一致性

### 4. Window4 預計算表

**概念**：
- 預計算 1G, 2G, ..., 15G
- 將 256-bit 標量切成 64 個 4-bit nibbles
- 每次處理 4 bits，減少點加法次數

**當前問題**：
- nibble 採樣順序（MSB vs LSB）
- 初始 state 處理
- 預計算表 limb 順序

### 5. GPU-FULL 管線

**組件**：
1. GPU 隨機數生成（CuPy）
2. GPU secp256k1（CUDA kernel）
3. GPU Keccak-256（Keccak-f[1600] RawModule）
4. GPU SHA-256（Base58Check 雙重雜湊）
5. GPU Base58Check 編碼（結果回傳 CPU 彙整）

**性能瓶頸**：
- Base58 字串回傳需搬移至 CPU（小量資料，但仍有傳輸成本）
- GPU ↔ CPU 之間的同步/傳輸開銷
- Window4 預計算快取仍待壓測最佳化

---

## 🔜 待辦清單

### 高優先級 🔴

1. **Window4 ECC 壓測**
   - [ ] 大量樣本與多輪迴圈的穩定性
   - [ ] 分析快取策略與 GPU 記憶體佔用
   - [ ] 評估與位元掃描內核的效能差異

2. **GPU-FULL 長時間 benchmark**
   - [ ] 量測 GPU Keccak/Base58 上線後的吞吐演變
   - [ ] 監控溫度與記憶體使用情況

### 中優先級 🟡

3. **性能優化**
   - [ ] 優化 GPU Base58 字串回傳與批次化策略
   - [ ] 減少 GPU ↔ CPU 數據傳輸
   - [ ] 優化批次大小選擇
   - [ ] 記憶體池管理

4. **完整測試**
   - [ ] Keccak & Window4 修正後重新 benchmark
   - [ ] GPU Keccak 單獨 throughput 測試
   - [ ] 壓力測試（長時間運行）
   - [ ] 邊界條件測試

### 低優先級 🟢

5. **文檔與工具**
   - [ ] 更新 devbook.md
   - [ ] 添加 API 文檔
   - [ ] 創建使用教程
   - [ ] 添加更多示例

6. **功能增強**
   - [ ] 支持更多地址模式（正則表達式）
   - [ ] 多 GPU 支持
   - [ ] 進度保存與恢復
   - [ ] Web UI

---

## 📝 常用指令

### 環境檢查

```bash
# 檢查所有依賴
python scripts/check_env.py

# 檢查 CUDA 版本
nvcc --version

# 檢查 GPU 信息
nvidia-smi
```

### 測試命令

```bash
# GPU vs CPU 公鑰比對（16 筆）
PYTHONPATH=src python -m tron_vanity.test_gpu_vs_cpu --n 16

# GPU Keccak-256 vs CPU 參考（32 筆）
PYTHONPATH=src python -m tron_vanity.test_keccak_gpu_vs_cpu --n 32

# GPU 模運算測試（64 筆）
PYTHONPATH=src python -m tron_vanity.test_mod_arith_gpu --n 64

# GPU Base58Check vs CPU 參考（64 筆）
PYTHONPATH=src python -m tron_vanity.test_base58_gpu_vs_cpu --n 64

# Window4 vs 參考內核比對（可指定多輪）
PYTHONPATH=src python -m tron_vanity.test_ecc_w4_vs_ref --n 128 --repeat 3 --warmup
```

### 性能測試

```bash
# 基準測試（100k 地址）
export VANITY_EXPERIMENTAL_GPU_SECP=1
PYTHONPATH=src python3 scripts/benchmark.py
```

### V1 Demo

```bash
# 生成隨機地址
PYTHONPATH=src python -m tron_vanity.v1_demo

# 從私鑰生成
PYTHONPATH=src python -m tron_vanity.v1_demo \
  --privkey-hex a1b2c3d4e5f6...
```

### V2 Vanity

```bash
# CPU 模式（4 線程）
PYTHONPATH=src python -m tron_vanity.v2_vanity \
  --prefix T7 \
  --threads 4 \
  --batch 2048 \
  --timeout 30

# GPU 模式（僅 secp256k1）
PYTHONPATH=src python -m tron_vanity.v2_vanity \
  --prefix T7 \
  --threads 0 \
  --gpu-batch 4096 \
  --timeout 30

# GPU-FULL 模式（完整管線）
PYTHONPATH=src python -m tron_vanity.v2_vanity \
  --prefix T7 \
  --threads 0 \
  --gpu-full \
  --gpu-batch 4096 \
  --timeout 30
```

---

## 📁 項目結構

```
tron-vanity/
├── README.md                    # 本文件
├── requirements.txt             # Python 依賴
├── .gitignore                   # Git 忽略規則
│
├── scripts/                     # 工具腳本
│   ├── check_env.py            # 環境檢查
│   ├── benchmark.py            # 性能測試
│   └── test_gpu_v2.py          # GPU 測試
│
├── src/tron_vanity/            # 核心代碼
│   ├── __init__.py             # 包初始化
│   │
│   ├── addr.py                 # CPU 地址生成
│   ├── validate.py             # 地址驗證
│   │
│   ├── gpu_random.py           # GPU 隨機數
│   ├── gpu_secp256k1.py        # GPU secp256k1 (583 行)
│   ├── hardware_config.py      # GPU/CPU 硬體自適應配置
│   ├── gpu_keccak.py           # GPU Keccak-256
│   ├── gpu_addr.py             # GPU 地址生成管線
│   │
│   ├── v1_demo.py              # V1 演示模式
│   ├── v2_vanity.py            # V2 靚號搜索
│   │
│   ├── test_mod_arith_gpu.py   # GPU 模運算測試
│   ├── test_gpu_vs_cpu.py      # GPU/CPU 一致性測試
│   ├── test_ecc_w4_vs_ref.py   # Window4 測試
│   └── test_secp256k1_debug.py # secp256k1 調試
│
├── agent.md                     # Agent 開發日誌
├── claude.md                    # Claude 對話記錄
└── devbook.md                   # 開發手冊
```

---

## 🧭 CLI 產品化計畫

為了讓 TRON 靚號生成器以「產品級 CLI」形式交付，我們規劃以下三階段工作，詳見 `CODEX_TASK_PRODUCT_CLI.md`：

1. **環境檢測與自動配置**
   - `system_info.py`：收集 OS / CPU / GPU / 記憶體 / Python 版本資訊
   - `dependency_checker.py`：檢查 Python 套件與系統工具（CUDA Toolkit、`nvidia-smi`、Node.js 等）
   - `auto_installer.py`：針對缺失的 Python 依賴執行 `pip install`，若遇系統級工具，提供官方安裝指引與命令範例

2. **動態配置與性能預估**
   - 強化 `hardware_config.py`，依硬體自動調整批次、Stream 數、Window4 門檻、CUDA threads 與記憶體池上限
   - `performance_estimator.py`：根據靚號末碼長度預估搜尋時間（例如 5 位 ≈ 15 分鐘 @ 700k addr/s）

3. **產品級 CLI UI**
   - `cli.py` 為主入口，整合 ASCII Logo、硬體摘要、靚號輸入與難度評估
   - `monitor.py` + `ui_components.py` 使用 `rich` 呈現速率、已檢查數量、GPU/CPU 負載、溫度、性能火花線等資訊
   - 支援前綴與尾碼搜尋、優雅 Ctrl+C、結果匯出與續跑

> **注意**：CUDA Toolkit / NVIDIA Driver / Node.js 等系統級元件仍需使用者依平台指南手動安裝，CLI 會於檢測階段提供推薦命令與官方連結。

---

## 🤝 貢獻指南

歡迎貢獻！請遵循以下步驟：

1. Fork 本倉庫
2. 創建特性分支（`git checkout -b feature/AmazingFeature`）
3. 提交更改（`git commit -m 'Add some AmazingFeature'`）
4. 推送到分支（`git push origin feature/AmazingFeature`）
5. 開啟 Pull Request

### 代碼規範

- 遵循 PEP 8
- 添加類型註解
- 編寫單元測試
- 更新文檔

---

## ⚠️ 安全警告

1. **私鑰安全**：
   - 永遠不要分享您的私鑰
   - 不要在公共網絡上傳輸私鑰
   - 使用硬件錢包存儲重要私鑰

2. **代碼審計**：
   - 本項目仍在開發中
   - GPU Keccak-256 和 Window4 尚未完成驗證
   - 使用前請自行審計代碼

3. **測試網優先**：
   - 建議先在測試網測試
   - 確認功能正常後再用於主網

---

## 📄 許可證

本項目採用 MIT 許可證 - 詳見 [LICENSE](LICENSE) 文件

---

## 🙏 致謝

- [tronpy](https://github.com/andelf/tronpy) - TRON Python SDK
- [coincurve](https://github.com/ofek/coincurve) - secp256k1 綁定
- [CuPy](https://cupy.dev/) - GPU 加速數組運算
- TRON 社區

---

## 📞 聯繫方式

- GitHub: [@xu666xu888-ai](https://github.com/xu666xu888-ai)
- Email: xu666xu888@gmail.com

---

## 🔄 更新日誌

### v0.2.0 (開發中)

**新增**：
- ✅ 完整 GPU secp256k1 實現（540k keys/s）
- ✅ GPU 地址生成管線（~430k addr/s，GPU-FULL 約 35x 加速）
- ✅ GPU Keccak-256 / SHA-256 / Base58Check 內核
- ✅ V1 Demo 和 V2 Vanity 模式
- ✅ 完整測試套件（含 Base58、Window4、Keccak 壓測）

**已知問題**：
- ⚠️ Window4 優化尚需壓測與效能調校
- ⚠️ GPU-FULL 模式尚未完成長時間穩定性測試

---

> **提醒**：Window4 仍在進行壓測，建議先以預設位元掃描內核為主。GPU-FULL 模式已啟用 Keccak/Base58 GPU 核心，但長時間運行仍需觀察溫度與吞吐變化。

> **下次開發重點**：
> 1. Window4 多輪壓測與快取策略調整
> 2. GPU-FULL 長時間 benchmark 與監控報告
> 3. 更新性能數據與文檔（含 Base58 GPU 化成果）

祝開發順利！🚀
