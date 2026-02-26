# TRON GPU 地址生成器

高性能 TRON 靚號地址生成器，**全 GPU 加速管線**。

## 快速開始

```bash
source .venv/bin/activate

# GPU 模式
PYTHONPATH=src python -m tron_vanity --suffix 88888

# CPU 模式
PYTHONPATH=src python -m tron_vanity --suffix 88888 --cpu-only
```

## GPU 管線

```
私鑰(32B) → secp256k1 → 公鑰(65B) → Keccak-256 → addr20 → SHA256d+Base58 → TAddress
```

| 模組 | 實作 | 檔案 | 速率 |
|------|------|------|------|
| secp256k1 (wNAF) | schoolbook Montgomery + wNAF W4/W6/W8 | `gpu_secp256k1.py` | ~1.44 Mkeys/s |
| **secp256k1 (增量)** | **增量點加法 P += stride*G** | **`gpu_incremental.py`** | **15.35 Mkeys/s** |
| Keccak-256 | 純 CUDA Keccak-f[1600] | `gpu_keccak.py` | — |
| SHA-256 + Base58 | 融合 CUDA kernel | `gpu_addr.py` | — |
| 硬體自適應 | 自動選擇 wNAF 窗口/批次/threads | `hardware_config.py` | — |

## 開發規則

> **實驗版優先**：所有修改先在 `tron_vanity_experimental/` 開發測試，確認無誤後再複製到穩定版 `tron_vanity/`。

## 依賴

```bash
pip install -r requirements.txt
pip install cupy-cuda12x
```

## 環境

- Python 3.13+ / CUDA 12.x
- NVIDIA L4 (23GB VRAM, CC 8.9)
- pycryptodome（Keccak-256 ≠ SHA3-256）

---

## ⚠️ NVCC 優化踩坑記錄（避免重蹈覆轍）

以下 CUDA kernel 層級的 ECC 數學優化**全部導致 3 倍效能倒退**：

| 方案 | 失敗根因 |
|------|----------|
| CIOS Montgomery (`t[16]`→`t[9]`) | CIOS shift-down 迴圈讓 NVCC 產生更差指令排程，大量 spill 至 local memory |
| `__launch_bounds__(128, 4/5)` | 強制 128 regs 的 spill 量大於 occupancy 增益 |
| `point_add_mixed`（新 `__device__` 函式） | **NVCC 強制 inline 所有 `__device__`** → kernel 膨脹 → register allocator 崩潰 |
| 專用 `montgomery_sqr` | 需要 `t[16]` 陣列，buffer overflow，無法省暫存器 |

### 核心教訓

1. **不要新增 `__device__` 函式** — NVCC inline 會膨脹 kernel，導致 register spill
2. **不要用 `__launch_bounds__`** — 在此 workload 下強制 spill 效果更差
3. **不要改 Montgomery multiplication** — 原始 schoolbook `t[16]` 已是 NVCC 近最佳
4. **效能提升應從演算法層級** — 如增量掃描（1 add/key 取代 320 ops/key）

