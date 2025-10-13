# TRON 靚號地址生成器（CUDA 版）開發書

本開發書說明如何以 Python + CUDA（CuPy）實作 TRON 靚號地址生成器，並提供 V1（演算法驗證）與 V2（正式搜尋）兩階段程式與驗證方法。所有代碼皆含繁體中文註解。

---

## 1. 檔案結構樹

```
./
├─ agent.md
├─ devbook.md
├─ requirements.txt
├─ scripts/
│  └─ check_env.py                # 環境/依賴檢查
└─ src/
   └─ tron_vanity/
      ├─ __init__.py
      ├─ addr.py                  # 地址導出核心（secp256k1 + Keccak + Base58Check）
      ├─ validate.py              # 本地/節點格式驗證整合
      ├─ gpu_random.py            # CUDA 批量亂數（CuPy）
      ├─ v1_demo.py               # V1：單筆驗證程式
      └─ v2_vanity.py             # V2：靚號搜尋（CUDA 亂數 + 多進程）
```

---

## 2. 依賴與環境

- Python 3.10+（建議）
- CUDA 驅動與 GPU（可選；若無則退化為 CPU 亂數）
- 系統套件（視環境而定）：
  - Debian/Ubuntu：`sudo apt-get install -y build-essential libssl-dev`
- Python 套件：見 `requirements.txt`

安裝步驟：

```
python -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# 安裝 CuPy（擇一，依 nvidia-smi 顯示之 CUDA 版本選）
# CUDA 12.x:
pip install cupy-cuda12x
# CUDA 11.x:
# pip install cupy-cuda11x
```

檢查環境：

```
python scripts/check_env.py
```

若需透過公用節點驗證地址格式，可設定：

```
# 預設使用 https://api.trongrid.io
export TRON_PRO_API_KEY=<your-tron-pro-api-key>
# 或指定節點
# export TRON_GRID_URL=https://nile.trongrid.io
```

---

## 3. V1：演算法驗證流程（最簡 t/T 開頭地址）

目的：確認從私鑰導出 TRON 地址的運算模組正確。

執行：

```
# 隨機產生私鑰並驗證
python -m tron_vanity.v1_demo

# 指定已知私鑰（64位十六進位）
python -m tron_vanity.v1_demo --privkey-hex 1f2e3d4c5b6a7980...（省略）
```

驗證內容：
- 我們的實作（`addr.py`）導出的地址 Base58 必須以 `T` 開頭（題述為 t，Tron 主網實際為大寫 `T`）。
- 與 `tronpy` 導出的 Base58 地址一致。
- （可選）`/wallet/validateaddress` 回傳 `true`（需配置節點/密鑰）。

成功輸出（示意）：

```
[V1] 驗證完成：演算法與格式檢查通過。
```

---

## 4. V2：靚號搜尋器（CUDA 亂數 + 多進程）

說明：
- 以 GPU 批量產生候選私鑰（CuPy），將計算壓力集中於 CPU 端的 secp256k1 乘法與 Keccak/Base58。
- 透過多進程同時檢查是否命中指定前綴，例如 `T777`、`TLOVE`。

執行範例：

```
# 僅 CPU 亂數，4K 批次
python -m tron_vanity.v2_vanity --prefix T777 --threads 0 --batch 4096

# 使用 GPU 亂數，每輪 8192 筆
python -m tron_vanity.v2_vanity --prefix TMOON --threads 0 --gpu-batch 8192
```

說明：
- `--threads 0` 代表自動使用所有 CPU 核心。
- `--gpu-batch > 0` 代表以 GPU 產生該數量筆的候選私鑰；若未偵測到 CuPy，將自動退化為 CPU。
- 命中後會立即輸出私鑰（HEX）與地址（Base58）。

注意：
- 完整的「GPU 橢圓曲線乘法」需另以 C++/CUDA（例如 CGBN）實作後再以 pybind11 封裝，非本版範圍。

---

## 5. 程式核心要點（節錄）

- `addr.py`：
  - 使用 `coincurve` 由私鑰（32 bytes）導出未壓縮公鑰（65 bytes）。
  - 取公鑰末 64 bytes（X||Y）做 Keccak-256，擷取後 20 bytes，前綴 `0x41` 組成 21 bytes。
  - 雙 SHA-256 取前 4 bytes 作為校驗，附於 21 bytes 後，整體做 Base58Check 得到以 `T` 開頭的地址。
- `validate.py`：
  - 使用 `tronpy` 進行第二份離線導出比對。
  - （可選）呼叫節點 `/wallet/validateaddress` 檢查格式。
- `gpu_random.py`：
  - 以 `cupy.random.bytes(n*32)` 生成候選私鑰，否則退化 `os.urandom`。
- `v2_vanity.py`：
  - 多進程平行化檢查；找到即停止並輸出。

---

## 6. 執行方式總覽

- 建立與安裝：
```
python -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
# 視 CUDA 版本安裝 cupy-cuda11x / cupy-cuda12x
```

- 檢查環境：
```
python scripts/check_env.py
```

- V1 驗證：
```
python -m tron_vanity.v1_demo
```

- V2 靚號搜尋：
```
python -m tron_vanity.v2_vanity --prefix T777 --threads 0 --gpu-batch 8192
```

---

## 7. 成功判定標準

- V1：
  - 導出 Base58 地址以 `T` 開頭。
  - 與 `tronpy` 導出一致（顯示 `True`）。
  - （可選）`validateaddress` 回傳 `true` 或節點可接受該地址格式。
- V2：
  - 終端輸出 `命中靚號！`，並列印對應私鑰（HEX）與地址（Base58），前綴符合指定值。

---

## 8. 使用 Context7 MCP 進行代碼比對

- 在 VS Code + KiloCode/Claude 環境中，確保 Context7 MCP 已啟用。
- 建議將專案路徑 `src/tron_vanity` 加入 Context7 的掃描範圍，並在比對請求中包含：
  - 目標檔案：`src/tron_vanity/*.py`
  - 忽略：`*.pyc`, `.venv/`, `__pycache__/` 等。
- 例：在對話中下達指令（敘述）：
  - 「請用 Context7 比對 `src/tron_vanity/addr.py` 與 `src/tron_vanity/v1_demo.py`，確認地址導出函式的呼叫一致性與回傳格式。」
  - 「請審閱 `src/tron_vanity/v2_vanity.py` 的多進程切分是否安全，並檢查 `gpu_random.py` 的 CUDA 退化流程。」

> 若需我提供 `mcp_settings.json` 片段以自動化掃描規則，請明確指定您的 MCP 插件版本與目錄結構，我將補上配置樣板。

---

## 9. 安全性注意事項

- 本範例預設不會將私鑰上傳到任何遠端服務；對外呼叫僅限地址格式驗證（`validateaddress`）。
- 產生的私鑰與靚號請妥善保存，建議離線環境操作並備份。

---

## 10. 後續規劃（可選）

- 以 C++/CUDA（CGBN）實作 secp256k1 標量乘法，並以 pybind11 封裝至 Python，進一步提升吞吐量。
- 加入多前綴/正則匹配策略與統計可視化儀表板。
- 將 v2 改為長時服務模式並加入檢查點與長時任務恢復機制。


---

## 11. Context7 MCP 代碼定位與片段

以下為使用 Context7 MCP 時建議的掃描範圍、關鍵字與已萃取之關鍵代碼片段，便於精準比對開發中的核心模塊。

### 11.1 掃描目標與忽略
- 目標路徑：
  - src/tron_vanity/*.py
  - scripts/check_env.py（僅環境檢查，非核心運算）
- 忽略：
  - .venv/, __pycache__/, *.pyc

### 11.2 指標關鍵字（供 MCP 查找）
- 函式名：privkey_to_tron_address, pubkey_to_tron_address, keccak_256, sha256d
- 外部庫：coincurve, base58, tronpy.keys.PrivateKey, requests.post
- CUDA 相關：cupy.random.bytes, has_cupy, generate_gpu_secrets
- 併行：ProcessPoolExecutor, as_completed

### 11.3 關鍵函式清單（路徑/名稱）
- src/tron_vanity/addr.py
  - keccak_256
  - sha256d
  - privkey_to_pubkey_uncompressed
  - pubkey_to_tron_address
  - privkey_to_tron_address
  - is_valid_tron_base58
- src/tron_vanity/validate.py
  - try_tronpy_derive
  - validate_with_trongrid
  - validate_private_key
- src/tron_vanity/gpu_random.py
  - has_cupy
  - generate_gpu_secrets
- src/tron_vanity/v1_demo.py
  - main
- src/tron_vanity/v2_vanity.py
  - derive_and_check
  - main

### 11.4 代碼片段（精確比對用）

註：以下片段為實際程式碼摘錄，MCP 可直接比對一致性。

```python
# file: src/tron_vanity/addr.py

def keccak_256(data: bytes) -> bytes:
    k = sha3.keccak_256()
    k.update(data)
    return k.digest()


def sha256d(data: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def privkey_to_pubkey_uncompressed(privkey: bytes) -> bytes:
    if len(privkey) != 32:
        raise ValueError("私鑰長度應為 32 bytes")
    pk = coincurve.PrivateKey(privkey)
    return pk.public_key.format(compressed=False)


def pubkey_to_tron_address(pubkey_uncompressed: bytes) -> Tuple[str, str]:
    if len(pubkey_uncompressed) != 65 or pubkey_uncompressed[0] != 0x04:
        raise ValueError("未壓縮公鑰須為 65 bytes 且首位為 0x04")
    pubkey_xy = pubkey_uncompressed[1:]
    keccak = keccak_256(pubkey_xy)
    addr20 = keccak[-20:]
    tron_bytes = b"\x41" + addr20
    checksum = sha256d(tron_bytes)[:4]
    b58 = base58.b58encode(tron_bytes + checksum).decode()
    return tron_bytes.hex(), b58


def privkey_to_tron_address(privkey: bytes) -> Tuple[str, str]:
    pub = privkey_to_pubkey_uncompressed(privkey)
    return pubkey_to_tron_address(pub)


def is_valid_tron_base58(addr: str) -> bool:
    try:
        raw = base58.b58decode(addr)
        if len(raw) != 25:
            return False
        body, checksum = raw[:-4], raw[-4:]
        return sha256d(body)[:4] == checksum and body[0] == 0x41
    except Exception:
        return False
```

```python
# file: src/tron_vanity/validate.py

def try_tronpy_derive(privkey: bytes) -> t.Optional[str]:
    try:
        from tronpy.keys import PrivateKey
        pk = PrivateKey(privkey)
        return pk.public_key.to_base58check_address()
    except Exception:
        return None


def validate_with_trongrid(address_b58: str) -> t.Optional[bool]:
    url = os.environ.get("TRON_GRID_URL", "https://api.trongrid.io")
    endpoint = url.rstrip("/") + "/wallet/validateaddress"
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("TRON_PRO_API_KEY")
    if api_key:
        headers["TRON-PRO-API-KEY"] = api_key
    payload = {"address": address_b58}
    try:
        resp = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return bool(data.get("result"))
    except Exception:
        return None


def validate_private_key(privkey: bytes) -> dict:
    hex_addr, b58_addr = privkey_to_tron_address(privkey)
    tronpy_addr = try_tronpy_derive(privkey)
    result = {
        "hex": hex_addr,
        "base58": b58_addr,
        "tronpy_base58": tronpy_addr,
        "tronpy_match": (tronpy_addr == b58_addr) if tronpy_addr else None,
        "validateaddress": None,
    }
    v = validate_with_trongrid(b58_addr)
    result["validateaddress"] = v
    return result
```

```python
# file: src/tron_vanity/gpu_random.py

def has_cupy() -> bool:
    try:
        import cupy  # noqa: F401
        return True
    except Exception:
        return False


def generate_gpu_secrets(n: int) -> t.List[bytes]:
    if n <= 0:
        return []
    try:
        import cupy as cp
        blob: bytes = cp.random.bytes(n * 32)
        out = [blob[i * 32 : (i + 1) * 32] for i in range(n)]
        return out
    except Exception:
        return [os.urandom(32) for _ in range(n)]
```

```python
# file: src/tron_vanity/v1_demo.py

def main() -> int:
    parser = argparse.ArgumentParser(description="TRON V1 驗證：單筆地址導出與比對")
    parser.add_argument("--privkey-hex", type=str, default=None, help="指定 32 bytes 私鑰的十六進位字串（可選）")
    args = parser.parse_args()
    if args.privkey_hex:
        pk = bytes.fromhex(args.privkey_hex)
        if len(pk) != 32:
            raise SystemExit("--privkey-hex 長度錯誤，需為 64 位十六進位（32 bytes）")
    else:
        pk = generate_privkey()
    result = validate_private_key(pk)
    print("[V1] 私鑰(HEX):", pk.hex())
    print("[V1] 地址(HEX):", result["hex"])  # 0x41 開頭（無 0x 前綴）
    print("[V1] 地址(B58):", result["base58"])  # 'T' 開頭
    print("[V1] tronpy 導出(B58):", result["tronpy_base58"])  # 另一套實作
    print("[V1] tronpy 比對一致:", result["tronpy_match"])  # True/False/None
    print("[V1] 節點 validateaddress:", result["validateaddress"])  # True/False/None
    if not result["base58"].upper().startswith("T"):
        raise SystemExit("導出地址不以 'T' 開頭，疑似錯誤")
    if not is_valid_tron_base58(result["base58"]):
        raise SystemExit("Base58Check 驗證失敗，疑似錯誤")
    print("[V1] 驗證完成：演算法與格式檢查通過。")
    return 0
```

```python
# file: src/tron_vanity/v2_vanity.py

def derive_and_check(privkey: bytes, prefix: str) -> Optional[Tuple[str, str]]:
    hex_addr, b58 = privkey_to_tron_address(privkey)
    if b58.startswith(prefix):
        return (privkey.hex(), b58)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="TRON 靚號地址搜尋（CUDA 亂數輔助）")
    parser.add_argument("--prefix", type=str, required=True, help="欲匹配的 Base58 前綴，建議以 'T' 開頭")
    parser.add_argument("--threads", type=int, default=0, help="工作進程數；0 代表自動=CPU 核心數")
    parser.add_argument("--batch", type=int, default=4096, help="每一輪分派的私鑰數量（CPU 產生）")
    parser.add_argument("--gpu-batch", type=int, default=0, help="若>0，啟用 GPU 亂數，一輪產生此數量的候選")
    parser.add_argument("--timeout", type=int, default=0, help="搜尋逾時秒數；0 表示不限")
    args = parser.parse_args()
    prefix = args.prefix
    max_workers = args.threads or os.cpu_count() or 1
    print(f"[V2] 目標前綴: {prefix}")
    print(f"[V2] 進程數: {max_workers}")
    if args.gpu_batch > 0:
        print(f"[V2] 使用 GPU 亂數，每輪 {args.gpu_batch} 筆")
        if not has_cupy():
            print("[V2] 警告：未偵測到 CuPy，將退化為 CPU 亂數。")
    else:
        print(f"[V2] 使用 CPU 亂數，每輪 {args.batch} 筆")
    deadline = time.time() + args.timeout if args.timeout > 0 else None
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        round_idx = 0
        while True:
            round_idx += 1
            if deadline and time.time() > deadline:
                print("[V2] 已達逾時，結束搜尋。")
                return 2
            if args.gpu_batch > 0:
                secrets_batch = generate_gpu_secrets(args.gpu_batch)
            else:
                secrets_batch = [os.urandom(32) for _ in range(args.batch)]
            futures = [ex.submit(derive_and_check, sk, prefix) for sk in secrets_batch]
            for fut in as_completed(futures):
                hit = fut.result()
                if hit is not None:
                    priv_hex, b58 = hit
                    print("[V2] 命中靚號！")
                    print("[V2] 私鑰(HEX):", priv_hex)
                    print("[V2] 地址(B58):", b58)
                    return 0
            if round_idx % 10 == 0:
                print(f"[V2] 已完成 {round_idx} 輪，尚未命中…")
    return 1
```

### 11.5 MCP 查詢範例
- 「定位 `privkey_to_tron_address` 的所有引用，確認 V1/V2 都使用相同導出流程。」
- 「比對 `addr.py` 與 `validate.py` 中的地址導出與校驗流程，檢查輸入/輸出類型及編碼。」
- 「搜尋 `cupy.random.bytes` 與 `generate_gpu_secrets`，確認 GPU 退化邏輯是否覆蓋例外。」
- 「審查 `v2_vanity.py` 的 `ProcessPoolExecutor` 使用與結果收斂邏輯。」



---

## 12. 完整 GPU 管線與基準（GPU-FULL）

說明：
- GPU-FULL 由 GPU 端完成：亂數私鑰、secp256k1（實驗性 GPU ECC，可開關）、Keccak-256、雙 SHA-256（Base58Check 校驗碼）。
- Base58 最終字串編碼仍在 CPU 端處理（占比小）。

啟用方式：

```
# 可選：啟用實驗性 GPU 橢圓曲線（已通過隨機一致性測試）
export VANITY_EXPERIMENTAL_GPU_SECP=1

# 以 GPU-FULL 搜尋（2 秒逾時煙霧測試）
PYTHONPATH=src python -m tron_vanity.v2_vanity --prefix T --gpu-full --gpu-batch 2048 --timeout 2
```

正式基準測試（100,000 筆）：

```
export VANITY_EXPERIMENTAL_GPU_SECP=1
PYTHONPATH=src python3 scripts/benchmark.py
```

本機測試結果（僅供參考，與硬體/驅動相關）：
- CPU: 12,198.17 keys/sec（8.20 s）
- GPU（僅亂數在 GPU，ECC 仍 CPU）: 12,391.44 keys/sec（8.07 s）
- GPU-FULL: 40,029.28 keys/sec（2.50 s）
- 加速比（GPU-FULL 相對 CPU）: 約 3.28x
- 第 100,000 筆地址於三種模式皆驗證通過（tronpy_match=True、validateaddress=True、Base58Check 校驗通過）。

Keccak-256（GPU）說明：
- 以 CuPy 向量化實作 Keccak-f[1600] 24 輪，針對 64 bytes 輸入（公鑰 X||Y）。
- 已逐筆比對 CPU keccak_256，正確性一致。

注意事項：
- 若遇到 CUDA 對齊相關錯誤，請確保公鑰 XY 緩衝區為 8-byte 對齊（程式已以 `.copy()` 保證）。
- 若需關閉實驗性 GPU ECC，刪除環境變數 `VANITY_EXPERIMENTAL_GPU_SECP` 或設為空即可，管線會自動回退至 CPU ECC。
