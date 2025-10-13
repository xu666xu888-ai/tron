# -*- coding: utf-8 -*-
"""
V1：基本驗證流程（支援 GPU 亂數）
- 生成一組私鑰（或使用傳入的十六進位私鑰）
- 導出 TRON 地址（hex + base58）
- 使用 tronpy 進行第二份獨立計算比對
- （選配）呼叫節點 /wallet/validateaddress 格式驗證

執行：
  python -m tron_vanity.v1_demo
或：
  python -m tron_vanity.v1_demo --privkey-hex <64位十六進位>
或（GPU 生成私鑰）：
  python -m tron_vanity.v1_demo --use-gpu
"""
from __future__ import annotations

import argparse

from .addr import generate_privkey, privkey_to_tron_address, is_valid_tron_base58
from .validate import validate_private_key


def main() -> int:
    parser = argparse.ArgumentParser(description="TRON V1 驗證：單筆地址導出與比對")
    parser.add_argument(
        "--privkey-hex",
        type=str,
        default=None,
        help="指定 32 bytes 私鑰的十六進位字串（可選）",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="使用 GPU 生成私鑰（需 CuPy，可自動退化）",
    )
    args = parser.parse_args()

    label = "[V1]"

    if args.privkey_hex:
        pk = bytes.fromhex(args.privkey_hex)
        if len(pk) != 32:
            raise SystemExit("--privkey-hex 長度錯誤，需為 64 位十六進位（32 bytes）")
    elif args.use_gpu:
        # 啟用 GPU 私鑰生成（CuPy），若不可用則退化為 CPU
        try:
            from .gpu_random import generate_gpu_secrets, has_cupy
            if not has_cupy():
                print("[V1-GPU] 警告：未偵測到 CuPy，改用 CPU 亂數。")
                pk = generate_privkey()
            else:
                pk = generate_gpu_secrets(1)[0]
                label = "[V1-GPU]"
        except Exception as e:
            print(f"[V1-GPU] 警告：GPU 亂數產生失敗（{e}），改用 CPU 亂數。")
            pk = generate_privkey()
    else:
        pk = generate_privkey()

    result = validate_private_key(pk)

    print(f"{label} 私鑰(HEX):", pk.hex())
    print(f"{label} 地址(HEX):", result["hex"])  # 0x41 開頭（無 0x 前綴）
    print(f"{label} 地址(B58):", result["base58"])  # 'T' 開頭
    print(f"{label} tronpy 導出(B58):", result["tronpy_base58"])  # 另一套實作
    print(f"{label} tronpy 比對一致:", result["tronpy_match"])  # True/False/None
    print(f"{label} 節點 validateaddress:", result["validateaddress"])  # True/False/None

    # 以最簡單規則確認 'T' 開頭（TRON 主網地址必須以 'T' 開頭）
    if not result["base58"].startswith("T"):
        raise SystemExit("導出地址不以 'T' 開頭，疑似錯誤")

    # 基本格式檢查（Base58Check 校驗）
    if not is_valid_tron_base58(result["base58"]):
        raise SystemExit("Base58Check 驗證失敗，疑似錯誤")

    print(f"{label} 驗證完成：演算法與格式檢查通過。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
