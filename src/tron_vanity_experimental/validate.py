# -*- coding: utf-8 -*-
"""
驗證工具：
- 雙實作比對：本地演算法 vs tronpy（獨立實作）
- 選配：呼叫 TRON 節點 REST API 的 /wallet/validateaddress 驗證格式
  注意：出於安全性，本模組不會將私鑰送出網路，只會上傳地址做格式驗證。
"""
from __future__ import annotations

import os
import json
import typing as t

import requests

from .addr import (
    privkey_to_tron_address,
)


def try_tronpy_derive(privkey: bytes) -> t.Optional[str]:
    """使用 tronpy 依私鑰導出 Base58 地址（離線演算法），若套件缺失則回傳 None。"""
    try:
        from tronpy.keys import PrivateKey

        pk = PrivateKey(privkey)
        return pk.public_key.to_base58check_address()
    except Exception:
        return None


def validate_with_trongrid(address_b58: str) -> t.Optional[bool]:
    """呼叫 TronGrid/FullNode REST 的 /wallet/validateaddress。
    - 預設 URL: https://api.trongrid.io
    - 可用環境變數覆寫：TRON_GRID_URL, TRON_PRO_API_KEY
    - 回傳 True/False 或 None（呼叫失敗/未配置）
    """
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
        # 依照節點回應格式，通常含有 result: bool
        return bool(data.get("result"))
    except Exception:
        return None


def validate_private_key(privkey: bytes) -> dict:
    """整合驗證：
    - 使用本地演算法導出地址
    - 使用 tronpy 導出地址，與本地比對
    - （選配）呼叫節點 validateaddress 以檢查地址格式
    """
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


__all__ = [
    "validate_private_key",
    "validate_with_trongrid",
    "try_tronpy_derive",
]
