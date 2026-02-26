# -*- coding: utf-8 -*-
"""
TRON 地址計算與編碼工具
- 以 secp256k1 橢圓曲線從私鑰導出公鑰
- 以 Keccak-256 取得雜湊後最後 20 bytes 組合成 TRON 地址
- 以 Base58Check 編碼輸出主網地址（以 'T' 開頭）

所有註解採用繁體中文，便於審核與維護。
"""
from __future__ import annotations

import os
import secrets
import hashlib
from typing import Tuple

import base58  # Base58Check 編碼
import coincurve  # 以 libsecp256k1 加速的 Python 綁定
from Crypto.Hash import keccak as _keccak_mod  # pycryptodome Keccak-256（取代 pysha3）


# ------ 基礎工具函式 ------

def keccak_256(data: bytes) -> bytes:
    """計算 Keccak-256 雜湊。
    參考以太坊/波場地址導出流程，對未壓縮公鑰去掉開頭 0x04 後的 64 bytes 做 keccak。
    """
    return _keccak_mod.new(data=data, digest_bits=256).digest()


def sha256d(data: bytes) -> bytes:
    """雙重 SHA-256（用於 Base58Check 校驗碼）。"""
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


# ------ 金鑰與地址導出 ------

def generate_privkey() -> bytes:
    """產生 32 位元組隨機私鑰（secp256k1），使用作業系統安全亂數。"""
    return secrets.token_bytes(32)


def privkey_to_pubkey_uncompressed(privkey: bytes) -> bytes:
    """由私鑰導出未壓縮公鑰（65 bytes，0x04 + X(32) + Y(32)）。"""
    if len(privkey) != 32:
        raise ValueError("私鑰長度應為 32 bytes")
    pk = coincurve.PrivateKey(privkey)
    return pk.public_key.format(compressed=False)


def pubkey_to_tron_address(pubkey_uncompressed: bytes) -> Tuple[str, str]:
    """將未壓縮公鑰轉為 TRON 地址。
    回傳 (hex_address, base58_address)
    - hex_address: 以 0x41 開頭的 21 bytes 十六進位字串（不含 0x 前綴）
    - base58_address: 以 'T' 開頭的 Base58Check 字串
    """
    if len(pubkey_uncompressed) != 65 or pubkey_uncompressed[0] != 0x04:
        raise ValueError("未壓縮公鑰須為 65 bytes 且首位為 0x04")

    # 取未壓縮公鑰的 X||Y，共 64 bytes
    pubkey_xy = pubkey_uncompressed[1:]
    keccak = keccak_256(pubkey_xy)

    # 取 Keccak 的後 20 bytes，前綴 0x41（主網）
    addr20 = keccak[-20:]
    tron_bytes = b"\x41" + addr20

    # Base58Check：附加前 4 bytes 的雙 SHA-256 校驗碼
    checksum = sha256d(tron_bytes)[:4]
    b58 = base58.b58encode(tron_bytes + checksum).decode()

    return tron_bytes.hex(), b58


def privkey_to_tron_address(privkey: bytes) -> Tuple[str, str]:
    """私鑰直接轉 TRON 地址（hex 與 base58）。"""
    pub = privkey_to_pubkey_uncompressed(privkey)
    return pubkey_to_tron_address(pub)


def is_valid_tron_base58(addr: str) -> bool:
    """基本格式校驗：是否為 Base58 且解碼後長度/校驗碼符合 TRON 地址規格。"""
    try:
        raw = base58.b58decode(addr)
        if len(raw) != 25:
            return False
        body, checksum = raw[:-4], raw[-4:]
        return sha256d(body)[:4] == checksum and body[0] == 0x41
    except Exception:
        return False


__all__ = [
    "generate_privkey",
    "privkey_to_pubkey_uncompressed",
    "pubkey_to_tron_address",
    "privkey_to_tron_address",
    "is_valid_tron_base58",
]
