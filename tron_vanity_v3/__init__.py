# -*- coding: utf-8 -*-
"""
頂層套件匯入器：讓 `python -m tron_vanity_v3` 可直接在專案根目錄啟動，
同時重用 `src/tron_vanity_v3` 內的完整模組實作。
"""

from __future__ import annotations

import sys
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_SRC_PACKAGE = _PACKAGE_DIR.parent / "src" / "tron_vanity_v3"

if _SRC_PACKAGE.is_dir():
    parent = _SRC_PACKAGE.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    __path__ = [str(_SRC_PACKAGE)]
else:  # pragma: no cover - 保底路徑不存在時
    __path__ = []

_globals = globals()
try:
    code = (_SRC_PACKAGE / "__init__.py").read_text(encoding="utf-8")
except FileNotFoundError:  # pragma: no cover
    raise ImportError("找不到 src/tron_vanity_v3 套件，請確認專案結構") from None
exec(compile(code, str(_SRC_PACKAGE / "__init__.py"), "exec"), _globals, _globals)
