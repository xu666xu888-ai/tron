# -*- coding: utf-8 -*-
"""允許在專案根目錄直接執行 `python -m tron_vanity_v3`。"""

from __future__ import annotations

import sys
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_SRC_PACKAGE = _PACKAGE_DIR.parent / "src" / "tron_vanity_v3"
parent = _SRC_PACKAGE.parent
if str(parent) not in sys.path:
    sys.path.insert(0, str(parent))

from tron_vanity_v3.cli import run

if __name__ == "__main__":
    run()
