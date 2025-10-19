"""允許以 `python -m tron_vanity_experimental` 或 `python src/tron_vanity_experimental` 啟動 CLI。"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, "", "__main__"):
    package_root = Path(__file__).resolve().parent
    if str(package_root.parent) not in sys.path:
        sys.path.insert(0, str(package_root.parent))
    __package__ = "tron_vanity_experimental"

from .cli import run

run()
