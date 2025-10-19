"""
性能預估模組：根據硬體配置估算地址生成速率與靚號搜尋所需時間。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

from .hardware_config import HardwareAdaptiveConfig


@dataclass(frozen=True)
class SpeedEstimate:
    """封裝硬體環境的吞吐量預估。"""

    backend: str
    addr_per_sec: float
    label: str
    notes: Tuple[str, ...]

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DifficultyEstimate:
    """封裝靚號搜尋難度與所需時間的推估。"""

    suffix_length: int
    attempts: int
    expected_seconds: float
    difficulty: str
    human_time: str
    feasible: bool

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


_DIFFICULTY_TIERS = {
    0: "瞬間",
    1: "簡單",
    2: "簡單",
    3: "中等",
    4: "中等",
    5: "困難",
    6: "極難",
}


def _format_duration(seconds: float) -> str:
    """將秒數轉換為易讀中文描述。"""

    if not math.isfinite(seconds):
        return "不可估計"
    if seconds < 1:
        return f"{seconds * 1000:.1f} 毫秒"
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    parts = []
    if days >= 1:
        parts.append(f"{int(days)} 天")
    if hours >= 1:
        parts.append(f"{int(hours)} 小時")
    if minutes >= 1 and len(parts) < 2:
        parts.append(f"{int(minutes)} 分")
    if not parts or len(parts) < 2:
        parts.append(f"{sec:.1f} 秒")
    return " ".join(parts)


def estimate_speed(config: HardwareAdaptiveConfig) -> SpeedEstimate:
    """
    根據偵測到的硬體配置估算地址生成速率。
    若配置附帶備註（例如 CPU 回退提示），將一併返回。
    """

    addr_per_sec = max(1.0, float(config.estimated_addr_per_sec))
    label = "GPU 加速" if config.backend == "GPU" else "CPU 模式"
    notes = config.notes
    if config.backend == "GPU" and not config.cupy_available:
        notes = notes + ("偵測到 GPU 但缺少 CuPy，請安裝後重啟程序",)
    return SpeedEstimate(
        backend=config.backend,
        addr_per_sec=addr_per_sec,
        label=label,
        notes=notes,
    )


def expected_attempts_for_suffix(length: int) -> int:
    """
    估算末幾碼靚號所需嘗試次數。
    採用 Base58 字元空間：每一碼有 58 種組合。
    """

    if length <= 0:
        return 1
    return int(pow(58, length))


def estimate_suffix_difficulty(length: int, config: HardwareAdaptiveConfig) -> DifficultyEstimate:
    """
    針對「末幾碼」靚號估算難度、預期嘗試次數與所需時間。
    """

    attempts = expected_attempts_for_suffix(length)
    speed = estimate_speed(config).addr_per_sec
    seconds = attempts / speed if speed > 0 else math.inf

    difficulty = _DIFFICULTY_TIERS.get(length, "極難")
    feasible = seconds <= 3600 * 24 * 7  # 超過 7 天視為不可行
    if not feasible and difficulty != "極難":
        difficulty = "極難"

    return DifficultyEstimate(
        suffix_length=length,
        attempts=attempts,
        expected_seconds=seconds,
        difficulty=difficulty,
        human_time=_format_duration(seconds),
        feasible=feasible,
    )


__all__ = [
    "SpeedEstimate",
    "DifficultyEstimate",
    "estimate_speed",
    "estimate_suffix_difficulty",
    "expected_attempts_for_suffix",
]

