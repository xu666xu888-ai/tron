"""
搜尋監控模組：負責統計與 Rich Live 介面更新。
目前提供最小可用版本，後續可擴充顯示更多指標與圖表。
"""

from __future__ import annotations

import math
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

try:
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ModuleNotFoundError:  # pragma: no cover - 若 rich 未安裝，CLI 會先自動補齊
    Console = None  # type: ignore
    Group = None  # type: ignore
    Layout = None  # type: ignore
    Live = None  # type: ignore
    Panel = None  # type: ignore
    Table = None  # type: ignore
    Text = None  # type: ignore

try:
    import psutil  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    psutil = None

_SPARK_CHARS = "▁▂▃▄▅▆▇█"


@dataclass
class RuntimeStats:
    """紀錄搜尋過程中的統計資料。"""

    target: str
    start_time: float = field(default_factory=time.time)
    checked: int = 0
    hits: int = 0
    speed_history: Deque[float] = field(default_factory=lambda: deque(maxlen=120))
    last_speed: float = 0.0

    def update_speed(self, addresses: int, elapsed: float) -> None:
        """更新速率資訊，不改變累計計數。"""

        if addresses <= 0:
            return
        if elapsed <= 0:
            elapsed = 1e-6
        speed = addresses / elapsed
        self.last_speed = speed
        self.speed_history.append(speed)

    @property
    def uptime(self) -> float:
        """回傳從開始至今的秒數。"""

        return max(0.0, time.time() - self.start_time)

    @property
    def average_speed(self) -> float:
        """回傳平均速率。"""

        if not self.speed_history:
            return self.last_speed
        return sum(self.speed_history) / len(self.speed_history)

    def sparkline(self) -> str:
        """以簡易 sparkline 呈現速度歷史。"""

        values = list(self.speed_history)
        if not values:
            return ""
        min_v = min(values)
        max_v = max(values)
        if math.isclose(max_v, min_v):
            return _SPARK_CHARS[-1] * len(values)
        scale = len(_SPARK_CHARS) - 1
        line = []
        for v in values:
            idx = int((v - min_v) / (max_v - min_v) * scale)
            line.append(_SPARK_CHARS[idx])
        return "".join(line)


class VanitySearchMonitor:
    """簡易 Rich Live 監控器。"""

    def __init__(self, console: Console, target: str, refresh_per_second: int = 4):
        if Live is None or Layout is None or Panel is None:
            raise RuntimeError("需安裝 rich 套件才能啟用監控介面")
        self._console = console
        self._target = target
        self._refresh_per_second = refresh_per_second
        self._stats = RuntimeStats(target=target)
        self._live: Optional[Live] = None
        self._last_metrics: Optional[dict] = None
        self._gpu_status_note: Optional[str] = None

        if psutil is not None:
            try:
                psutil.cpu_percent(interval=None)  # 初始化計算基準
            except Exception:  # pragma: no cover
                pass

    def _build_layout(self) -> Layout:
        """組裝監控畫面佈局。"""

        layout = Layout(name="root")
        layout.split(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )
        layout["header"].update(
            Panel(
                Text(f"目標靚號：{self._target}", style="bold magenta"),
                border_style="bright_magenta",
            )
        )
        layout["body"].update(self._build_body_panel())
        layout["footer"].update(
            Panel(Text("按 Ctrl+C 可優雅退出搜尋", style="yellow"), border_style="yellow")
        )
        return layout

    def _build_body_panel(self) -> Panel:
        """產出核心統計面板。"""

        stats = self._stats
        metrics = self._collect_metrics()

        info_table = Table.grid(padding=(0, 1))
        info_table.add_column(justify="right", style="cyan", width=12)
        info_table.add_column(style="white")
        info_table.add_row("已檢查", f"{stats.checked:,} addr")
        info_table.add_row("命中數", str(stats.hits))
        info_table.add_row("最近速率", f"{stats.last_speed:,.0f} addr/s")
        info_table.add_row("平均速率", f"{stats.average_speed:,.0f} addr/s")
        info_table.add_row("運行時間", f"{stats.uptime:.1f} 秒")
        if stats.speed_history:
            info_table.add_row("速率趨勢", stats.sparkline())

        metrics_table = Table.grid(padding=(0, 1))
        metrics_table.add_column(justify="right", style="cyan", width=12)
        metrics_table.add_column(style="white")
        if metrics.get("cpu_percent") is not None:
            metrics_table.add_row("CPU 使用", f"{metrics['cpu_percent']:.1f}%")
        if metrics.get("cpu_freq") is not None:
            metrics_table.add_row("CPU 頻率", f"{metrics['cpu_freq']:.1f} MHz")
        if metrics.get("memory_percent") is not None:
            metrics_table.add_row("記憶體使用", f"{metrics['memory_percent']:.1f}%")
        if metrics.get("gpu_entries"):
            for idx, item in enumerate(metrics["gpu_entries"], start=1):
                label = f"GPU{idx} 使用"
                metrics_table.add_row(label, f"{item['load']:.1f}% · {item['mem_used']:.1f}/{item['mem_total']:.1f} GiB")
                if item["temperature"] is not None:
                    metrics_table.add_row(f"GPU{idx} 溫度", f"{item['temperature']:.1f} °C")
        elif self._gpu_status_note:
            metrics_table.add_row("GPU 監控", self._gpu_status_note)

        combined = Table.grid(padding=1)
        combined.add_row(info_table, metrics_table)
        return Panel(combined, border_style="bright_cyan", title="搜尋狀態")

    def _collect_metrics(self) -> dict:
        """採集 CPU/GPU 指標。"""

        metrics = {
            "cpu_percent": None,
            "cpu_freq": None,
            "memory_percent": None,
            "gpu_entries": [],
        }

        if psutil is not None:
            try:
                metrics["cpu_percent"] = psutil.cpu_percent(interval=None)
                freq = psutil.cpu_freq()
                metrics["cpu_freq"] = freq.current if freq else None
                metrics["memory_percent"] = psutil.virtual_memory().percent
            except Exception:  # pragma: no cover
                pass

        gpu_entries = self._collect_gpu_metrics()
        if gpu_entries:
            metrics["gpu_entries"] = gpu_entries

        self._last_metrics = metrics
        return metrics

    def _collect_gpu_metrics(self) -> list:
        """使用 GPUtil 或 nvidia-smi 收集 GPU 指標。"""

        # 優先使用 GPUtil（若可用）
        try:
            import GPUtil  # type: ignore
        except ModuleNotFoundError:
            GPUtil = None  # type: ignore

        if GPUtil is not None:
            try:
                gpu_list = []
                for gpu in GPUtil.getGPUs():
                    gpu_list.append(
                        {
                            "load": (gpu.load or 0.0) * 100.0,
                            "mem_used": getattr(gpu, "memoryUsed", 0.0),
                            "mem_total": getattr(gpu, "memoryTotal", 0.0),
                            "temperature": getattr(gpu, "temperature", None),
                        }
                    )
                if gpu_list:
                    self._gpu_status_note = None
                    return gpu_list
            except Exception:  # pragma: no cover
                pass

        # 回退到 nvidia-smi
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0 or not proc.stdout.strip():
                self._gpu_status_note = "無法透過 GPUtil/nvidia-smi 取得 GPU 指標"
                return []
            entries = []
            for line in proc.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 4:
                    continue
                try:
                    load = float(parts[0])
                    mem_used = float(parts[1])
                    mem_total = float(parts[2])
                    temperature = float(parts[3])
                except ValueError:
                    continue
                    entries.append(
                        {
                            "load": load,
                            "mem_used": mem_used / 1024.0,  # MB -> GiB
                            "mem_total": mem_total / 1024.0,
                            "temperature": temperature,
                        }
                    )
            if entries:
                self._gpu_status_note = "使用 nvidia-smi 回退監控"
            return entries
        except Exception:  # pragma: no cover
            self._gpu_status_note = "無法透過 nvidia-smi 取得 GPU 指標"
            return []

    def start(self) -> None:
        """啟動 Live 刷新循環。"""

        if self._live is not None:
            return
        layout = self._build_layout()
        self._live = Live(
            layout,
            console=self._console,
            refresh_per_second=self._refresh_per_second,
            transient=False,
        )
        self._live.__enter__()

    def stop(self) -> None:
        """終止 Live 畫面。"""

        if self._live is not None:
            self._live.__exit__(None, None, None)
            self._live = None

    def update(self, *, checked: int, hits: int, last_batch: int, elapsed: float) -> None:
        """外部呼叫：更新統計並重繪畫面。"""

        self._stats.checked = checked
        self._stats.hits = hits
        self._stats.update_speed(last_batch, elapsed)
        if self._live is not None:
            layout = self._build_layout()
            self._live.update(layout, refresh=True)


__all__ = [
    "RuntimeStats",
    "VanitySearchMonitor",
]
