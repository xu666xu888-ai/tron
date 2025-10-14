"""
產品級 CLI 主入口：整合系統檢測、自動安裝與性能預估。
後續將擴充實際靚號搜尋與監控流程。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, List
from collections import deque
import time
from rich.table import Table

if __package__ in (None, "", "__main__"):
    # 允許直接以 `python src/tron_vanity/cli.py` 執行
    repo_src = Path(__file__).resolve().parents[1]
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))
    __package__ = "tron_vanity"

from . import __version__
DEFAULT_CONFIG_LOCATIONS = (
    Path.cwd() / "tron_vanity_cli.json",
    Path.home() / ".tron_vanity_cli.json",
)
HISTORY_PATH = Path.home() / ".tron_vanity_history.jsonl"
SESSION_POLL_INTERVAL = 0.5
_LOG_LEVEL_ENV = "VANITY_LOG_LEVEL"
_LOG_KEEP_ENV = "VANITY_LOG_KEEP_BYTES"
_DEFAULT_LOG_KEEP = 512_000  # 512 KB


def _configure_logging() -> None:
    """根據環境變數設定 logging 等級。"""

    level_name = os.environ.get(_LOG_LEVEL_ENV, "").strip().upper()
    default_level = logging.INFO
    if level_name:
        level = getattr(logging, level_name, None)
        if not isinstance(level, int):
            level = default_level
    else:
        level = default_level

    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    else:
        root.setLevel(level)
    logging.getLogger("tron_vanity").setLevel(level)
    root.debug("啟用 logging，VANITY_LOG_LEVEL=%s", level_name or "INFO")


def _trim_worker_log(log_file: Path) -> None:
    """確保 worker 日誌不會無限制增長，只保留尾端區段。"""

    try:
        keep_raw = os.environ.get(_LOG_KEEP_ENV)
        keep_bytes = int(keep_raw) if keep_raw else _DEFAULT_LOG_KEEP
    except ValueError:
        keep_bytes = _DEFAULT_LOG_KEEP
    if keep_bytes <= 0:
        return
    try:
        if not log_file.exists():
            return
        size = log_file.stat().st_size
        if size <= keep_bytes:
            return
        with log_file.open("rb") as fh:
            fh.seek(-keep_bytes, os.SEEK_END)
            data = fh.read()
        with log_file.open("wb") as fh:
            fh.write(data)
    except Exception:  # pragma: no cover
        logging.getLogger(__name__).debug("Worker log trim 失敗，忽略", exc_info=True)


class StopRequested(Exception):
    """背景任務收到停止指令時拋出的例外。"""

    pass

CONFIG_KEYS = (
    "suffix",
    "timeout",
    "max_attempts",
    "cpu_only",
    "output",
    "auto_install",
    "auto_confirm",
    "monitor",
)


from .auto_installer import ensure_python_dependencies, plan_manual_actions
from .dependency_checker import DependencyReport, check_dependencies
from .hardware_config import HardwareAdaptiveConfig, detect_hardware_config
from .performance_estimator import (
    DifficultyEstimate,
    SpeedEstimate,
    estimate_speed,
    estimate_suffix_difficulty,
)
from .system_info import SystemInfo, collect_system_info
from .monitor import VanitySearchMonitor
from .search_engine import SearchHit, SearchResult, VanitySearchEngine
from .ui_components import (
    ascii_logo,
    build_dependency_table,
    build_difficulty_panel,
    build_manual_actions_panel,
    build_system_summary,
)
from .session_manager import (
    clear_session,
    get_log_file,
    get_session_file,
    initialize_session,
    is_session_active,
    load_session,
    record_pid,
    request_stop,
    update_session,
)

Console = None
Panel = None


def _import_rich() -> bool:
    """嘗試載入 rich，若失敗則回傳 False。"""

    global Console, Panel
    try:
        from rich.console import Console  # type: ignore
        from rich.panel import Panel  # type: ignore

        return True
    except ModuleNotFoundError:
        return False


def _prepare_console(allow_auto_install: bool = True) -> "Console":
    """確保 rich 可用並回傳 Console 實例。"""

    if not _import_rich():
        if not allow_auto_install:
            raise SystemExit("未安裝 rich，請先執行 `pip install rich` 或移除 --no-auto-install 選項。")
        print("未偵測到 rich，正在安裝 CLI 介面依賴...")
        report = ensure_python_dependencies(include_optional=True)
        if not _import_rich():
            raise SystemExit("rich 安裝失敗，請手動執行 `pip install rich` 後重試。")
        if report.manual_actions:
            for action in report.manual_actions:
                print(f"[提示] {action}")
    return Console()


def _ensure_config_structure(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(config) if isinstance(config, dict) else {}
    presets = cfg.get("presets")
    if not isinstance(presets, dict):
        presets = {}
    cfg["presets"] = presets
    cfg.setdefault("default_preset", None)
    return cfg


def _collect_settings_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "suffix": args.suffix,
        "timeout": args.timeout,
        "max_attempts": args.max_attempts,
        "cpu_only": bool(args.cpu_only),
        "output": args.output,
        "auto_install": not args.no_auto_install,
        "auto_confirm": bool(args.yes),
        "monitor": not args.no_monitor,
    }


def _render_dependency_section(console: "Console", report: DependencyReport) -> None:
    """輸出依賴檢測結果。"""

    console.print(build_dependency_table(report))


def _render_system_summary(
    console: "Console",
    system_info: SystemInfo,
    *,
    force_cpu: bool = False,
) -> Tuple[HardwareAdaptiveConfig, SpeedEstimate]:
    """輸出系統與硬體摘要。"""

    hw_config = detect_hardware_config(force_cpu=force_cpu)
    speed = estimate_speed(hw_config)
    console.print(build_system_summary(system_info, hw_config, speed))
    return hw_config, speed


def _render_difficulty(console: "Console", suffix: str, estimate: DifficultyEstimate) -> None:
    """顯示難度面板。"""

    console.print(build_difficulty_panel(suffix, estimate))


def _save_result(
    result: SearchResult,
    suffix: str,
    output_path: Optional[str],
    system_info: SystemInfo,
    config: HardwareAdaptiveConfig,
) -> Optional[Path]:
    """將命中結果寫入指定檔案（JSON Lines）。"""

    if not output_path or not result.hits:
        return None
    path = Path(output_path).expanduser()
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)
        path = path / "tron_vanity_hits.jsonl"
    record = {
        "suffix": suffix,
        "backend": result.backend,
        "attempts": result.attempts,
        "elapsed_seconds": result.elapsed,
        "address_hex": result.hits[0].address_hex,
        "address_base58": result.hits[0].address_base58,
        "privkey_hex": result.hits[0].privkey_hex,
        "system": {
            "os": f"{system_info.os_name} {system_info.os_version}",
            "cpu": system_info.cpu.model,
            "gpu": [
                {"name": gpu.name, "memory_gb": gpu.memory_total_gb, "cuda": gpu.cuda_version}
                for gpu in system_info.gpus
            ],
        },
        "config": {
            "backend": config.backend,
            "profile": config.profile,
            "default_batches": config.default_batches,
            "max_batch_size": config.max_batch_size,
        },
    }
    with path.open("a", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False)
        fh.write("\n")
    return path


def _merge_settings_into_args(
    args: argparse.Namespace,
    settings: Dict[str, Any],
    user_flags: Dict[str, bool],
    meta: Dict[str, bool],
) -> None:
    if not settings:
        return

    if args.suffix is None and settings.get("suffix"):
        args.suffix = str(settings["suffix"])
        meta["suffix_from_config"] = True

    if args.timeout is None and settings.get("timeout") is not None:
        args.timeout = _as_float(settings.get("timeout"))

    if args.max_attempts is None and settings.get("max_attempts") is not None:
        args.max_attempts = _as_int(settings.get("max_attempts"))

    if not args.cpu_only and settings.get("cpu_only"):
        args.cpu_only = True

    if args.output is None and settings.get("output"):
        args.output = str(settings["output"])

    if not user_flags.get("no_auto_install_cli"):
        auto_install = settings.get("auto_install")
        if auto_install is not None:
            args.no_auto_install = not bool(auto_install)
            meta["auto_install_set"] = True

    if not user_flags.get("yes_cli"):
        auto_confirm = settings.get("auto_confirm")
        if auto_confirm is not None:
            args.yes = bool(auto_confirm)
            meta["auto_confirm_set"] = True
            meta["yes_from_config"] = bool(auto_confirm)

    monitor_setting = settings.get("monitor")
    if monitor_setting is not None and monitor_setting is False:
        args.no_monitor = True


def _record_history(
    suffix: str,
    result: SearchResult,
    preset: Optional[str],
    config_path: Optional[Path],
    output_path: Optional[Path],
) -> None:
    history_entry = {
        "suffix": suffix,
        "status": "found" if result.found else result.reason,
        "address": result.hits[0].address_base58 if result.hits else None,
        "privkey": result.hits[0].privkey_hex if result.hits else None,
        "attempts": result.attempts,
        "elapsed": result.elapsed,
        "backend": result.backend,
        "preset": preset,
        "config": str(config_path) if config_path else None,
        "output_file": str(output_path) if output_path else None,
        "timestamp": time.time(),
    }
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("a", encoding="utf-8") as fh:
        json.dump(history_entry, fh, ensure_ascii=False)
        fh.write("\n")


def _display_history(console: "Console", limit: Optional[int]) -> None:
    if not HISTORY_PATH.exists():
        console.print("[yellow]尚無搜尋紀錄。[/yellow]")
        return
    entries = deque(maxlen=limit or 10)
    with HISTORY_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not entries:
        console.print("[yellow]尚無搜尋紀錄。[/yellow]")
        return
    table = Table(title="搜尋紀錄", header_style="bold cyan")
    table.add_column("時間")
    table.add_column("尾碼")
    table.add_column("狀態")
    table.add_column("地址")
    table.add_column("嘗試數")
    table.add_column("耗時(s)")
    table.add_column("Preset")
    for entry in reversed(entries):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry.get("timestamp", 0)))
        suffix = entry.get("suffix") or "-"
        status = entry.get("status") or "-"
        address = entry.get("address") or "-"
        attempts = f"{entry.get('attempts', 0):,}"
        elapsed = f"{entry.get('elapsed', 0):.2f}"
        preset = entry.get("preset") or "-"
        table.add_row(ts, suffix, status, address, attempts, elapsed, preset)
    console.print(table)


def _clear_history(console: "Console") -> None:
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()
        console.print("[green]已清除搜尋紀錄。[/green]")
    else:
        console.print("[yellow]目前沒有紀錄檔可清除。[/yellow]")


def _list_presets(console: "Console", config: Dict[str, Any]) -> None:
    config = _ensure_config_structure(config)
    presets = config.get("presets", {})
    if not presets:
        console.print("[yellow]尚未建立任何預設設定。[/yellow]")
        return
    table = Table(title="預設設定", header_style="bold cyan")
    table.add_column("名稱", style="white")
    table.add_column("尾碼", style="cyan")
    table.add_column("超時(s)", style="magenta")
    table.add_column("最大嘗試", style="magenta")
    table.add_column("模式", style="green")
    table.add_column("輸出", style="yellow")
    default_preset = config.get("default_preset")
    for name, settings in presets.items():
        suffix = settings.get("suffix", "-")
        timeout = settings.get("timeout")
        max_attempts = settings.get("max_attempts")
        cpu_only = "CPU" if settings.get("cpu_only") else "GPU"
        output = settings.get("output") or "-"
        label = f"{name} (預設)" if name == default_preset else name
        table.add_row(
            label,
            str(suffix or "-"),
            "-" if timeout is None else str(timeout),
            "-" if max_attempts is None else f"{max_attempts:,}",
            cpu_only,
            output,
        )
    console.print(table)


def _prompt_suffix(
    console: "Console",
    default: Optional[str],
    *,
    skip_prompt: bool,
) -> str:
    """互動詢問尾碼；若 skip_prompt 為 True 則直接回傳預設值。"""

    default = (default or "").strip()
    interactive = sys.stdin.isatty()
    if skip_prompt or not interactive:
        if not default and not interactive:
            raise SystemExit("偵測到非互動環境，請使用 --suffix 或預設設定指定尾碼。")
        if not default:
            return ""
        if not interactive and not skip_prompt:
            console.print(f"[yellow]非互動環境偵測，使用預設尾碼：{default}[/yellow]")
        return default

    while True:
        if default:
            prompt = (
                "[bold cyan]請輸入欲搜尋的末幾碼 (例如 88888)，直接 Enter 使用預設 "
                f"[white]{default}[/white]：[/] "
            )
        else:
            prompt = "[bold cyan]請輸入欲搜尋的末幾碼 (例如 88888)：[/] "
        value = console.input(prompt).strip()
        if value:
            return value
        if default:
            console.print(f"[green]已套用預設尾碼：{default}[/green]")
            return default
        console.print("[red]尾碼不可為空白，請重新輸入。[/red]")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        return lowered in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _load_cli_config(path: Optional[str]) -> Tuple[Dict[str, Any], Optional[Path]]:
    candidates: List[Path] = []
    if path:
        candidates.append(Path(path).expanduser())
    else:
        candidates.extend(DEFAULT_CONFIG_LOCATIONS)

    for candidate in candidates:
        if candidate.is_file():
            try:
                with candidate.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    return _ensure_config_structure(data), candidate
                raise ValueError("配置內容需為 JSON 物件")
            except Exception as exc:
                raise SystemExit(f"讀取配置檔案失敗：{candidate} ({exc})") from exc
    return _ensure_config_structure({}), None


def _apply_config(
    args: argparse.Namespace,
    config: Dict[str, Any],
    user_flags: Dict[str, bool],
) -> Dict[str, bool]:
    config = _ensure_config_structure(config)
    meta = {
        "suffix_from_config": False,
        "auto_confirm_set": False,
        "auto_install_set": False,
        "yes_from_config": False,
        "config_loaded": bool(config),
        "active_preset": None,
        "preset_from_cli": user_flags.get("preset_cli", False),
    }
    base_settings = {key: config.get(key) for key in CONFIG_KEYS}
    _merge_settings_into_args(args, base_settings, user_flags, meta)

    presets = config.get("presets", {})
    preset_name = args.preset or config.get("default_preset")
    if preset_name:
        preset_settings = presets.get(preset_name)
        if preset_settings is None:
            if user_flags.get("preset_cli"):
                raise SystemExit(f"找不到預設檔：{preset_name}")
        else:
            _merge_settings_into_args(args, preset_settings, user_flags, meta)
            meta["active_preset"] = preset_name
            args.preset = preset_name

    return meta


def _write_config(config: Dict[str, Any], target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, ensure_ascii=False, indent=2)
    return target


def _save_cli_config(
    config: Dict[str, Any],
    path: Optional[str],
    args: argparse.Namespace,
    *,
    preset: Optional[str] = None,
    set_default: bool = False,
) -> Path:
    cfg = _ensure_config_structure(config)
    target = Path(path).expanduser() if path else DEFAULT_CONFIG_LOCATIONS[-1]
    settings = _collect_settings_from_args(args)
    if preset:
        cfg["presets"][preset] = settings
        if set_default or cfg.get("default_preset") is None:
            cfg["default_preset"] = preset
    else:
        for key, value in settings.items():
            cfg[key] = value
        if set_default and args.preset:
            cfg["default_preset"] = args.preset
    return _write_config(cfg, target)


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRON 靚號地址生成 CLI")
    parser.add_argument(
        "--suffix",
        type=str,
        default="6666",
        help="直接指定欲搜尋的 Base58 尾碼（例如 88888）",
    )
    parser.add_argument("--timeout", type=float, help="設定搜尋逾時秒數")
    parser.add_argument("--max-attempts", type=int, help="限制最多嘗試次數")
    parser.add_argument("--cpu-only", action="store_true", help="強制使用 CPU 模式搜尋")
    parser.add_argument("--output", type=str, help="將命中結果追加寫入檔案（JSON Lines）；可為資料夾")
    parser.add_argument("--no-auto-install", action="store_true", help="避免自動安裝缺失的 Python 套件")
    parser.add_argument("--yes", action="store_true", help="跳過互動確認並立即開始搜尋")
    parser.add_argument("--no-confirm", dest="yes", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-monitor", action="store_true", help="停用 Rich 監控介面（純文字輸出）")
    parser.add_argument("--preset", type=str, help="套用指定的預設設定名稱")
    parser.add_argument("--save-preset", type=str, help="將目前參數儲存為指定名稱的預設設定")
    parser.add_argument("--list-presets", action="store_true", help="列出所有可用預設設定")
    parser.add_argument("--set-default-preset", type=str, help="設定預設預設設定名稱")
    parser.add_argument("--config", type=str, help="指定配置檔路徑（JSON）")
    parser.add_argument(
        "--show-history",
        nargs="?",
        type=int,
        const=10,
        help="顯示最近的搜尋紀錄，可指定要顯示的筆數（預設 10）",
    )
    parser.add_argument("--clear-history", action="store_true", help="清除搜尋紀錄檔")
    parser.add_argument(
        "--save-config",
        nargs="?",
        const="",
        help="將目前參數存成配置檔；若未提供路徑，預設寫入 ~/.tron_vanity_cli.json",
    )
    parser.add_argument("--attach", action="store_true", help="連接背景搜尋任務的即時狀態")
    parser.add_argument("--stop", action="store_true", help="要求目前的背景搜尋任務停止")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--session-file", type=str, help=argparse.SUPPRESS)
    return parser.parse_args([] if argv is None else list(argv))


def _start_background_worker(session_path: Path) -> int:
    """啟動背景工作進程，並將輸出寫入共用日誌。"""

    cmd = [sys.executable, "-m", "tron_vanity.cli", "--worker", "--session-file", str(session_path)]
    env = os.environ.copy()
    src_root = Path(__file__).resolve().parents[1]  # 指向 .../src
    env_py = env.get("PYTHONPATH")
    if env_py:
        env["PYTHONPATH"] = f"{src_root}{os.pathsep}{env_py}"
    else:
        env["PYTHONPATH"] = str(src_root)
    log_file = get_log_file()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    _trim_worker_log(log_file)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    state = load_session(session_path) or {}
    suffix = state.get("task", {}).get("suffix", "-")
    with log_file.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"[{timestamp}] launch worker for suffix={suffix}\n")
        log_handle.flush()
        proc = subprocess.Popen(
            cmd,
            start_new_session=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            close_fds=True,
            env=env,
        )
    return proc.pid


def _stop_session(console: "Console") -> int:
    session_path = get_session_file()
    state = load_session(session_path)
    if not state or not state.get("pid"):
        console.print("[yellow]目前沒有背景搜尋任務。[/yellow]")
        clear_session(session_path)
        return 0
    if not is_session_active(session_path):
        console.print("[yellow]背景搜尋任務已結束。[/yellow]")
        clear_session(session_path)
        return 0
    request_stop(session_path)
    console.print("[cyan]已送出停止請求，可稍後使用 --attach 查看狀態。[/cyan]")
    return 0


def _wait_for_session_completion(console: "Console", session_path: Path, timeout: float = 60.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if not is_session_active(session_path):
            clear_session(session_path)
            console.print("[green]背景搜尋任務已停止，可重新設定新的任務。[/green]")
            return True
        time.sleep(0.5)
    console.print("[yellow]背景任務仍在執行，請稍後使用 --attach 查看或再次嘗試止任務。[/yellow]")
    return False


def _handle_active_session(
    console: "Console",
    args: argparse.Namespace,
    session_path: Optional[Path] = None,
) -> Optional[int]:
    """偵測並處理既有背景任務，視使用者選擇決定後續流程。"""

    session_path = session_path or get_session_file()
    existing_state = load_session(session_path)
    if not existing_state:
        return None
    pid = existing_state.get("pid")
    if not pid or not is_session_active(session_path):
        clear_session(session_path)
        return None

    task_existing = existing_state.get("task", {})
    suffix_existing = task_existing.get("suffix", "-")
    status_existing = existing_state.get("status", "running")
    console.print(
        f"[yellow]偵測到背景搜尋任務正在執行：PID {pid}，目標尾碼 {suffix_existing}，狀態 {status_existing}。[/yellow]"
    )

    if args.yes or not sys.stdin.isatty():
        console.print("[cyan]已自動接續顯示背景任務，可使用 --stop 終止。[/cyan]")
        return _attach_session(console)

    while True:
        choice = console.input(
            "[bold cyan]選擇操作：[/bold cyan]"
            "[A] 接續顯示 / [R] 停止並重新設定 / [Q] 取消："
        ).strip().lower()
        if choice in {"", "a", "attach"}:
            return _attach_session(console)
        if choice in {"r", "replace", "s", "stop"}:
            _stop_session(console)
            if _wait_for_session_completion(console, session_path):
                return None
            return 0
        if choice in {"q", "cancel", "n"}:
            console.print("[yellow]已取消操作，背景任務持續執行。[/yellow]")
            return 0
        console.print("[red]無效的選項，請重新輸入。[/red]")


def _build_search_result_from_state(state: dict) -> Optional[SearchResult]:
    result_data = state.get("result") or {}
    if not result_data:
        return None
    hits_raw = result_data.get("hits") or []
    hits = [
        SearchHit(
            address_hex=item.get("address_hex", ""),
            address_base58=item.get("address_base58", ""),
            privkey_hex=item.get("privkey_hex", ""),
        )
        for item in hits_raw
    ]
    return SearchResult(
        found=bool(result_data.get("found", False)),
        backend=result_data.get("backend", "GPU"),
        attempts=int(result_data.get("attempts", state.get("checked", 0))),
        elapsed=float(result_data.get("elapsed", state.get("elapsed_total", 0.0))),
        hits=hits,
        reason=result_data.get("reason", "found"),
    )


def _present_session_result(console: "Console", state: dict) -> None:
    task = state.get("task", {})
    suffix = task.get("suffix", "-")
    status = state.get("status", "unknown")
    console.rule("搜尋結果")

    if status == "completed":
        result = _build_search_result_from_state(state)
        if result and result.found and result.hits:
            summary = (
                f"[bold green]命中靚號！[/bold green]\n"
                f"後端：{result.backend}\n"
                f"耗時：{result.elapsed:.2f} 秒\n"
                f"嘗試次數：約 {result.attempts:,} 次\n"
                f"Base58 地址：{result.hits[0].address_base58}\n"
                f"HEX 地址：{result.hits[0].address_hex}\n"
                f"私鑰 (HEX)：{result.hits[0].privkey_hex}"
            )
            console.print(Panel(summary, border_style="green", title="成功"))
            system_info_obj = collect_system_info()
            hw_config_obj = detect_hardware_config(force_cpu=bool(task.get("cpu_only")))
            output_path = _save_result(result, suffix, task.get("output"), system_info_obj, hw_config_obj)
            if output_path:
                console.print(f"[green]已將結果寫入：{output_path}[/green]")
            recap_preset = task.get("preset") or task.get("save_preset")
            config_path_str = task.get("config_path")
            config_path_obj = Path(config_path_str).expanduser() if config_path_str else None
            save_preset_name = task.get("save_preset")
            set_default = task.get("set_default_preset")
            config_data = task.get("config_data") or {}
            args_settings = task.get("args_settings") or {}
            if save_preset_name and config_data and args_settings:
                args_namespace = argparse.Namespace(**args_settings)
                preset_path = _save_cli_config(
                    config_data,
                    config_path_str,
                    args_namespace,
                    preset=save_preset_name,
                    set_default=(set_default == save_preset_name),
                )
                console.print(f"[green]已更新預設設定：{save_preset_name}（檔案：{preset_path}）[/green]")
                if set_default == save_preset_name:
                    console.print(f"[cyan]已同步將 {save_preset_name} 設為預設設定。[/cyan]")
                config_path_obj = preset_path
                recap_preset = save_preset_name
            _record_history(suffix, result, recap_preset, config_path_obj, output_path)
        else:
            reason_map = {
                "timeout": "已達設定的逾時限制",
                "max_attempts": "已達最大嘗試次數限制",
                "found": "搜尋完成",
            }
            reason_text = reason_map.get(result.reason if result else "", "未命中")
            attempts_val = result.attempts if result else int(state.get("checked", 0))
            elapsed_val = result.elapsed if result else float(state.get("elapsed_total", 0.0))
            console.print(
                Panel(
                    f"[yellow]未命中靚號。[/yellow]\n"
                    f"原因：{reason_text}\n"
                    f"耗時：{elapsed_val:.2f} 秒\n"
                    f"嘗試次數：約 {attempts_val:,} 次",
                    border_style="yellow",
                    title="未命中",
                )
            )
            if result:
                _record_history(suffix, result, task.get("preset"), None, None)
    elif status == "stopped":
        console.print("[yellow]背景搜尋已依要求停止。[/yellow]")
    elif status == "error":
        console.print(f"[red]背景搜尋發生錯誤：{state.get('error_message', '未知錯誤')}[/red]")
        console.print(f"[yellow]請檢視日誌：{get_log_file()}[/yellow]")
    else:
        console.print(f"[yellow]背景搜尋已結束，狀態：{status}[/yellow]")


def _attach_session(console: "Console") -> int:
    session_path = get_session_file()
    state = load_session(session_path)
    if not state:
        console.print("[yellow]目前沒有背景搜尋任務。[/yellow]")
        return 0
    task = state.get("task", {})
    suffix = task.get("suffix", "-")
    monitor = VanitySearchMonitor(console, suffix)
    monitor.start()
    last_state = state
    try:
        while True:
            state = load_session(session_path)
            if not state:
                break
            if state.get("start_time"):
                monitor.set_start_time(state["start_time"])
            metrics = state.get("metrics") or {}
            last_batch = state.get("last_batch", metrics.get("current_batch", 0))
            elapsed = metrics.get("batch_time", state.get("batch_time", 1.0))
            if not elapsed:
                elapsed = 1e-6
            monitor.update(
                checked=int(state.get("checked", 0)),
                hits=int(state.get("hits", 0)),
                last_batch=int(last_batch),
                elapsed=float(elapsed),
                metrics=metrics,
            )
            last_state = state
            status = state.get("status", "running")
            if status in {"completed", "stopped", "error", "timeout", "max_attempts"}:
                break
            time.sleep(SESSION_POLL_INTERVAL)
    except KeyboardInterrupt:
        monitor.stop()
        console.print("\n[yellow]已離開即時監控，背景任務仍持續執行。[/yellow]")
        return 130
    finally:
        monitor.stop()

    if last_state:
        _present_session_result(console, last_state)
    else:
        console.print("[yellow]背景任務狀態無法讀取。[/yellow]")
    clear_session(session_path)
    return 0


def _worker_main(args: argparse.Namespace) -> int:
    session_path = Path(args.session_file) if args.session_file else get_session_file()
    state = load_session(session_path)
    if not state:
        print("[worker] session file not found")
        return 1
    task = state.get("task", {})
    suffix = task.get("suffix")
    if not suffix:
        print("[worker] suffix missing")
        return 1

    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    system_info = collect_system_info()
    hw_config = detect_hardware_config(force_cpu=bool(task.get("cpu_only")))
    engine = VanitySearchEngine(hw_config)
    timeout = task.get("timeout")
    max_attempts = task.get("max_attempts")

    start_time = time.time()

    def _init_state(data: dict) -> dict:
        data.setdefault("task", task)
        data["pid"] = os.getpid()
        data["status"] = "running"
        data["start_time"] = start_time
        data["system_info"] = system_info.as_dict()
        data["hardware"] = {
            "backend": hw_config.backend,
            "profile": hw_config.profile,
            "default_batches": list(hw_config.default_batches),
            "max_batch_size": hw_config.max_batch_size,
        }
        data.setdefault("metrics", {})
        data.setdefault("checked", 0)
        data.setdefault("hits", 0)
        data.setdefault("last_batch", 0)
        data.setdefault("batch_time", 0.0)
        data["elapsed_total"] = 0.0
        return data

    update_session(_init_state, session_path)

    def _progress(checked: int, hits: int, last_batch: int, elapsed: float, metrics: Optional[Dict[str, float]]) -> None:
        def _mutate(data: dict) -> dict:
            data.setdefault("task", task)
            data["pid"] = os.getpid()
            data["status"] = "running"
            data["checked"] = checked
            data["hits"] = hits
            data["last_batch"] = last_batch
            data["batch_time"] = elapsed
            data["elapsed_total"] = time.time() - start_time
            data["metrics"] = metrics or {}
            data["last_update"] = time.time()
            return data

        updated = update_session(_mutate, session_path)
        if updated.get("stop_requested"):
            raise StopRequested()

    try:
        result = engine.search_suffix(
            suffix,
            timeout=timeout,
            max_attempts=max_attempts,
            progress_callback=_progress,
        )
        hits_payload = [
            {
                "address_hex": hit.address_hex,
                "address_base58": hit.address_base58,
                "privkey_hex": hit.privkey_hex,
            }
            for hit in result.hits
        ]

        def _complete(data: dict) -> dict:
            data["status"] = "completed" if result.found else result.reason
            data["result"] = {
                "found": result.found,
                "backend": result.backend,
                "attempts": result.attempts,
                "elapsed": result.elapsed,
                "hits": hits_payload,
                "reason": result.reason,
            }
            data["checked"] = result.attempts
            data["hits"] = len(result.hits)
            data["elapsed_total"] = result.elapsed
            data["stop_requested"] = False
            data["last_update"] = time.time()
            return data

        update_session(_complete, session_path)
    except StopRequested:
        def _stopped(data: dict) -> dict:
            data["status"] = "stopped"
            data["stop_requested"] = False
            data["last_update"] = time.time()
            return data

        update_session(_stopped, session_path)
    except Exception as exc:  # pragma: no cover - 以狀態檔回報錯誤
        def _error(data: dict) -> dict:
            data["status"] = "error"
            data["error_message"] = str(exc)
            data["stop_requested"] = False
            data["last_update"] = time.time()
            return data

        update_session(_error, session_path)
        raise

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 主流程。"""

    _configure_logging()
    args = _parse_args(argv)
    if args.worker:
        return _worker_main(args)

    auto_install = not args.no_auto_install
    console = _prepare_console(allow_auto_install=auto_install)

    if args.attach:
        return _attach_session(console)
    if args.stop:
        return _stop_session(console)

    user_flags = {
        "yes_cli": bool(args.yes),
        "no_auto_install_cli": bool(args.no_auto_install),
        "preset_cli": args.preset is not None,
    }
    session_path = get_session_file()
    early_session_result = _handle_active_session(console, args, session_path)
    if early_session_result is not None:
        return early_session_result

    config, config_path = _load_cli_config(args.config)
    meta = _apply_config(args, config, user_flags)
    config = _ensure_config_structure(config)

    pending_default_preset: Optional[str] = None

    if args.list_presets:
        _list_presets(console, config)
        return 0

    if args.set_default_preset:
        preset_name = args.set_default_preset
        if preset_name not in config.get("presets", {}):
            if args.save_preset == preset_name:
                pending_default_preset = preset_name
            else:
                console.print(f"[red]找不到預設設定：{preset_name}[/red]")
                return 1
        else:
            config["default_preset"] = preset_name
            target_path = Path(args.config).expanduser() if args.config else (config_path or DEFAULT_CONFIG_LOCATIONS[-1])
            _write_config(config, target_path)
            console.print(f"[green]已將預設預設設定設為：{preset_name}[/green]")
            config_path = target_path

    if args.clear_history:
        _clear_history(console)
        return 0

    if args.show_history is not None:
        _display_history(console, args.show_history)
        return 0

    console.clear()
    if config_path:
        console.print(f"[cyan]載入 CLI 設定：{config_path}[/cyan]")
    console.print(ascii_logo(__version__), justify="center")
    console.rule("系統檢測")

    with console.status("[green]正在收集系統資訊...[/]"):
        system_info = collect_system_info()

    with console.status("[green]正在檢查依賴...[/]"):
        dep_report = check_dependencies()

    manual_actions: List[str] = plan_manual_actions(dep_report)
    if auto_install and not dep_report.all_satisfied():
        console.print("[yellow]偵測到缺失依賴，啟動自動安裝...[/yellow]")
        install_report = ensure_python_dependencies(include_optional=False)
        dep_report = install_report.dependency_report
        manual_actions = install_report.manual_actions or plan_manual_actions(dep_report)

    _render_dependency_section(console, dep_report)
    if manual_actions:
        console.print(build_manual_actions_panel(manual_actions))
    required_missing = [pkg.display_name for pkg in dep_report.python_packages if pkg.required and not pkg.installed]
    optional_missing = [pkg.display_name for pkg in dep_report.python_packages if (not pkg.required) and (not pkg.installed)]
    required_missing += [tool.name for tool in dep_report.system_tools if not tool.installed]
    if required_missing:
        console.print(
            f"[red]仍缺少必要依賴：{', '.join(required_missing)}，請依上方指引補齊後再進行高強度測試。[/red]"
        )
    elif optional_missing:
        console.print(
            f"[yellow]以下為選用功能，可視需求安裝：{', '.join(optional_missing)}。[/yellow]"
        )
    hw_config, speed = _render_system_summary(console, system_info, force_cpu=args.cpu_only)
    if meta.get("active_preset"):
        console.print(f"[cyan]套用預設設定：{meta['active_preset']}[/cyan]")

    console.rule("靚號設定")
    skip_suffix_prompt = args.yes and not meta.get("yes_from_config", False)
    suffix = _prompt_suffix(console, args.suffix, skip_prompt=skip_suffix_prompt)
    suffix = suffix.strip()
    if not suffix:
        console.print("[red]未提供尾碼，無法開始搜尋。[/red]")
        return 1
    args.suffix = suffix

    difficulty = estimate_suffix_difficulty(len(suffix), hw_config)
    _render_difficulty(console, suffix, difficulty)

    if args.save_config is not None:
        save_path = _save_cli_config(config, args.save_config or None, args)
        console.print(f"[green]已將設定寫入：{save_path}[/green]")
        config_path = save_path

    console.rule("開始搜尋")
    skip_confirm = args.yes and not meta.get("yes_from_config", False)
    if not skip_confirm:
        if not sys.stdin.isatty():
            console.print("[yellow]非互動環境偵測，已自動開始搜尋。[/yellow]")
        else:
            confirmation = console.input("[bold green]按 Enter 開始搜尋，輸入 N 取消：[/] ").strip().lower()
            if confirmation in {"n", "no"}:
                console.print("[bold yellow]已取消搜尋。[/bold yellow]")
                return 0

    if args.cpu_only and hw_config.backend != "CPU":
        hw_config = detect_hardware_config(force_cpu=True)

    console.print("[cyan]初始設定完畢，背景搜尋即將啟動…[/cyan]")

    config_snapshot = json.loads(json.dumps(config))
    task_payload = {
        "suffix": suffix,
        "timeout": args.timeout,
        "max_attempts": args.max_attempts,
        "cpu_only": bool(args.cpu_only),
        "output": args.output,
        "preset": meta.get("active_preset") or args.preset,
        "config_path": str(config_path) if config_path else None,
        "save_preset": args.save_preset,
        "set_default_preset": pending_default_preset,
        "config_data": config_snapshot,
        "args_settings": _collect_settings_from_args(args),
    }

    clear_session(session_path)
    initialize_session(task_payload, session_path)
    pid = _start_background_worker(session_path)
    record_pid(pid, session_path)
    console.print(f"[cyan]已啟動背景搜尋任務（PID: {pid}）。[/cyan]")
    console.print(f"[cyan]背景日誌：{get_log_file()}[/cyan]")
    if args.no_monitor:
        console.print("[yellow]背景任務將持續執行，可隨時執行 `python -m tron_vanity --attach` 查看進度。[/yellow]")
        return 0

    return _attach_session(console)


def _signal_handler(sig, frame) -> None:
    """優雅處理 Ctrl+C。"""

    print("\n[CLI] 已接收中斷訊號，結束程序。")
    sys.exit(0)


def run() -> None:
    """提供 `python -m tron_vanity.cli` 的入口。"""

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    exit_code = main(sys.argv[1:])
    raise SystemExit(exit_code)


if __name__ == "__main__":
    run()
