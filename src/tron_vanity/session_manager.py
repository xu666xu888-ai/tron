# -*- coding: utf-8 -*-
"""
背景任務會話管理：負責儲存/載入搜尋狀態，使 CLI 可在 SSH 斷線後重新接續。
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

SESSION_FILE = Path.home() / ".tron_vanity_session.json"
LOG_FILE = Path.home() / ".tron_vanity_worker.log"

_LOCK = threading.Lock()


def _atomic_write(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    os.replace(tmp, path)


def get_session_file() -> Path:
    return SESSION_FILE


def get_log_file() -> Path:
    return LOG_FILE


def load_session(path: Optional[Path] = None) -> Optional[dict]:
    target = path or SESSION_FILE
    if not target.exists():
        return None
    try:
        with target.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError:
        return None


def initialize_session(task: dict, path: Optional[Path] = None) -> None:
    state = {
        "created_at": time.time(),
        "status": "starting",
        "task": task,
        "pid": None,
        "stop_requested": False,
    }
    write_session(state, path)


def write_session(state: dict, path: Optional[Path] = None) -> None:
    target = path or SESSION_FILE
    with _LOCK:
        _atomic_write(state, target)


def update_session(
    mutator: Callable[[dict], dict],
    path: Optional[Path] = None,
) -> dict:
    target = path or SESSION_FILE
    with _LOCK:
        current = load_session(target) or {}
        mutated = mutator(dict(current)) or current
        _atomic_write(mutated, target)
        return mutated


def clear_session(path: Optional[Path] = None) -> None:
    target = path or SESSION_FILE
    with _LOCK:
        try:
            target.unlink()
        except FileNotFoundError:
            pass


def is_session_active(path: Optional[Path] = None) -> bool:
    state = load_session(path)
    if not state:
        return False
    pid = state.get("pid")
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def request_stop(path: Optional[Path] = None) -> Optional[dict]:
    state = load_session(path)
    if not state:
        return None

    def _set_stop(data: dict) -> dict:
        data["stop_requested"] = True
        data.setdefault("status", "running")
        data["status"] = data.get("status") or "running"
        return data

    return update_session(_set_stop, path)


def record_pid(pid: int, path: Optional[Path] = None) -> dict:
    return update_session(lambda data: {**data, "pid": pid}, path)
