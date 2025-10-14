"""
靚號搜尋引擎：封裝 GPU 與 CPU 管線，提供統一的尾碼搜尋介面。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .addr import privkey_to_tron_address
from .hardware_config import HardwareAdaptiveConfig

ProgressCallback = Callable[[int, int, int, float, Optional[Dict[str, float]]], None]


@dataclass(frozen=True)
class SearchHit:
    """單筆命中結果。"""

    address_hex: str
    address_base58: str
    privkey_hex: str


@dataclass(frozen=True)
class SearchResult:
    """搜尋結束的摘要。"""

    found: bool
    backend: str
    attempts: int
    elapsed: float
    hits: List[SearchHit] = field(default_factory=list)
    reason: str = "found"


class VanitySearchEngine:
    """整合 GPU/CPU 搜尋流程。"""

    def __init__(self, config: HardwareAdaptiveConfig):
        self._config = config

    def search_suffix(
        self,
        suffix: str,
        *,
        timeout: Optional[float] = None,
        max_attempts: Optional[int] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> SearchResult:
        """主要介面：搜尋指定尾碼。"""

        suffix = (suffix or "").strip()
        if not suffix:
            raise ValueError("suffix 不可為空白字串")

        suffix_upper = suffix.upper()
        start_time = time.time()

        if self._config.backend == "GPU" and self._config.cupy_available:
            hits, attempts, reason = self._search_gpu(
                suffix_upper,
                start_time=start_time,
                timeout=timeout,
                max_attempts=max_attempts,
                progress_callback=progress_callback,
            )
        else:
            hits, attempts, reason = self._search_cpu(
                suffix_upper,
                start_time=start_time,
                timeout=timeout,
                max_attempts=max_attempts,
                progress_callback=progress_callback,
            )

        elapsed = time.time() - start_time
        return SearchResult(
            found=bool(hits),
            backend=self._config.backend,
            attempts=attempts,
            elapsed=elapsed,
            hits=hits,
            reason=reason,
        )

    def _search_gpu(
        self,
        suffix: str,
        *,
        start_time: float,
        timeout: Optional[float],
        max_attempts: Optional[int],
        progress_callback: Optional[ProgressCallback],
    ) -> Tuple[List[SearchHit], int, str]:
        """GPU 搜尋流程。"""

        try:
            from .gpu_addr import generate_tron_addresses_gpu
        except Exception as exc:  # pragma: no cover - 若 GPU 模組不可用，改用 CPU
            return self._search_cpu(
                suffix,
                start_time=start_time,
                timeout=timeout,
                max_attempts=max_attempts,
                progress_callback=progress_callback,
            )

        dynamic_batches = self._config.default_batches or (16384,)
        stream_count = max(2, self._config.default_streams)
        max_cap = self._config.max_batch_size or 0
        if dynamic_batches:
            if max_cap > 0:
                eligible = [b for b in dynamic_batches if b <= max_cap]
                if eligible:
                    batch_size = eligible[-1]
                else:
                    batch_size = max_cap
            else:
                batch_size = dynamic_batches[-1]
        else:
            batch_size = max_cap if max_cap > 0 else 16384
        batch_size = max(8192, batch_size)
        if max_cap > 0:
            batch_size = min(batch_size, max_cap)

        attempts = 0
        hits_total = 0
        reason = "max_attempts"

        while True:
            if max_attempts is not None and attempts >= max_attempts:
                reason = "max_attempts"
                break
            if timeout is not None and time.time() - start_time > timeout:
                reason = "timeout"
                break

            remaining = None
            if max_attempts is not None:
                remaining = max_attempts - attempts
                if remaining <= 0:
                    reason = "max_attempts"
                    break
            current_count = batch_size
            if remaining is not None:
                current_count = min(current_count, remaining)
            current_count = max(1, min(current_count, self._config.max_batch_size or current_count))

            batch_begin = time.time()
            addresses, privs, gpu_stats = generate_tron_addresses_gpu(
                count=current_count,
                batch_size=current_count,
                suffix=suffix,
                max_hits=1,
                dynamic_batches=dynamic_batches,
                stream_count=stream_count,
                return_stats=True,
            )
            batch_elapsed = time.time() - batch_begin
            attempts += current_count
            hits_total += len(addresses)
            if progress_callback is not None:
                gpu_metrics = {
                    "batch_time": batch_elapsed,
                    "streams": stream_count,
                    **(gpu_stats or {}),
                }
                progress_callback(attempts, hits_total, current_count, max(batch_elapsed, 1e-6), gpu_metrics)

            if addresses:
                hits = [
                    SearchHit(
                        address_hex=addr_hex,
                        address_base58=addr_b58,
                        privkey_hex=pk.hex(),
                    )
                    for (addr_hex, addr_b58), pk in zip(addresses, privs)
                ]
                return hits, attempts, "found"

            if batch_size < (self._config.max_batch_size or batch_size):
                batch_size = min(batch_size * 2, self._config.max_batch_size or batch_size)

        return [], attempts, reason

    def _search_cpu(
        self,
        suffix: str,
        *,
        start_time: float,
        timeout: Optional[float],
        max_attempts: Optional[int],
        progress_callback: Optional[ProgressCallback],
    ) -> Tuple[List[SearchHit], int, str]:
        """CPU 搜尋流程（單執行緒，提供保底功能）。"""

        attempts = 0
        hits_total = 0
        batch_size = min(max(self._config.max_batch_size or 4096, 1024), 16384)
        reason = "max_attempts"

        while True:
            if max_attempts is not None and attempts >= max_attempts:
                reason = "max_attempts"
                break
            if timeout is not None and time.time() - start_time > timeout:
                reason = "timeout"
                break

            remaining = None
            if max_attempts is not None:
                remaining = max_attempts - attempts
                if remaining <= 0:
                    reason = "max_attempts"
                    break
            current_batch = batch_size
            if remaining is not None:
                current_batch = min(current_batch, remaining)

            batch_begin = time.time()
            processed = 0
            hit: Optional[SearchHit] = None

            while processed < current_batch:
                privkey = os.urandom(32)
                addr_hex, addr_b58 = privkey_to_tron_address(privkey)
                processed += 1
                attempts += 1
                if addr_b58.endswith(suffix):
                    hit = SearchHit(
                        address_hex=addr_hex,
                        address_base58=addr_b58,
                        privkey_hex=privkey.hex(),
                    )
                    hits_total += 1
                    break

            batch_elapsed = time.time() - batch_begin
            if progress_callback is not None:
                progress_callback(attempts, hits_total, processed, max(batch_elapsed, 1e-6), None)

            if hit is not None:
                return [hit], attempts, "found"

        return [], attempts, reason


__all__ = [
    "SearchHit",
    "SearchResult",
    "VanitySearchEngine",
]
