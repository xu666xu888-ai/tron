# Window4 最佳化問題筆記（2025-10-14）

近期觀察：

- 目前單尾碼長跑時（例如 `tom888`）畫面顯示 `Window4(累積) = 停用`、`Window4(本批) = 停用`。
- 自適應門檻維持在 `32,768`，即使批次放大到 639,488 仍未觸發 Window4。
- Log (`~/.tron_vanity_worker.log`) 僅記錄工作啟動，並未記錄 Window4 相關告警，因此需要額外追蹤。

懷疑問題：

1. `gpu_addr._launch_batch` 裡的 `use_wnaf_this_batch` 條件可能過於保守；即便門檻自適應，也沒重新提升。
2. Window4 內核開啟後若失敗會自動回退並把 `_WNAF_BROKEN` 設為 `True`，但目前沒有記錄異常；需要確認是否曾經失敗一次後就永久停用。
3. `suffix_mod_params` 篩選命中後僅回傳索引，Window4 可能無命中就保持 `False`，需要在無命中情況也標註嘗試過。

建議實驗：

- 在 `gpu_addr._launch_batch` 中加上 logging，記錄每批 `cur`、`use_kernel`、`_WNAF_LAST_USED` 等狀況（可先用 `print` 或 `logger.debug`）。
- 強制啟用 Window4（設定 `VANITY_WNAF_MAX_BATCH` > 批次上限）驗證是否 kernel 仍可正常執行。
- 針對第一次失敗後 `_WNAF_BROKEN` 的行為，檢查是否需要自動重試或在 session 重新初始化期間復位。

參考指令：

```bash
# 檢查背景任務狀態與 Window4 門檻
python3 src/tron_vanity --attach

# 強制開啟 Window4 進行短測
VANITY_WNAF_MAX_BATCH=524288 python3 src/tron_vanity --suffix 8888 --no-monitor

# 加入 DEBUG logging（需要先 export）
export VANITY_LOG_LEVEL=DEBUG
python3 src_tron_vanity --suffix tom888 --no-monitor
```

> 註：背景會話儲存在 `~/.tron_vanity_session.json`，若要重新開始需先 `python3 src_tron_vanity --stop`。

---

## 明日對話建議 Prompt

```
我們昨天整理的 Window4 停用問題需要進一步調查。請協助：
1. 檢查 notes/window4_tuning_plan.md 中提到的 wnaf 條件與 `_WNAF_BROKEN` 旗標。
2. 新增適當的 debug log，記錄每批是否使用 Window4 及失敗原因。
3. 若確認 Window4 可用，調整自適應門檻策略，讓大批次能重新啟用 Window4。
```

使用此 prompt 開啟新的對話即可延續今天的工作。
