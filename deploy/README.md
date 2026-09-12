# GCP L4 一鍵部署

這條流程不需要保留 VM 或快照。從已提交的原始碼重建 Ubuntu 22.04、NVIDIA L4
驅動、Python 環境與 `tron-vanity` 指令。**執行會建立付費 VM 與 50 GB 磁碟。**

## 在 Mac 執行

前置條件：已安裝並登入 `gcloud`、Git、Python 3；GCP 專案已啟用 Compute Engine / IAP、
計費與 L4 配額。執行者需要建機、磁碟及防火牆權限、OS Login 管理員
(`roles/compute.osAdminLogin`) 及 IAP tunnel 存取權。使用專案的 `default` VPC / 子網路。

```bash
git clone https://github.com/xu666xu888-ai/tron.git
cd tron
bash deploy/gcloud_deploy.sh YOUR_PROJECT_ID asia-southeast1-c tron-vanity-l4
```

將 `YOUR_PROJECT_ID` 換成自己的 GCP project ID。既有副本先 `git pull --ff-only`。
腳本只部署當前 HEAD；來源與部署檔案如有尚未提交的變更，會拒絕執行。
如果同名 VM 已存在，也會拒絕，不會覆蓋或重啟它。

成功時最後會顯示 `DEPLOYMENT_READY=<commit>` 和 SSH 指令。這表示該次執行的
GPU 自檢通過、工具已安裝，且暫用外網 IP 已移除。不代表日後其他版本的驗證。

### 安裝內容與加速方式

- `g2-standard-4`：1 張 L4、4 vCPU、16 GB RAM，Ubuntu 22.04 / Python 3.10。
- 上傳 `git archive` 的已提交 `src/`、`deploy/`、`requirements.txt`，不傳家目錄或結果。
- 沿用 Google 官方 NVIDIA 安裝器的 binary / LTS 流程，已有可用驅動便跳過。
- 固定 CuPy 14.2.0、NumPy 2.2.6；其他套件沿用根目錄需求，並執行 `pip check`。
- 重跑安裝腳本時保留 venv / pip 快取；GPU 自檢會替 SSH 使用者預熱 CuPy 編譯快取。
- 用 CPU 與 tronpy 交叉檢查 GPU 命中；自檢不儲存或印出私鑰。

這主要減少手動步驟與版本差異，不承諾固定安裝分鐘數或更高搜尋速度。
OS 映像家族、官方 LTS 驅動安裝器及非固定的相依套件仍可能更新，並非位元級重現。
第一次仍須下載套件、安裝驅動及編譯 CUDA kernel；有需要時會重開機。
本次新增流程只做離線合約測試，尚未另外建立付費 VM 做完整乾淨部署驗證。

### 網路與權限

VM 不掛載服務帳戶。安裝時暫用外網 IP，供 apt / PyPI 下載，完成後移除；SSH 全程走 IAP。
若專案缺少 `allow-iap-ssh-tron-vanity`，會建立只允許 IAP 來源到 `iap-ssh` tag 的 TCP 22 規則。
同名規則如已存在，不會修改；必須確保它真的允許 IAP 連線。
**既有 default-allow-ssh 等較寬鬆的規則不會被收緊**，因此安裝期間的外網 IP
可能受那些規則允許入站。需嚴格隔離時，先由管理員配置合適 VPC / 防火牆。
不要透過此流程上傳帳戶憑證、SSH 私鑰或私鑰結果。

## SSH 裡使用

依部署完成後印出的指令登入；以下把專案換成自己的：

```bash
gcloud compute ssh tron-vanity-l4 --project=YOUR_PROJECT_ID \
  --zone=asia-southeast1-c --tunnel-through-iap
```

登入後：

```bash
nvidia-smi
tron-vanity --help
set +o history
tmux new -s vanity
```

在 tmux 裡再執行 `set +o history`（每個 Bash 工作階段各自設定），然後：

```bash
tron-vanity 88888 --timeout 900
```

自行替換尾號。按 `Ctrl-b` 再按 `d` 離開 tmux，程式仍繼續執行；重連後用
`tmux attach -t vanity` 回到畫面。關機會終止搜尋，不能從 RAM 進度續跑。
`--timeout` / `--max-attempts` 在整批 GPU 工作後檢查，可能超過指定上限一個批次。

**命中會自動顯示地址與私鑰**，通過本機 CPU / tronpy 驗證後，先保存
`~/tron-results/tron-*.json`（目錄 0700、檔案 0600），再印到終端。
別錄影、分享畫面、用 `tee` / 輸出重新導向或開啟 tmux 記錄。終端與 tmux 捲動紀錄
會保留畫面；`set +o history` 只控制命令歷史，不清除輸出。結果不會提交 Git。

## 安裝失敗／重試

失敗時會嘗試移除外網 IP 並停止本次新建 VM；不刪磁碟，**磁碟仍計費**。
若 API 權限失效、網路中斷、終端遭強制終止，清理不保證執行；務必查看 GCP Console。
Create API 失敗而狀態不明時，只有確認本次部署 token 的 VM 才會被停止。
L4 容量不足或配額不足無法靠腳本解決；可選有容量的區域，相關配額與子網路也須存在。

重試同一台：自行開機、暫加外網 IP（或提供可用的 Cloud NAT），透過 IAP 登入。
下列安裝步驟可重跑，不碰家目錄結果：

```bash
sudo bash /opt/tron-vanity/deploy/install_gpu_driver.sh
# 若 nvidia-smi 仍失敗，先 sudo reboot，重連後再繼續。
sudo bash /opt/tron-vanity/deploy/install_app.sh
PYTHONPATH=/opt/tron-vanity/src /opt/tron-vanity/.venv/bin/python /opt/tron-vanity/deploy/verify_gpu.py
```

完成後在 Mac 執行（失敗也需檢查是否仍有外網 IP）：

```bash
gcloud compute instances delete-access-config tron-vanity-l4 \
  --project=YOUR_PROJECT_ID --zone=asia-southeast1-c --access-config-name='External NAT'
```

若上傳程式包前就失敗，可在確認無任何需要的資料後刪除失敗的 VM / 磁碟，重新跑一鍵部署。

## 用完與費用

關機停止一般隨選 CPU / GPU 運算費，但磁碟仍計費。若不要保留環境，先自行安全保存
地址與私鑰，再刪 VM 與附掛磁碟。快照是獨立資源，需要另外刪除。刪除不等於關機，
沒有備份的結果無法恢復；此部署腳本不自動執行資料刪除，也不建立快照或 Cloud NAT。
不要刪除其他專案共用的網路、服務帳戶或防火牆來清理這一台機器。

## 離線驗證（不使用 GCP）

```bash
python3 -m unittest discover -s deploy/tests -v
bash -n deploy/gcloud_deploy.sh deploy/install_app.sh deploy/install_gpu_driver.sh deploy/tron-vanity
```

測試以假的 gcloud / 搜尋結果驗證流程與失敗處理；不代表 GPU 效能或乾淨部署驗收。
