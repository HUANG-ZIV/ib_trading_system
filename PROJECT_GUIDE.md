# IB Trading System - 專案完整說明

## 最後更新：2024-12-22

---

## 目錄

1. [專案概述](#專案概述)
2. [環境設定](#環境設定)
3. [專案結構](#專案結構)
4. [快速啟動](#快速啟動)
5. [系統架構](#系統架構)
6. [已完成功能](#已完成功能)
7. [配置說明](#配置說明)
8. [TWS 設定](#tws-設定)
9. [通知設定](#通知設定)
10. [常用命令](#常用命令)
11. [Git 操作](#git-操作)
12. [已知問題與解決方案](#已知問題與解決方案)
13. [下次對話開場白](#下次對話開場白)

---

## 專案概述

這是一個基於 Interactive Brokers API 的自動交易系統，使用 Python 開發。
- **GitHub**: https://github.com/HUANG-ZIV/ib_trading_system
- **開發者**: HUANG-ZIV
- **用途**: 外匯自動交易（可擴展至股票、期貨）

---

## 環境設定

| 項目 | 設定值 |
|------|--------|
| Python 版本 | **3.12**（不要用 3.14，與 ib_insync 不相容） |
| 虛擬環境路徑 | `/Users/zivhuang/Documents/VS code/venv312` |
| 專案路徑 | `/Users/zivhuang/Documents/VS code/ib_trading_system` |
| IB 端口 | 7497（Paper Trading） |
| 數據庫 | SQLite (`data_store/trading.db`) |

---

## 專案結構
```
ib_trading_system/
│
├── run_live.py                 # 🚀 主程式入口（外匯交易）
├── run_backtest.py             # 回測啟動腳本
├── run_test.py                 # 測試腳本
│
├── config/                     # ⚙️ 配置模組
│   ├── settings.py             # 全局配置（從環境變數載入）
│   ├── trading_modes.py        # 交易模式配置
│   └── symbols.py              # 交易標的定義
│
├── core/                       # 🔧 核心模組
│   ├── connection.py           # IB 連接管理（含自動重連）
│   ├── contracts.py            # 合約工廠（股票/外匯/期貨/商品）
│   ├── events.py               # 事件定義
│   └── event_bus.py            # 事件總線
│
├── data/                       # 📊 數據模組
│   ├── feed_handler.py         # 市場數據接收（即時/歷史）
│   ├── bar_aggregator.py       # K線聚合器
│   ├── cache.py                # 數據快取
│   └── database.py             # 數據庫操作（交易記錄）
│
├── strategies/                 # 📈 策略模組
│   ├── base.py                 # 策略基類（含預熱功能）
│   ├── registry.py             # 策略註冊器
│   └── examples/               # 範例策略
│       ├── sma_cross.py        # SMA 交叉策略
│       ├── test_strategy.py    # 測試策略
│       └── tick_scalper.py     # Tick 剝頭皮策略
│
├── risk/                       # 🛡️ 風控模組
│   ├── manager.py              # 風險管理器（含持倉同步）
│   ├── position_sizer.py       # 倉位計算器
│   └── circuit_breaker.py      # 熔斷機制
│
├── engine/                     # ⚡ 引擎模組
│   ├── strategy_engine.py      # 策略執行引擎
│   └── execution_engine.py     # 訂單執行引擎（含超時處理、OCA）
│
├── utils/                      # 🔨 工具模組
│   ├── logger.py               # 日誌配置
│   ├── market_hours.py         # 市場時間工具
│   ├── time_utils.py           # 時間工具
│   ├── notifier.py             # 通知服務（Telegram/Email）
│   └── performance.py          # 性能監控
│
├── backtest/                   # 📉 回測模組
│   ├── engine.py               # 回測引擎
│   └── data_loader.py          # 歷史數據載入
│
├── logs/                       # 📁 日誌目錄
├── data_store/                 # 📁 數據儲存目錄
│   └── trading.db              # SQLite 數據庫
│
├── .env                        # 環境變數（不上傳）
├── .env.example                # 環境變數範例
└── PROJECT_GUIDE.md            # 本文檔
```

---

## 快速啟動
```bash
# 1. 啟動虛擬環境
source "/Users/zivhuang/Documents/VS code/venv312/bin/activate"

# 2. 進入專案目錄
cd "/Users/zivhuang/Documents/VS code/ib_trading_system"

# 3. 確保 TWS 已啟動並登入

# 4. 執行系統
python run_live.py

# 5. 停止系統
# 按 Ctrl+C
```

---

## 系統架構

### 啟動流程
```
系統啟動
    │
    ▼
1. 連線 IB
    │
    ▼
2. 同步 IB 持倉 → RiskManager
    │
    ▼
3. 載入策略
    │
    ▼
4. 恢復策略持倉（從數據庫）
    │
    ▼
5. 策略預熱（載入歷史數據）
    │
    ▼
6. 訂閱即時數據
    │
    ▼
7. 開始主循環
    ├── 每秒：檢查訂單超時
    ├── 每 60 秒：輸出狀態
    └── 每 5 分鐘：同步持倉
```

### 訂單執行流程
```
策略信號 → 風控檢查 → 下單 → 成交 → 更新持倉 → 記錄數據庫
                ↓
            拒絕則跳過
```

### 停損/停利流程（OCA）
```
主訂單成交
    │
    ├── 停損單：STP  ┐
    │               ├── OCA Group（一個成交取消另一個）
    └── 停利單：LMT  ┘
```

---

## 已完成功能

### 優先級「高」

| 功能 | 說明 | 位置 |
|------|------|------|
| ✅ 啟動時同步持倉 | 從 IB 取得實際持倉同步到 RiskManager | `run_live.py` |
| ✅ 訂單狀態追蹤 | FillEvent 自動更新持倉 | `execution_engine.py` |
| ✅ 策略信號經風控 | 信號發出前經過 RiskManager 檢查 | `execution_engine.py` |
| ✅ 斷線自動重連 | 斷線後自動重連（最多 5 次） | `connection.py` |
| ✅ 數據庫記錄交易 | 所有成交記錄到 SQLite | `database.py` |

### 優先級「中」

| 功能 | 說明 | 位置 |
|------|------|------|
| ✅ 重啟恢復持倉 | 從數據庫恢復策略持倉狀態 | `run_live.py` |
| ✅ 持倉定期同步 | 每 5 分鐘比對 IB 與內部持倉 | `run_live.py` |
| ✅ 策略預熱 | 啟動時載入歷史數據 | `base.py` |
| ✅ 訂單超時處理 | 超時自動取消並通知策略 | `execution_engine.py` |
| ✅ OCA Group | 停損/停利互相取消 | `execution_engine.py` |
| ✅ 錯誤通知 | Telegram/Email 通知 | `notifier.py` |

---

## 配置說明

### run_live.py 配置
```python
# 持倉同步配置
POSITION_SYNC_INTERVAL = 300  # 秒（5 分鐘），設為 0 停用

# 訂單超時配置（秒），0 表示不超時
ORDER_TIMEOUT = {
    "MKT": 30,       # 市價單：30 秒
    "LMT": 300,      # 限價單：5 分鐘
    "STP": 0,        # 停損單：永不超時
    "STP_LMT": 0,    # 停損限價：永不超時
}

# 交易標的
LIVE_SYMBOLS = [
    "XAUUSD",    # 黃金
    "EUR/USD",   # 歐元/美元
    "GBP/USD",   # 英鎊/美元
    "USD/JPY",   # 美元/日圓
    "AUD/USD",   # 澳幣/美元
    "USD/CHF",   # 美元/瑞士法郎
]
```

### 策略預熱配置
```python
@dataclass
class StrategyConfig:
    # 預熱配置
    warmup_bars: int = 0                     # 需要的 K 線數量
    warmup_bar_size: str = "5 secs"          # K 線週期
    warmup_duration: str = ""                # 時間區間（如 "1 D"）
    warmup_what_to_show: str = "MIDPOINT"    # 數據類型
    warmup_required: bool = False            # 預熱失敗是否阻止啟動
```

### 停損/停利設定
```python
# 策略發信號時設定
signal = SignalEvent(
    ...
    stop_loss=4250.0,      # 停損價
    take_profit=4300.0,    # 停利價
)
```

---

## TWS 設定

| 設定項目 | 值 |
|----------|-----|
| Enable ActiveX and Socket Clients | ✅ 勾選 |
| Socket port | 7497 |
| Read-Only API | ❌ 不勾選 |
| 主API客戶ID | 1 |
| Allow connections from localhost only | ✅ 勾選 |

---

## 通知設定

### 環境變數（.env）
```bash
# Telegram（推薦）
TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrSTUvwxYZ
TELEGRAM_CHAT_ID=987654321

# Email
EMAIL_ENABLED=true
EMAIL_SMTP_HOST=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
EMAIL_FROM=your_email@gmail.com
EMAIL_TO=recipient@email.com
```

### 如何取得 Telegram Bot Token

1. 在 Telegram 搜尋 @BotFather
2. 發送 /newbot
3. 設定 bot 名稱
4. 取得 Bot Token
5. 搜尋你的 bot 並發送任意訊息
6. 訪問 `https://api.telegram.org/bot<TOKEN>/getUpdates`
7. 取得 chat_id

### 通知時機

| 情境 | 等級 |
|------|------|
| 系統啟動/關閉 | INFO |
| IB 連接斷開 | WARNING |
| 持倉不一致 | WARNING |
| IB 連接失敗 | ERROR |
| 嚴重持倉差異 | CRITICAL |
| 系統錯誤 | CRITICAL |

---

## 常用命令

### 系統操作
```bash
# 啟動虛擬環境
source "/Users/zivhuang/Documents/VS code/venv312/bin/activate"

# 進入專案
cd "/Users/zivhuang/Documents/VS code/ib_trading_system"

# 執行交易系統
python run_live.py

# 測試連接
python test_ib_connection.py

# 查看日誌
cat logs/trading_*.log
```

### 數據庫查詢
```python
from data.database import DatabaseManager
db = DatabaseManager("sqlite:///data_store/trading.db")

# 查詢交易記錄
trades = db.get_trades(strategy_id="test_strategy")
trades = db.get_trades(symbol="XAUUSD", days=7)

# 查詢未平倉持倉
positions = db.get_open_positions()
```

---

## Git 操作
```bash
# 推送更新
git add .
git commit -m "說明改了什麼"
git push

# 查看狀態
git status

# 查看歷史
git log --oneline
```

---

## 已知問題與解決方案

| 問題 | 解決方案 |
|------|----------|
| Python 3.14 不相容 | 使用 Python 3.12 |
| 股票數據需付費 | 改用外匯（免費） |
| TWS 連接超時 | 確認 API 設定正確，重啟 TWS |
| event loop 錯誤 | 使用 Python 3.12 + nest_asyncio |

---

## 下次對話開場白
```
我的 IB 交易系統專案在 https://github.com/HUANG-ZIV/ib_trading_system

環境：
- Python 3.12
- 虛擬環境：venv312
- IB TWS Paper Trading（端口 7497）
- 數據庫：SQLite (data_store/trading.db)

已完成功能：
- 啟動時同步持倉
- 策略信號經風控檢查
- 數據庫記錄交易
- 重啟恢復持倉
- 持倉定期同步（5分鐘）
- 策略預熱
- 訂單超時處理
- OCA（停損/停利互取消）
- Telegram/Email 通知

我想要...（說明你要做什麼）
```

---

## 聯絡資訊

- **GitHub**: https://github.com/HUANG-ZIV/ib_trading_system
- **Email**: ziv.yu.hsiang.huang@gmail.com
