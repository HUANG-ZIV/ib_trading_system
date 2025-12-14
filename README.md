# IB Trading System

基於 Interactive Brokers (IB) API 的自動化交易系統，採用事件驅動架構，支援日內高頻交易、低頻交易及波段交易策略。

## 專案簡介

本系統提供完整的自動化交易解決方案，透過 `ib_insync` 連接 IB TWS 或 Gateway，實現即時數據接收、策略執行、訂單管理及風險控制。系統採用模組化設計，可輕鬆擴展自定義策略。

### 核心架構

```
┌─────────────────────────────────────────────────────────────┐
│                      Event Bus (事件總線)                    │
└──────────┬───────────────┬───────────────┬─────────────────┘
           │               │               │
     ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
     │   Data    │   │ Strategy  │   │ Execution │
     │  Handler  │   │  Engine   │   │  Engine   │
     └───────────┘   └───────────┘   └───────────┘
           │               │               │
           │         ┌─────▼─────┐         │
           │         │   Risk    │         │
           │         │  Manager  │         │
           │         └───────────┘         │
           │                               │
     ┌─────▼───────────────────────────────▼─────┐
     │             IB Connection                  │
     │        (TWS / IB Gateway)                  │
     └────────────────────────────────────────────┘
```

## 特性列表

### 連接管理
- ✅ 支援 TWS 和 IB Gateway 連接
- ✅ 自動斷線重連機制
- ✅ 連接狀態事件通知

### 數據處理
- ✅ 即時 Tick 數據訂閱
- ✅ 即時 Bar 數據訂閱
- ✅ Tick 轉 Bar 聚合器（支援多時間週期）
- ✅ 歷史數據下載
- ✅ 內存快取（高頻交易優化）
- ✅ 數據持久化（SQLite/DuckDB）

### 策略引擎
- ✅ 事件驅動架構
- ✅ 策略基類與生命週期管理
- ✅ 策略註冊器
- ✅ 同時運行多策略
- ✅ 支援高頻 Tick 級別策略
- ✅ 支援低頻 Bar 級別策略

### 訂單執行
- ✅ 市價單 / 限價單 / 停損單
- ✅ 括號訂單（Bracket Order）
- ✅ OCO 訂單
- ✅ 訂單狀態追蹤
- ✅ 成交回報處理

### 風險管理
- ✅ 每日虧損上限
- ✅ 單一標的持倉上限
- ✅ 總曝險控制
- ✅ 倉位計算器（固定/百分比/ATR）
- ✅ 熔斷機制（連續虧損自動停止）

### 監控通知
- ✅ 結構化日誌（Loguru）
- ✅ Telegram 即時通知
- ✅ Email 告警
- ✅ 性能監控

### 回測
- ✅ 歷史數據回測引擎
- ✅ 績效指標計算
- ✅ 權益曲線生成

## 安裝步驟

### 前置需求

- Python 3.10+
- Interactive Brokers 帳戶
- TWS 或 IB Gateway

### 1. 克隆專案

```bash
git clone https://github.com/yourusername/ib_trading_system.git
cd ib_trading_system
```

### 2. 建立虛擬環境

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. 安裝依賴

```bash
pip install -r requirements.txt
```

### 4. 配置環境變數

```bash
cp .env.example .env
```

編輯 `.env` 文件，設定你的配置：

```env
# IB 連接設定
IB_HOST=127.0.0.1
IB_PORT=7497          # TWS Paper: 7497, TWS Live: 7496
IB_CLIENT_ID=1

# 交易模式
TRADING_MODE=paper    # paper 或 live

# 風控設定
MAX_DAILY_LOSS=1000
MAX_POSITION_SIZE=100
```

### 5. 配置 TWS / IB Gateway

1. 啟動 TWS 或 IB Gateway
2. 進入 **Edit → Global Configuration → API → Settings**
3. 勾選 **Enable ActiveX and Socket Clients**
4. 設定 **Socket Port** (預設 7497)
5. 勾選 **Allow connections from localhost only**

## 快速開始指南

### 實盤交易

```bash
python run_live.py
```

### 回測

```bash
python run_backtest.py --strategy sma_cross --symbol AAPL --start 2024-01-01 --end 2024-12-01
```

### 開發自定義策略

1. 在 `strategies/examples/` 建立新策略檔案：

```python
# strategies/examples/my_strategy.py

from strategies.base import BaseStrategy
from core.events import BarEvent, OrderAction

class MyStrategy(BaseStrategy):
    def __init__(self, strategy_id: str, symbols: list, **kwargs):
        super().__init__(strategy_id, symbols, **kwargs)
        self._params = {
            "threshold": 0.02,  # 自定義參數
        }
    
    def on_bar(self, event: BarEvent) -> None:
        """處理 Bar 數據，實作你的策略邏輯"""
        symbol = event.symbol
        if symbol not in self.symbols:
            return
        
        # 你的策略邏輯
        threshold = self.get_param("threshold")
        
        # 發送交易信號
        if your_buy_condition:
            self.emit_signal(symbol, OrderAction.BUY, strength=1.0)
        elif your_sell_condition:
            self.emit_signal(symbol, OrderAction.SELL, strength=1.0)
```

2. 註冊策略：

```python
from strategies.registry import get_registry
from strategies.examples.my_strategy import MyStrategy

registry = get_registry()
registry.register_class("my_strategy", MyStrategy)
```

## 目錄結構說明

```
ib_trading_system/
│
├── config/                     # 配置模組
│   ├── __init__.py
│   ├── settings.py            # 全局配置（從環境變數載入）
│   ├── trading_modes.py       # 交易模式配置（高頻/低頻/波段）
│   └── symbols.py             # 交易標的定義
│
├── core/                       # 核心模組
│   ├── __init__.py
│   ├── events.py              # 事件定義（Tick/Bar/Order/Fill等）
│   ├── event_bus.py           # 事件總線（發布/訂閱模式）
│   ├── connection.py          # IB 連接管理
│   └── contracts.py           # 合約工廠
│
├── data/                       # 數據模組
│   ├── __init__.py
│   ├── feed_handler.py        # 市場數據接收與分發
│   ├── bar_aggregator.py      # Tick 轉 Bar 聚合器
│   ├── database.py            # 數據庫操作（SQLAlchemy）
│   └── cache.py               # 內存快取（高頻用）
│
├── engine/                     # 引擎模組
│   ├── __init__.py
│   ├── strategy_engine.py     # 策略執行引擎
│   └── execution_engine.py    # 訂單執行引擎
│
├── strategies/                 # 策略模組
│   ├── __init__.py
│   ├── base.py                # 策略基類（抽象類）
│   ├── registry.py            # 策略註冊器
│   └── examples/              # 範例策略
│       ├── __init__.py
│       ├── sma_cross.py       # 均線交叉策略
│       └── tick_scalper.py    # Tick 剝頭皮策略
│
├── risk/                       # 風控模組
│   ├── __init__.py
│   ├── risk_manager.py        # 風險管理器
│   ├── position_sizer.py      # 倉位計算器
│   └── circuit_breaker.py     # 熔斷機制
│
├── execution/                  # 執行模組
│   ├── __init__.py
│   ├── order_manager.py       # 訂單管理器
│   └── order_types.py         # 訂單類型工具
│
├── utils/                      # 工具模組
│   ├── __init__.py
│   ├── logger.py              # 日誌配置（Loguru）
│   ├── notifier.py            # 通知服務（Telegram/Email）
│   ├── time_utils.py          # 交易時間工具
│   └── performance.py         # 性能監控
│
├── backtest/                   # 回測模組
│   ├── __init__.py
│   ├── engine.py              # 回測引擎
│   └── data_loader.py         # 歷史數據載入器
│
├── tests/                      # 測試模組
│   ├── __init__.py
│   ├── test_connection.py     # 連接測試
│   ├── test_strategies.py     # 策略測試
│   └── test_risk.py           # 風控測試
│
├── logs/                       # 日誌輸出目錄
├── data_store/                 # 數據儲存目錄
│
├── main.py                    # 主程式入口
├── run_live.py                # 實盤啟動腳本
├── run_backtest.py            # 回測啟動腳本
├── requirements.txt           # Python 依賴
├── .env.example               # 環境變數範例
├── .gitignore                 # Git 忽略清單
└── README.md                  # 專案說明
```

## 注意事項

⚠️ **風險警告**：自動化交易存在風險，請務必：

1. 先在模擬帳戶（Paper Trading）充分測試
2. 設定合理的風控參數
3. 監控系統運行狀態
4. 了解 IB 的訊息頻率限制（50 條/秒）
5. 確保網路連線穩定

## License

MIT License