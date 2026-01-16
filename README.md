# IB Trading System

基於 Interactive Brokers API 的自動化交易系統

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 目錄

- [功能特色](#-功能特色)
- [專案結構](#-專案結構)
- [安裝說明](#-安裝說明)
- [快速開始](#-快速開始)
- [模組說明](#-模組說明)
- [Event Contract 交易](#-event-contract-交易)
- [策略開發](#-策略開發)
- [TWS 設定](#-tws-設定)
- [常用命令](#-常用命令)
- [經濟日曆](#-經濟日曆)

---

## ✨ 功能特色

### 核心功能
- **多資產支持**：股票、外匯、期貨、期權、商品、Event Contract
- **事件驅動架構**：基於 EventBus 的高效事件處理
- **風險管理**：倉位控制、熔斷機制、最大回撤保護
- **策略引擎**：模組化策略框架，易於擴展
- **數據管理**：實時行情、K線聚合、歷史數據緩存

### 交易類型
| 類型 | 說明 | 狀態 |
|------|------|------|
| 外匯 (Forex) | EUR/USD, GBP/USD 等 | ✅ |
| 股票 (Stock) | 美股、ETF | ✅ |
| 期貨 (Futures) | ES, NQ, GC 等 | ✅ |
| 商品 (Commodity) | XAUUSD, XAGUSD | ✅ |
| Event Contract | ForecastEx, CME Event | ✅ 新增 |

### 新增：Event Contract 支持
- **ForecastEx**：Fed Funds Rate、CPI、失業率等經濟指標預測
- **經濟日曆**：2025-2026 年 FOMC、CPI、NFP 日期
- **專用策略**：Fed Funds 策略框架

---

## 📁 專案結構

```
ib_trading_system/
├── config/                     # ⚙️ 配置模組
│   ├── settings.py             # 全局配置
│   ├── symbols.py              # 交易標的定義
│   ├── trading_modes.py        # 交易模式
│   └── event_contracts.py      # 📅 經濟日曆 (新增)
│
├── core/                       # 🔧 核心模組
│   ├── connection.py           # IB 連接管理
│   ├── contracts.py            # 合約工廠 (含 Event Contract)
│   ├── events.py               # 事件定義
│   └── event_bus.py            # 事件總線
│
├── data/                       # 📊 數據模組
│   ├── feed_handler.py         # 市場數據接收
│   ├── bar_aggregator.py       # K線聚合器
│   ├── cache.py                # 數據快取
│   ├── database.py             # 數據庫操作
│   └── precious_metals_fetcher.py  # 貴金屬數據
│
├── strategies/                 # 📈 策略模組
│   ├── base.py                 # 策略基類
│   ├── registry.py             # 策略註冊器
│   ├── examples/               # 範例策略
│   ├── triangular_arbitrage/   # 三角套利策略
│   └── event_contracts/        # 🆕 Event Contract 策略
│       ├── base.py             # Event 策略基類
│       └── fed_funds.py        # Fed Funds 策略
│
├── risk/                       # 🛡️ 風控模組
│   ├── manager.py              # 風險管理器
│   ├── position_sizer.py       # 倉位計算器
│   └── circuit_breaker.py      # 熔斷機制
│
├── engine/                     # ⚡ 引擎模組
│   ├── strategy_engine.py      # 策略執行引擎
│   └── execution_engine.py     # 訂單執行引擎
│
├── execution/                  # 📝 執行模組
│   ├── order_manager.py        # 訂單管理器
│   └── order_types.py          # 訂單類型
│
├── utils/                      # 🔨 工具模組
│   ├── logger.py               # 日誌配置
│   ├── market_hours.py         # 市場時間
│   ├── time_utils.py           # 時間工具
│   ├── notifier.py             # 通知服務
│   └── performance.py          # 性能監控
│
├── backtest/                   # 📉 回測模組
│   ├── engine.py               # 回測引擎
│   ├── data_loader.py          # 歷史數據載入
│   └── visualization.py        # 視覺化
│
├── tests/                      # 🧪 測試模組
├── examples/                   # 📚 範例腳本
│
├── run_live.py                 # 🚀 實盤交易入口
├── run_backtest.py             # 📊 回測入口
└── README.md                   # 說明文件
```

---

## 🔧 安裝說明

### 環境需求

| 項目 | 版本/設定 |
|------|-----------|
| Python | **3.12**（必須） |
| IB TWS | Paper Trading 或 Live |
| 作業系統 | macOS / Windows / Linux |

> ⚠️ **重要**：不要使用 Python 3.14，與 ib_insync 不相容

### 安裝步驟

```bash
# 1. Clone 專案
git clone https://github.com/HUANG-ZIV/ib_trading_system.git
cd ib_trading_system

# 2. 創建虛擬環境
python3.12 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 3. 安裝依賴
pip install ib_insync pandas numpy loguru python-dotenv sqlalchemy aiosqlite nest_asyncio

# 4. 設定環境變數（可選）
cp .env.example .env
# 編輯 .env 設定你的參數
```

### 依賴套件

```
ib_insync          # IB API 封裝
pandas             # 數據處理
numpy              # 數值計算
loguru             # 日誌
python-dotenv      # 環境變數
sqlalchemy         # 數據庫 ORM
aiosqlite          # 異步 SQLite
nest_asyncio       # 異步兼容
```

---

## 🚀 快速開始

### 1. 啟動 IB TWS

確保 TWS 已啟動並登入，API 設定正確（見 [TWS 設定](#-tws-設定)）

### 2. 測試連接

```bash
python -c "
from ib_insync import IB
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)
print(f'連接成功！帳戶: {ib.managedAccounts()}')
ib.disconnect()
"
```

### 3. 執行交易系統

```bash
python run_live.py
```

### 4. 基本使用範例

```python
from ib_insync import IB
from core.contracts import get_contract_factory

# 連接 IB
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# 取得合約工廠
factory = get_contract_factory()

# 建立各種合約
aapl = factory.stock("AAPL")
eurusd = factory.forex("EUR", "USD")
es = factory.future("ES", "202503")
gold = factory.commodity("XAUUSD")

# Event Contract (新功能)
ff = factory.fed_funds("20260128", 4.375, is_yes=True)

# 訂閱行情
ib.reqMktData(eurusd)

# 下單範例
from ib_insync import MarketOrder
order = MarketOrder('BUY', 20000)
trade = ib.placeOrder(eurusd, order)
```

---

## 📦 模組說明

### Core - 核心模組

#### `contracts.py` - 合約工廠

```python
from core.contracts import get_contract_factory

factory = get_contract_factory()

# 股票
aapl = factory.stock("AAPL", primary_exchange="NASDAQ")

# 外匯
eurusd = factory.forex("EUR", "USD")

# 期貨
es = factory.future("ES", "202503", exchange="CME")

# 期權
call = factory.option("AAPL", "20250321", 200.0, "C")

# 商品
gold = factory.commodity("XAUUSD")

# Event Contract (新增)
ff_yes = factory.forecastex("FF", "20260128", 4.375, is_yes=True)
ff_no = factory.forecastex("FF", "20260128", 4.375, is_yes=False)
cpi = factory.cpi("20260212", 3.5)
```

#### `connection.py` - 連接管理

```python
from core.connection import IBConnection

conn = IBConnection(host='127.0.0.1', port=7497, client_id=1)
await conn.connect()

# 自動重連、心跳檢測
```

### Risk - 風控模組

```python
from risk.manager import RiskManager
from risk.position_sizer import PositionSizer
from risk.circuit_breaker import CircuitBreaker

# 風險管理器
risk_mgr = RiskManager(max_drawdown=0.1, max_position_pct=0.2)

# 倉位計算
sizer = PositionSizer(account_value=100000, risk_per_trade=0.01)
size = sizer.calculate(entry_price=1.1000, stop_loss=1.0950)

# 熔斷機制
breaker = CircuitBreaker(max_daily_loss=1000)
```

---

## 🎯 Event Contract 交易

### 什麼是 Event Contract？

Event Contract 是基於經濟指標結果的二元預測合約：
- **ForecastEx**：IB 的經濟指標預測交易所
- **結算方式**：二元（贏 = $1.00，輸 = $0.00）
- **價格 = 機率**：$0.65 表示市場認為 65% 機率發生

### 支持的市場

| 代碼 | 名稱 | 說明 |
|------|------|------|
| FF | Fed Funds Rate | 聯邦基金利率 |
| CPI | Consumer Price Index | 消費者物價指數 |
| UNRATE | Unemployment Rate | 失業率 |
| GDP | Gross Domestic Product | 國內生產總值 |

### 使用範例

```python
from core.contracts import get_contract_factory
from config.event_contracts import (
    get_next_fomc_date, 
    days_until_fomc,
    print_economic_calendar
)

# 查看經濟日曆
print_economic_calendar()

# 取得下一個 FOMC 日期
next_fomc = get_next_fomc_date()  # "20260128"
days = days_until_fomc(next_fomc)  # 12 天

# 建立合約
factory = get_contract_factory()

# Fed Funds Rate YES 合約（預測利率 >= 4.375%）
ff_yes = factory.fed_funds(next_fomc, 4.375, is_yes=True)

# Fed Funds Rate NO 合約（預測利率 < 4.375%）
ff_no = factory.fed_funds(next_fomc, 4.375, is_yes=False)

# 建立合約對
yes, no = factory.forecastex_pair("FF", next_fomc, 4.375)
```

### 交易規則

1. **只能 BUY**：ForecastEx 不支持賣空
2. **平倉方式**：買入相反合約（Yes 平倉買 No）
3. **價格範圍**：0.01 - 0.99
4. **結算**：事件結果公布後自動結算

### Fed Funds 策略

```python
from strategies.event_contracts import FedFundsStrategy

strategy = FedFundsStrategy(
    ib=ib,
    current_rate=4.375,
    min_edge=0.08,  # 最小 8% edge
)

# 設定你的預測
strategy.set_rate_prediction(
    expiry="20260128",
    cut=0.05,    # 5% 降息機率
    hold=0.90,   # 90% 維持機率
    hike=0.05,   # 5% 升息機率
)

# 分析市場
strategy.print_market_analysis("20260128")

# 啟動策略
strategy.start()
```

---

## 📝 策略開發

### 策略基類

```python
from strategies.base import StrategyBase

class MyStrategy(StrategyBase):
    def __init__(self, params):
        super().__init__("MyStrategy")
        self.fast_period = params.get("fast_period", 10)
        self.slow_period = params.get("slow_period", 20)
    
    def on_bar(self, bar):
        """K線回調"""
        # 你的策略邏輯
        pass
    
    def on_tick(self, tick):
        """Tick 回調"""
        pass
```

### 註冊策略

```python
from strategies.registry import StrategyRegistry

registry = StrategyRegistry()
registry.register("my_strategy", MyStrategy)
```

---

## ⚙️ TWS 設定

### API 設定

1. 打開 TWS → **Edit** → **Global Configuration**
2. 選擇 **API** → **Settings**
3. 設定如下：

| 設定項目 | 值 |
|----------|-----|
| Enable ActiveX and Socket Clients | ✅ 勾選 |
| Socket port | **7497**（Paper）/ 7496（Live） |
| Read-Only API | ❌ 不勾選 |
| Allow connections from localhost only | ✅ 勾選 |

### 連接參數

```python
# Paper Trading
ib.connect('127.0.0.1', 7497, clientId=1)

# Live Trading
ib.connect('127.0.0.1', 7496, clientId=1)
```

---

## 💻 常用命令

### 系統操作

```bash
# 啟動虛擬環境
source venv/bin/activate

# 執行交易系統
python run_live.py

# 執行回測
python run_backtest.py

# 查看日誌
tail -f logs/trading_*.log
```

### Git 操作

```bash
# 查看狀態
git status

# 提交更新
git add .
git commit -m "說明"
git push

# 拉取更新
git pull
```

### 測試

```bash
# 執行所有測試
pytest tests/

# 執行特定測試
pytest tests/test_connection.py -v
```

---

## 📅 經濟日曆

### 2026 FOMC 會議

| 日期 | SEP* | 說明 |
|------|------|------|
| 01-28 | | |
| 03-18 | ✓ | 含 Dot Plot |
| 04-29 | | |
| 06-17 | ✓ | 含 Dot Plot |
| 07-29 | | |
| 09-16 | ✓ | 含 Dot Plot |
| 10-28 | | |
| 12-09 | ✓ | 含 Dot Plot |

*SEP = Summary of Economic Projections

### 查看日曆

```python
from config.event_contracts import print_economic_calendar
print_economic_calendar()
```

輸出：
```
==================================================
📅 FOMC 會議日曆
==================================================

🏛️ 2025 FOMC:
  2025-01-29 (Wed)
  2025-03-19 (Wed)
  ...

🏛️ 2026 FOMC:
  2026-01-28 (Wed)
  2026-03-18 (Wed)
  ...

⏰ 下一個 FOMC: 2026-01-28 (12 天後)
==================================================
```

---

## 📞 支援

- **GitHub Issues**: [提交問題](https://github.com/HUANG-ZIV/ib_trading_system/issues)
- **Email**: ziv.yu.hsiang.huang@gmail.com

---

## 📄 License

MIT License

---

## 更新日誌

### v2.0.0 (2026-01)
- ✨ 新增 Event Contract 支持（ForecastEx）
- ✨ 新增 2025-2026 FOMC 經濟日曆
- ✨ 新增 Fed Funds 策略框架
- 🔧 專案結構重組（tests/, examples/）

### v1.0.0 (2024-12)
- 🎉 初始版本
- ✅ 外匯交易支持
- ✅ SMA 策略範例
- ✅ 風險管理模組
