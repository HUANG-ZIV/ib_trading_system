# IB Trading System - 程式設計文檔

## 最後更新：2024-12-14

---

## 目錄

1. [系統架構總覽](#系統架構總覽)
2. [程式執行流程](#程式執行流程)
3. [事件驅動架構](#事件驅動架構)
4. [各模組詳細說明](#各模組詳細說明)
   - [config/ 配置模組](#config-配置模組)
   - [core/ 核心模組](#core-核心模組)
   - [data/ 數據模組](#data-數據模組)
   - [strategies/ 策略模組](#strategies-策略模組)
   - [risk/ 風控模組](#risk-風控模組)
   - [engine/ 引擎模組](#engine-引擎模組)
   - [execution/ 執行模組](#execution-執行模組)
   - [utils/ 工具模組](#utils-工具模組)
   - [backtest/ 回測模組](#backtest-回測模組)
5. [主程式說明](#主程式說明)
6. [數據流向圖](#數據流向圖)
7. [如何開發新策略](#如何開發新策略)
8. [如何新增交易標的](#如何新增交易標的)
9. [常見修改場景](#常見修改場景)

---

## 系統架構總覽

```
┌─────────────────────────────────────────────────────────────────┐
│                        run_live.py                               │
│                      （主程式入口）                               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LiveTrader 類別                             │
│  負責初始化和協調所有組件                                         │
└─────────────────────────────┬───────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  IBConnection │    │   EventBus    │    │  RiskManager  │
│  （IB 連接）   │    │  （事件總線）  │    │  （風險管理）  │
└───────┬───────┘    └───────┬───────┘    └───────────────┘
        │                    │
        ▼                    │
┌───────────────┐            │
│  FeedHandler  │◄───────────┤
│ （數據接收）   │            │
└───────┬───────┘            │
        │                    │
        ▼                    │
┌───────────────┐            │
│StrategyEngine │◄───────────┤
│ （策略引擎）   │            │
└───────┬───────┘            │
        │                    │
        ▼                    │
┌───────────────┐            │
│ExecutionEngine│◄───────────┘
│ （執行引擎）   │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   IB TWS      │
│  （交易所）    │
└───────────────┘
```

---

## 程式執行流程

### 啟動流程

```
1. python run_live.py
        │
        ▼
2. main() 函數
        │
        ├── 設定日誌系統
        ├── 解析命令列參數
        ├── 建立 LiveTrader 實例
        │
        ▼
3. LiveTrader.initialize()
        │
        ├── 初始化 EventBus（事件總線）
        ├── 初始化 Notifier（通知服務）
        ├── 初始化 PerformanceMonitor（性能監控）
        ├── 初始化 MarketDataCache（數據快取）
        ├── 初始化 IBConnection（IB 連接）
        ├── 初始化 FeedHandler（數據處理）
        ├── 初始化 BarAggregator（K線聚合）
        ├── 初始化 RiskManager（風險管理）
        ├── 初始化 PositionSizer（倉位計算）
        ├── 初始化 CircuitBreaker（熔斷機制）
        ├── 初始化 StrategyEngine（策略引擎）
        └── 初始化 ExecutionEngine（執行引擎）
        │
        ▼
4. LiveTrader.connect()
        │
        ├── 連接到 IB TWS
        ├── 取得帳戶資訊
        └── 訂閱市場數據
        │
        ▼
5. LiveTrader.run()
        │
        ├── 載入策略
        ├── 啟動所有組件
        └── 進入主循環
                │
                ▼
        ┌──────────────────┐
        │  主循環（每秒）   │◄────┐
        │                  │     │
        │  - 處理事件      │     │
        │  - 更新狀態      │     │
        │  - 檢查連接      │     │
        │  - 記錄性能      │     │
        └────────┬─────────┘     │
                 │               │
                 └───────────────┘
        │
        ▼
6. LiveTrader.shutdown()（按 Ctrl+C）
        │
        ├── 停止策略引擎
        ├── 取消未成交訂單
        ├── 停止風控組件
        ├── 斷開 IB 連接
        └── 發送關閉通知
```

### 交易信號流程

```
1. 市場數據進入
        │
        ▼
2. FeedHandler 接收 Tick/Bar 數據
        │
        ▼
3. 發布事件到 EventBus
        │
        ▼
4. StrategyEngine 接收事件
        │
        ▼
5. 策略處理數據，產生信號
        │
        ▼
6. RiskManager 檢查風險
        │
        ├── 通過 → 繼續
        └── 拒絕 → 記錄日誌，結束
        │
        ▼
7. PositionSizer 計算倉位
        │
        ▼
8. ExecutionEngine 建立訂單
        │
        ▼
9. 發送訂單到 IB TWS
        │
        ▼
10. 接收成交回報
        │
        ▼
11. 更新持倉和帳戶資訊
```

---

## 事件驅動架構

### 事件類型（core/events.py）

```python
EventType:
    # 連接事件
    CONNECTED          # IB 連接成功
    DISCONNECTED       # IB 斷線
    
    # 數據事件
    TICK               # Tick 數據更新
    BAR                # K線數據更新
    
    # 交易事件
    SIGNAL             # 策略信號
    ORDER              # 訂單事件
    FILL               # 成交事件
    POSITION           # 持倉更新
    
    # 系統事件
    ERROR              # 錯誤事件
    WARNING            # 警告事件
```

### 事件流向

```
┌──────────────┐     發布事件      ┌──────────────┐
│  FeedHandler │ ───────────────► │   EventBus   │
└──────────────┘                  └──────┬───────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
           ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
           │StrategyEngine│     │ RiskManager  │     │    Cache     │
           │  訂閱: TICK  │     │ 訂閱: ORDER  │     │ 訂閱: TICK   │
           │       BAR    │     │       FILL   │     │       BAR    │
           └──────────────┘     └──────────────┘     └──────────────┘
```

---

## 各模組詳細說明

---

### config/ 配置模組

#### settings.py
```
用途：全局配置管理
功能：
  - 從環境變數載入配置
  - 提供 IB 連接設定（host, port, client_id）
  - 提供風控設定（max_daily_loss, max_position_size）
  - 提供日誌設定

主要類別：
  - Settings: 主配置類
  - IBSettings: IB 連接配置
  - RiskSettings: 風控配置

使用方式：
  from config.settings import get_settings
  settings = get_settings()
  print(settings.ib.host)  # 127.0.0.1
```

#### trading_modes.py
```
用途：定義交易模式
功能：
  - 高頻交易模式設定
  - 低頻交易模式設定
  - 波段交易模式設定

主要類別：
  - TradingMode: 交易模式枚舉
  - ModeConfig: 模式配置
```

#### symbols.py
```
用途：定義交易標的
功能：
  - 美股標的列表
  - ETF 標的列表
  - 外匯標的列表

主要變數：
  - US_TECH_STOCKS: 科技股列表
  - US_ETFS: ETF 列表
```

---

### core/ 核心模組

#### connection.py
```
用途：管理 IB TWS 連接
功能：
  - 建立連接
  - 斷線重連
  - 連接狀態監控

主要類別：
  - ConnectionConfig: 連接配置
      - host: str (預設 "127.0.0.1")
      - port: int (預設 7497)
      - client_id: int (預設 1)
      - timeout: int (預設 20)
      - readonly: bool (預設 False)
  
  - IBConnection: 連接管理器
      方法：
      - connect(): 建立連接
      - disconnect(): 斷開連接
      - reconnect(): 重新連接
      - is_connected: 連接狀態屬性
      - ib: 底層 ib_insync.IB 實例

使用方式：
  config = ConnectionConfig(host="127.0.0.1", port=7497)
  connection = IBConnection(config)
  connection.connect()
```

#### contracts.py
```
用途：建立各種合約
功能：
  - 股票合約
  - 外匯合約
  - 期貨合約
  - 選擇權合約
  - 指數合約

主要類別：
  - ContractFactory: 合約工廠
      方法：
      - stock(symbol, exchange, currency): 建立股票合約
      - forex(base, quote, exchange): 建立外匯合約
      - future(symbol, expiry, exchange): 建立期貨合約
      - option(symbol, expiry, strike, right): 建立選擇權合約
      - index(symbol, exchange): 建立指數合約

使用方式：
  factory = ContractFactory()
  aapl = factory.stock("AAPL")
  eurusd = factory.forex("EUR", "USD")
```

#### events.py
```
用途：定義所有事件類型
功能：
  - 事件基類
  - 各種事件類型定義

主要類別：
  - EventType: 事件類型枚舉
  - Event: 事件基類
  - TickEvent: Tick 數據事件
  - BarEvent: K線數據事件
  - SignalEvent: 交易信號事件
  - OrderEvent: 訂單事件
  - FillEvent: 成交事件
  - PositionEvent: 持倉事件

使用方式：
  event = TickEvent(
      symbol="AAPL",
      timestamp=datetime.now(),
      price=150.0,
      volume=100
  )
```

#### event_bus.py
```
用途：事件發布/訂閱系統
功能：
  - 事件發布
  - 事件訂閱
  - 異步事件處理

主要類別：
  - EventBus: 事件總線
      方法：
      - subscribe(event_type, handler): 訂閱事件
      - unsubscribe(event_type, handler): 取消訂閱
      - publish(event): 發布事件
      - start(): 啟動事件處理
      - stop(): 停止事件處理

使用方式：
  bus = EventBus()
  bus.subscribe(EventType.TICK, my_handler)
  bus.publish(tick_event)
```

---

### data/ 數據模組

#### feed_handler.py
```
用途：接收和分發市場數據
功能：
  - 訂閱即時數據
  - 訂閱歷史數據
  - 數據格式轉換

主要類別：
  - SubscriptionType: 訂閱類型枚舉
      - TICK: Tick 數據
      - BAR: K線數據
      - BOTH: 兩者都要
  
  - FeedHandler: 數據處理器
      方法：
      - subscribe(contract, subscription_type): 訂閱數據
      - unsubscribe(symbol): 取消訂閱
      - unsubscribe_all(): 取消所有訂閱
      - get_historical_data(contract, duration, bar_size): 取得歷史數據

使用方式：
  handler = FeedHandler(connection, event_bus)
  await handler.subscribe(contract)
```

#### bar_aggregator.py
```
用途：將 Tick 數據聚合成 K線
功能：
  - 多時間週期聚合（1分/5分/15分/1小時/日線）
  - 自動完成 K線並發布事件

主要類別：
  - BarAggregator: K線聚合器
      方法：
      - add_symbol(symbol, timeframes): 新增標的
      - on_tick(tick_event): 處理 Tick 數據
      - start(): 啟動聚合器
      - stop(): 停止聚合器

使用方式：
  aggregator = BarAggregator(event_bus)
  aggregator.add_symbol("AAPL", ["1m", "5m", "1h"])
```

#### cache.py
```
用途：記憶體快取市場數據
功能：
  - 快取最新 Tick
  - 快取最新 Bar
  - 高頻交易優化

主要類別：
  - MarketDataCache: 數據快取
      方法：
      - update_tick(symbol, tick): 更新 Tick
      - update_bar(symbol, timeframe, bar): 更新 Bar
      - get_latest_tick(symbol): 取得最新 Tick
      - get_latest_bar(symbol, timeframe): 取得最新 Bar
      - get_bar_history(symbol, timeframe, count): 取得歷史 Bar

使用方式：
  cache = MarketDataCache(event_bus)
  latest = cache.get_latest_tick("AAPL")
```

#### database.py
```
用途：數據持久化儲存
功能：
  - SQLite 數據庫
  - 儲存歷史數據
  - 儲存交易記錄

主要類別：
  - Database: 數據庫管理器
  - TickRecord: Tick 記錄模型
  - BarRecord: Bar 記錄模型
  - TradeRecord: 交易記錄模型
```

---

### strategies/ 策略模組

#### base.py
```
用途：策略基類定義
功能：
  - 策略生命週期管理
  - 信號發送介面
  - 參數管理

主要類別：
  - StrategyConfig: 策略配置
      屬性：
      - name: 策略名稱
      - symbols: 交易標的列表
      - max_position_size: 最大持倉
      - stop_loss_pct: 停損百分比
      - take_profit_pct: 停利百分比
  
  - BaseStrategy: 策略基類（抽象類）
      方法（需覆寫）：
      - on_tick(event): 處理 Tick 數據
      - on_bar(event): 處理 Bar 數據
      
      方法（可使用）：
      - initialize(): 初始化策略
      - start(): 啟動策略
      - stop(): 停止策略
      - emit_signal(symbol, action, strength): 發送交易信號
      - get_param(name): 取得參數
      - set_param(name, value): 設定參數

使用方式：
  class MyStrategy(BaseStrategy):
      def on_bar(self, event):
          # 實作策略邏輯
          if buy_condition:
              self.emit_signal(symbol, OrderAction.BUY, 1.0)
```

#### registry.py
```
用途：策略註冊和管理
功能：
  - 註冊策略類別
  - 根據名稱建立策略實例

主要類別：
  - StrategyRegistry: 策略註冊器
      方法：
      - register_class(name, strategy_class): 註冊策略
      - create(name, **kwargs): 建立策略實例
      - list_strategies(): 列出所有已註冊策略

使用方式：
  registry = get_registry()
  registry.register_class("my_strategy", MyStrategy)
  strategy = registry.create("my_strategy", symbols=["AAPL"])
```

#### examples/sma_cross.py
```
用途：SMA 均線交叉策略範例
功能：
  - 快線上穿慢線 → 買入
  - 快線下穿慢線 → 賣出

主要類別：
  - SMACrossConfig: 策略配置
      - fast_period: 快線週期（預設 10）
      - slow_period: 慢線週期（預設 20）
      - quantity: 交易數量（預設 100）
  
  - SMACrossStrategy: SMA 交叉策略
      方法：
      - on_bar(event): 處理 K線，計算均線，判斷交叉

使用方式：
  strategy = SMACrossStrategy(
      strategy_id="sma_cross",
      symbols=["EUR/USD"],
      fast_period=10,
      slow_period=20,
      quantity=20000
  )
```

#### examples/tick_scalper.py
```
用途：Tick 剝頭皮策略範例
功能：
  - 高頻交易策略
  - 基於 Tick 價格變動快速進出

主要類別：
  - TickScalperConfig: 策略配置
  - TickScalperStrategy: 剝頭皮策略
```

---

### risk/ 風控模組

#### manager.py
```
用途：整體風險管理
功能：
  - 每日虧損控制
  - 持倉限制檢查
  - 訂單風險驗證

主要類別：
  - RiskManager: 風險管理器
      方法：
      - check_order(order): 檢查訂單是否符合風控
      - update_account_value(value): 更新帳戶價值
      - get_daily_pnl(): 取得當日損益
      - reset_daily_stats(): 重置每日統計
      - start(): 啟動風控
      - stop(): 停止風控

使用方式：
  manager = RiskManager(config, event_bus, account_value=100000)
  is_ok = manager.check_order(order)
```

#### position_sizer.py
```
用途：計算交易倉位大小
功能：
  - 固定數量
  - 固定百分比
  - 基於 ATR 的動態倉位

主要類別：
  - SizingMethod: 倉位計算方法枚舉
      - FIXED: 固定數量
      - PERCENT: 百分比
      - ATR: 基於 ATR
  
  - PositionSizer: 倉位計算器
      方法：
      - calculate(symbol, price, stop_loss): 計算倉位
      - set_account_value(value): 設定帳戶價值
      - set_method(method): 設定計算方法

使用方式：
  sizer = PositionSizer(
      account_value=100000,
      risk_per_trade=0.01,
      max_position_pct=0.1
  )
  size = sizer.calculate("AAPL", 150.0, 145.0)
```

#### circuit_breaker.py
```
用途：熔斷機制
功能：
  - 連續虧損達標後暫停交易
  - 冷卻期後自動恢復

主要類別：
  - BreakerConfig: 熔斷配置
      - max_consecutive_losses: 最大連續虧損次數
      - cooldown_seconds: 冷卻時間（秒）
  
  - CircuitBreaker: 熔斷器
      方法：
      - record_trade(is_win): 記錄交易結果
      - is_triggered: 是否已觸發熔斷
      - get_remaining_cooldown(): 剩餘冷卻時間
      - reset(): 重置熔斷器

使用方式：
  breaker = CircuitBreaker(config, event_bus)
  if breaker.is_triggered:
      print("熔斷中，暫停交易")
```

---

### engine/ 引擎模組

#### strategy_engine.py
```
用途：管理和執行所有策略
功能：
  - 策略生命週期管理
  - 事件分發到策略
  - 策略狀態監控

主要類別：
  - StrategyEngine: 策略引擎
      方法：
      - add_strategy(strategy): 新增策略
      - remove_strategy(strategy_id): 移除策略
      - get_strategy(strategy_id): 取得策略
      - start(): 啟動所有策略
      - stop(): 停止所有策略
      - on_tick(event): 分發 Tick 事件
      - on_bar(event): 分發 Bar 事件

使用方式：
  engine = StrategyEngine(event_bus)
  engine.add_strategy(my_strategy)
  engine.start()
```

#### execution_engine.py
```
用途：處理訂單執行
功能：
  - 訂單建立
  - 訂單發送
  - 成交處理

主要類別：
  - ExecutionEngine: 執行引擎
      方法：
      - submit_order(order): 提交訂單
      - cancel_order(order_id): 取消訂單
      - cancel_all_orders(): 取消所有訂單
      - get_open_orders(): 取得未成交訂單
      - start(): 啟動引擎
      - stop(): 停止引擎

使用方式：
  engine = ExecutionEngine(connection, event_bus)
  engine.submit_order(order)
```

---

### execution/ 執行模組

#### order_manager.py
```
用途：訂單狀態管理
功能：
  - 追蹤訂單狀態
  - 訂單歷史記錄

主要類別：
  - OrderManager: 訂單管理器
      方法：
      - add_order(order): 新增訂單
      - update_order(order_id, status): 更新狀態
      - get_order(order_id): 取得訂單
      - get_open_orders(): 取得未成交訂單
```

#### order_types.py
```
用途：訂單類型工具
功能：
  - 建立各種訂單類型
  - 市價單、限價單、停損單等

主要函數：
  - market_order(action, quantity): 市價單
  - limit_order(action, quantity, price): 限價單
  - stop_order(action, quantity, stop_price): 停損單
  - bracket_order(action, quantity, entry, take_profit, stop_loss): 括號訂單
```

---

### utils/ 工具模組

#### logger.py
```
用途：日誌系統配置
功能：
  - 結構化日誌
  - 檔案輸出
  - 控制台輸出

主要函數：
  - setup_logger(log_dir, log_level): 設定日誌
  - get_logger(name): 取得 logger 實例

使用方式：
  setup_logger(log_dir="logs", log_level="INFO")
  logger = get_logger("MyModule")
  logger.info("Hello")
```

#### market_hours.py
```
用途：市場交易時間工具
功能：
  - 判斷市場是否開盤
  - 計算距離開盤時間
  - 時區轉換

主要函數：
  - is_market_open(): 市場是否開盤
  - time_until_market_open(): 距離開盤時間
  - get_eastern_time(): 取得美東時間
  - get_taiwan_time(): 取得台灣時間

使用方式：
  if is_market_open():
      print("市場開盤中")
  else:
      remaining = time_until_market_open()
      print(f"距離開盤: {remaining}")
```

#### time_utils.py
```
用途：時間處理工具
功能：
  - 時間格式化
  - 時間計算

主要函數：
  - format_duration(timedelta): 格式化時間長度
  - parse_time(time_str): 解析時間字串
```

#### notifier.py
```
用途：通知服務
功能：
  - Telegram 通知
  - Email 通知

主要類別：
  - NotificationLevel: 通知等級
      - INFO, WARNING, ERROR, CRITICAL
  
  - NotificationConfig: 通知配置
  
  - Notifier: 通知器
      方法：
      - alert(message, level): 發送通知
      - initialize(): 初始化
      - shutdown(): 關閉

使用方式：
  notifier = Notifier(config)
  await notifier.alert("交易信號", NotificationLevel.INFO)
```

#### performance.py
```
用途：性能監控
功能：
  - 記錄事件處理時間
  - 系統資源監控

主要類別：
  - PerformanceMonitor: 性能監控器
      方法：
      - record_event(event_name): 記錄事件
      - get_stats(): 取得統計
      - start(): 啟動監控
      - stop(): 停止監控
```

---

### backtest/ 回測模組

#### engine.py
```
用途：回測引擎
功能：
  - 歷史數據回放
  - 模擬交易執行
  - 績效計算

主要類別：
  - BacktestEngine: 回測引擎
      方法：
      - run(strategy, data, start_date, end_date): 執行回測
      - get_results(): 取得回測結果
```

#### data_loader.py
```
用途：載入歷史數據
功能：
  - 從 IB 下載歷史數據
  - 從本地檔案載入

主要類別：
  - DataLoader: 數據載入器
      方法：
      - load_from_ib(symbol, start, end): 從 IB 載入
      - load_from_csv(filepath): 從 CSV 載入
```

---

## 主程式說明

### run_live.py

```
這是實盤交易的主程式入口。

主要組成：
1. Python 3.14 相容性修復（設定 event loop）
2. LIVE_SYMBOLS: 交易標的設定
3. STRATEGY_CONFIG: 策略配置
4. LiveTrader 類別: 交易運行器
5. main() 函數: 程式入口

重要方法：
- initialize(): 初始化所有組件
- connect(): 連接 IB 並訂閱數據
- run(): 主運行循環
- shutdown(): 安全關閉

修改交易標的：
  編輯 LIVE_SYMBOLS 列表

修改策略參數：
  編輯 STRATEGY_CONFIG 字典
```

### run_backtest.py

```
這是回測的主程式入口。

使用方式：
  python run_backtest.py --strategy sma_cross --symbol AAPL --start 2024-01-01 --end 2024-12-01

參數：
  --strategy: 策略名稱
  --symbol: 交易標的
  --start: 開始日期
  --end: 結束日期
```

### main.py

```
備用主程式，提供更多命令列選項。
```

---

## 數據流向圖

```
┌─────────────────────────────────────────────────────────────────┐
│                         IB TWS                                   │
└─────────────────────────────┬───────────────────────────────────┘
                              │ Tick/Bar 數據
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      IBConnection                                │
│                   (core/connection.py)                           │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FeedHandler                                 │
│                   (data/feed_handler.py)                         │
│                                                                  │
│  - 接收原始數據                                                   │
│  - 轉換為 TickEvent/BarEvent                                     │
│  - 發布到 EventBus                                               │
└─────────────────────────────┬───────────────────────────────────┘
                              │ 發布事件
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        EventBus                                  │
│                   (core/event_bus.py)                            │
└───────┬─────────────────────┬─────────────────────┬─────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ MarketDataCache│   │ BarAggregator │    │StrategyEngine │
│ (data/cache.py)│   │(data/bar_agg.)│    │(engine/strat.)│
│               │    │               │    │               │
│ 快取最新數據   │    │ Tick→Bar聚合  │    │ 分發給策略    │
└───────────────┘    └───────┬───────┘    └───────┬───────┘
                             │                    │
                             │ 新的 BarEvent      │
                             └─────────┬──────────┘
                                       │
                                       ▼
                              ┌───────────────┐
                              │   Strategy    │
                              │ (strategies/) │
                              │               │
                              │ 計算信號      │
                              └───────┬───────┘
                                      │ SignalEvent
                                      ▼
                              ┌───────────────┐
                              │  RiskManager  │
                              │  (risk/)      │
                              │               │
                              │ 風險檢查      │
                              └───────┬───────┘
                                      │ 通過
                                      ▼
                              ┌───────────────┐
                              │PositionSizer  │
                              │  (risk/)      │
                              │               │
                              │ 計算倉位      │
                              └───────┬───────┘
                                      │
                                      ▼
                              ┌───────────────┐
                              │ExecutionEngine│
                              │  (engine/)    │
                              │               │
                              │ 建立並發送訂單│
                              └───────┬───────┘
                                      │ Order
                                      ▼
                              ┌───────────────┐
                              │   IB TWS      │
                              └───────┬───────┘
                                      │ Fill
                                      ▼
                              ┌───────────────┐
                              │ 更新持倉/帳戶 │
                              └───────────────┘
```

---

## 如何開發新策略

### 步驟 1：建立策略檔案

在 `strategies/examples/` 目錄下建立新檔案，例如 `rsi_strategy.py`：

```python
"""
RSI 策略範例
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from strategies.base import BaseStrategy, StrategyConfig
from core.events import BarEvent, OrderAction


@dataclass
class RSIConfig(StrategyConfig):
    """RSI 策略配置"""
    period: int = 14
    overbought: float = 70.0
    oversold: float = 30.0
    quantity: int = 100


class RSIStrategy(BaseStrategy):
    """
    RSI 策略
    
    - RSI < oversold (30) → 買入
    - RSI > overbought (70) → 賣出
    """
    
    def __init__(
        self,
        strategy_id: str = "rsi",
        symbols: Optional[List[str]] = None,
        period: int = 14,
        overbought: float = 70.0,
        oversold: float = 30.0,
        quantity: int = 100,
        **kwargs
    ):
        config = RSIConfig(
            period=period,
            overbought=overbought,
            oversold=oversold,
            quantity=quantity,
            symbols=symbols or []
        )
        
        super().__init__(
            strategy_id=strategy_id,
            symbols=symbols,
            config=config,
            **kwargs
        )
        
        self._period = period
        self._overbought = overbought
        self._oversold = oversold
        self._quantity = quantity
        
        # 儲存價格歷史
        self._prices: Dict[str, List[float]] = {}
    
    def on_bar(self, event: BarEvent) -> None:
        """處理 K線數據"""
        symbol = event.symbol
        
        if symbol not in self.symbols:
            return
        
        # 初始化價格列表
        if symbol not in self._prices:
            self._prices[symbol] = []
        
        # 新增收盤價
        self._prices[symbol].append(event.close)
        
        # 保留足夠的數據
        if len(self._prices[symbol]) > self._period + 1:
            self._prices[symbol] = self._prices[symbol][-(self._period + 1):]
        
        # 數據不足，跳過
        if len(self._prices[symbol]) < self._period + 1:
            return
        
        # 計算 RSI
        rsi = self._calculate_rsi(self._prices[symbol])
        
        # 交易邏輯
        if rsi < self._oversold:
            self.emit_signal(symbol, OrderAction.BUY, strength=1.0)
        elif rsi > self._overbought:
            self.emit_signal(symbol, OrderAction.SELL, strength=1.0)
    
    def _calculate_rsi(self, prices: List[float]) -> float:
        """計算 RSI"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-self._period:])
        avg_loss = np.mean(losses[-self._period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
```

### 步驟 2：註冊策略

在 `strategies/examples/__init__.py` 中加入：

```python
from .rsi_strategy import RSIStrategy
```

### 步驟 3：在 run_live.py 中使用

```python
from strategies.examples.rsi_strategy import RSIStrategy

# 在 _load_strategies 方法中加入：
rsi_strategy = RSIStrategy(
    strategy_id="rsi_live",
    symbols=self._symbols,
    period=14,
    overbought=70,
    oversold=30,
    quantity=20000
)
self._strategy_engine.add_strategy(rsi_strategy)
rsi_strategy.initialize()
rsi_strategy.start()
```

---

## 如何新增交易標的

### 新增外匯

編輯 `run_live.py` 中的 `LIVE_SYMBOLS`：

```python
LIVE_SYMBOLS = [
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    "AUD/USD",
    "USD/CHF",
    "NZD/USD",    # 新增
    "USD/CAD",    # 新增
]
```

### 新增股票（需訂閱數據）

```python
LIVE_SYMBOLS = [
    "AAPL",
    "MSFT",
    "GOOGL",
]
```

並修改 `_subscribe_symbols` 方法中的判斷邏輯。

### 新增期貨

需要修改 `_subscribe_symbols` 方法：

```python
# 在 _subscribe_symbols 中新增期貨判斷
if symbol.startswith("ES"):
    contract = self._contract_factory.future("ES", "202503")
```

---

## 常見修改場景

### 場景 1：修改策略參數

編輯 `run_live.py` 中的 `STRATEGY_CONFIG`：

```python
STRATEGY_CONFIG = {
    "sma_cross": {
        "enabled": True,
        "params": {
            "fast_period": 5,      # 改成 5
            "slow_period": 15,     # 改成 15
            "position_size": 10000, # 改成 10000
        },
    },
}
```

### 場景 2：新增風控限制

編輯 `risk/manager.py` 或在初始化時設定：

```python
self._risk_manager = RiskManager(
    config=settings.risk,
    event_bus=self._event_bus,
    account_value=100000,
)
# 可以調整 settings.risk 中的參數
```

### 場景 3：新增通知

編輯 `.env` 檔案設定 Telegram/Email：

```
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
```

### 場景 4：改成股票交易

1. 修改 `LIVE_SYMBOLS` 為股票代碼
2. 訂閱 IB 市場數據（付費）
3. 修改 `_subscribe_symbols` 邏輯

---

## 附錄：檔案快速索引

| 我想要... | 修改哪個檔案 |
|----------|-------------|
| 改交易標的 | `run_live.py` → LIVE_SYMBOLS |
| 改策略參數 | `run_live.py` → STRATEGY_CONFIG |
| 新增策略 | `strategies/examples/` 新增檔案 |
| 改風控設定 | `config/settings.py` 或 `.env` |
| 改連接設定 | `config/settings.py` 或 `.env` |
| 改日誌設定 | `utils/logger.py` |
| 改通知設定 | `utils/notifier.py` 和 `.env` |
| 改倉位計算 | `risk/position_sizer.py` |
| 改熔斷條件 | `risk/circuit_breaker.py` |

---

*文檔結束*
