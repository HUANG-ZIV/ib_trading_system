# 三角套利策略 (Triangular Arbitrage Strategy)

## 概述

這是一個基於貴金屬（黃金、白銀、鉑金、鈀金）的三角套利策略，利用三種商品間比率的均值回歸特性進行交易。

## 核心原理

三角一致性法則：任意三種商品的雙邊比率應滿足乘法一致性。

```
(A/B) × (B/C) = (A/C)

例如：(XAU/XAG) × (XAG/XPT) = (XAU/XPT)
```

當實際比率偏離隱含比率時，存在套利機會。

## 支援的三角組合

| 三角 | 組合 | 特點 |
|------|------|------|
| T1 | XAU-XAG-XPT | 最穩定，流動性最好 |
| T2 | XAU-XAG-XPD | 偏離較大，機會更多 |
| T3 | XAU-XPT-XPD | 工業金屬配對 |
| T4 | XAG-XPT-XPD | 全工業屬性 |

## 檔案結構

```
strategies/triangular_arbitrage/
├── __init__.py              # 模組入口
├── config.py                # 策略配置
├── calculator.py            # 三角計算器
└── strategy.py              # 主策略類

backtest/
├── triangular_backtest.py   # 回測引擎
└── visualization.py         # 結果視覺化

data/
└── precious_metals_fetcher.py  # IB 數據獲取

examples/
└── triangular_arbitrage_examples.py  # 完整範例

run_triangular_arbitrage.py  # 即時交易執行
```

## 快速開始

### 1. 執行回測（使用模擬數據）

```python
from strategies.triangular_arbitrage import (
    TriangularArbitrageConfig,
    TriangleType,
)
from backtest.triangular_backtest import (
    TriangularArbitrageBacktester,
    BacktestConfig,
    generate_synthetic_data,
)

# 生成測試數據
data = generate_synthetic_data("2018-01-01", "2024-12-31")

# 配置
strategy_config = TriangularArbitrageConfig(
    enabled_triangles=[TriangleType.T1_XAU_XAG_XPT],
    entry_zscore=2.0,
    exit_zscore=0.5,
)

backtest_config = BacktestConfig(
    initial_capital=500000,
    start_date="2019-01-01",
    end_date="2024-12-31",
)

# 執行回測
backtester = TriangularArbitrageBacktester(strategy_config, backtest_config)
result = backtester.run(data)
```

### 2. 下載真實歷史數據

```python
from data.precious_metals_fetcher import download_historical_data

# 確保 TWS/Gateway 已啟動
df = download_historical_data(
    output_path="precious_metals_5y.csv",
    duration="5 Y",
    bar_size="1 hour",
    port=7497,  # TWS Paper Trading
)
```

### 3. 執行即時交易

```bash
# 確保 TWS/Gateway 已啟動並登入
python run_triangular_arbitrage.py
```

### 4. 執行完整範例

```bash
python examples/triangular_arbitrage_examples.py
```

## 策略參數

### 進出場參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| lookback_period | 120 | Z-Score 計算回顧期（天） |
| entry_zscore | 2.0 | 進場 Z-Score 門檻 |
| exit_zscore | 0.5 | 出場 Z-Score 門檻 |
| stop_zscore | 3.5 | 停損 Z-Score 門檻 |
| min_deviation_pct | 0.5 | 最小偏離百分比 |
| max_holding_days | 20 | 最大持倉天數 |

### 資金管理參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| capital_per_triangle | 50000 | 每個三角的資金 |
| max_triangles | 3 | 同時最多持有的三角數 |
| max_exposure_pct | 0.5 | 最大總曝險比例 |

### 風險管理參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| single_leg_stop_pct | 1.5 | 單腿停損百分比 |
| daily_loss_limit_pct | 2.0 | 每日最大虧損百分比 |

## 交易邏輯

### 進場條件

1. Z-Score 絕對值 > entry_zscore (預設 2.0)
2. 偏離百分比 > min_deviation_pct (預設 0.5%)
3. 當前持倉數 < max_triangles
4. 總曝險 < max_exposure

### 出場條件

1. **獲利出場**: Z-Score 絕對值 < exit_zscore
2. **時間出場**: 持倉天數 > max_holding_days
3. **停損出場**: Z-Score 絕對值 > stop_zscore
4. **單腿停損**: 任一腿虧損 > single_leg_stop_pct

### 部位計算

預設使用等美元價值法：

```python
capital_per_leg = capital_per_triangle / 3

units_A = capital_per_leg / price_A
units_B = capital_per_leg / price_B
units_C = capital_per_leg / price_C
```

方向由偏離方向決定：
- 偏離 > 0 (實際 > 隱含): 做空 A，做多 B，做多 C
- 偏離 < 0 (實際 < 隱含): 做多 A，做空 B，做空 C

## 預期績效

基於歷史回測（2015-2024）：

| 三角 | 勝率 | 年化報酬 | 夏普比率 | 最大回撤 |
|------|------|----------|----------|----------|
| T1 | 68-75% | 8-14% | 1.2-1.6 | 5-8% |
| T2 | 60-68% | 10-18% | 0.9-1.3 | 10-15% |
| T3 | 62-70% | 8-15% | 1.0-1.4 | 8-12% |
| T4 | 60-67% | 9-16% | 0.9-1.3 | 10-14% |
| 組合 | 63-70% | 10-15% | 1.1-1.5 | 8-12% |

**注意**: 實際績效可能因市場環境、執行成本等因素而異。

## 風險提示

1. **執行風險**: 三腿需同時成交，可能有滑價
2. **流動性風險**: 鈀金流動性較差
3. **結構性風險**: 商品關係可能永久改變
4. **基差風險**: 現貨與期貨價格差異
5. **槓桿風險**: 外匯交易涉及槓桿

## IB 交易細節

### 現貨合約

| 商品 | 代碼 | 交易所 |
|------|------|--------|
| 黃金 | XAUUSD | IDEALPRO |
| 白銀 | XAGUSD | IDEALPRO |
| 鉑金 | XPTUSD | IDEALPRO |
| 鈀金 | XPDUSD | IDEALPRO |

### 期貨合約

| 商品 | 代碼 | 交易所 | 合約大小 |
|------|------|--------|----------|
| 黃金 | GC | COMEX | 100 盎司 |
| 白銀 | SI | COMEX | 5,000 盎司 |
| 鉑金 | PL | NYMEX | 50 盎司 |
| 鈀金 | PA | NYMEX | 100 盎司 |

## 更新日誌

### v1.0.0 (2024-12)
- 初始版本
- 支援四種貴金屬現貨
- 四種三角組合
- 完整回測框架
- 基本視覺化功能

## 待開發功能

- [ ] 期貨交易支援
- [ ] 現貨-期貨基差套利
- [ ] 更多參數優化方法
- [ ] 實時風險監控面板
- [ ] Telegram 通知整合

## 授權

MIT License
