"""
三角套利策略配置
Triangular Arbitrage Strategy Configuration
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from enum import Enum


class AssetType(Enum):
    """資產類型"""
    SPOT = "spot"
    FUTURES = "futures"


class TriangleType(Enum):
    """三角類型"""
    T1_XAU_XAG_XPT = "XAU-XAG-XPT"  # 金銀鉑
    T2_XAU_XAG_XPD = "XAU-XAG-XPD"  # 金銀鈀
    T3_XAU_XPT_XPD = "XAU-XPT-XPD"  # 金鉑鈀
    T4_XAG_XPT_XPD = "XAG-XPT-XPD"  # 銀鉑鈀


@dataclass
class TriangleDefinition:
    """三角定義"""
    name: str
    asset_a: str  # 分子商品
    asset_b: str  # 中間商品
    asset_c: str  # 分母商品
    
    # 歷史統計（可更新）
    mean_deviation: float = 0.0
    std_deviation: float = 0.01
    half_life_days: float = 5.0
    
    # 流動性評級 (1-5, 5最好)
    liquidity_score: int = 3


# 現貨標的定義
SPOT_SYMBOLS = {
    "XAU": "XAUUSD",
    "XAG": "XAGUSD",
    "XPT": "XPTUSD",
    "XPD": "XPDUSD",
}

# 期貨標的定義
FUTURES_SYMBOLS = {
    "XAU": {"symbol": "GC", "exchange": "COMEX", "multiplier": 100},
    "XAG": {"symbol": "SI", "exchange": "COMEX", "multiplier": 5000},
    "XPT": {"symbol": "PL", "exchange": "NYMEX", "multiplier": 50},
    "XPD": {"symbol": "PA", "exchange": "NYMEX", "multiplier": 100},
}

# 三角組合定義
TRIANGLE_DEFINITIONS = {
    TriangleType.T1_XAU_XAG_XPT: TriangleDefinition(
        name="Gold-Silver-Platinum",
        asset_a="XAU",
        asset_b="XAG",
        asset_c="XPT",
        mean_deviation=0.0002,
        std_deviation=0.008,
        half_life_days=5.0,
        liquidity_score=5,
    ),
    TriangleType.T2_XAU_XAG_XPD: TriangleDefinition(
        name="Gold-Silver-Palladium",
        asset_a="XAU",
        asset_b="XAG",
        asset_c="XPD",
        mean_deviation=0.0005,
        std_deviation=0.015,
        half_life_days=10.0,
        liquidity_score=3,
    ),
    TriangleType.T3_XAU_XPT_XPD: TriangleDefinition(
        name="Gold-Platinum-Palladium",
        asset_a="XAU",
        asset_b="XPT",
        asset_c="XPD",
        mean_deviation=0.0003,
        std_deviation=0.012,
        half_life_days=8.0,
        liquidity_score=3,
    ),
    TriangleType.T4_XAG_XPT_XPD: TriangleDefinition(
        name="Silver-Platinum-Palladium",
        asset_a="XAG",
        asset_b="XPT",
        asset_c="XPD",
        mean_deviation=0.0004,
        std_deviation=0.014,
        half_life_days=10.0,
        liquidity_score=2,
    ),
}


@dataclass
class TriangularArbitrageConfig:
    """三角套利策略配置"""
    
    # === 策略基本設定 ===
    strategy_id: str = "triangular_arbitrage"
    enabled_triangles: List[TriangleType] = field(default_factory=lambda: [
        TriangleType.T1_XAU_XAG_XPT,
        TriangleType.T2_XAU_XAG_XPD,
    ])
    
    # === 進出場參數 ===
    lookback_period: int = 120  # Z-Score 計算回顧期（天）
    entry_zscore: float = 2.0   # 進場 Z-Score 門檻
    exit_zscore: float = 0.5    # 出場 Z-Score 門檻
    stop_zscore: float = 3.5    # 停損 Z-Score 門檻
    min_deviation_pct: float = 0.5  # 最小偏離百分比 (%)
    max_holding_days: int = 20  # 最大持倉天數
    
    # === 資金管理 ===
    capital_per_triangle: float = 50000  # 每個三角的資金 (USD)
    max_triangles: int = 3      # 同時最多持有的三角數
    max_exposure_pct: float = 0.5  # 最大總曝險 (佔總資金)
    
    # === 風險管理 ===
    single_leg_stop_pct: float = 1.5   # 單腿停損百分比
    daily_loss_limit_pct: float = 2.0  # 每日最大虧損百分比
    position_scale_out: List[Tuple[float, float]] = field(default_factory=lambda: [
        (2.5, 0.33),  # Z > 2.5 時減倉 33%
        (3.0, 0.33),  # Z > 3.0 時再減倉 33%
        (3.5, 1.0),   # Z > 3.5 時全部平倉
    ])
    
    # === 執行設定 ===
    use_futures: bool = False   # 是否使用期貨
    prefer_liquid: bool = True  # 優先選擇流動性好的三角
    execution_delay_ms: int = 100  # 各腿執行間隔（毫秒）
    
    # === 時間過濾 ===
    trading_hours_utc: Tuple[int, int] = (7, 21)  # 交易時段 (UTC)
    avoid_news_minutes: int = 30  # 重大新聞前後避開（分鐘）
    
    # === 數據設定 ===
    bar_size: str = "1 hour"    # K線週期
    warmup_bars: int = 150      # 預熱所需的 K 線數量


@dataclass 
class BacktestConfig:
    """回測配置"""
    
    # === 時間範圍 ===
    start_date: str = "2015-01-01"
    end_date: str = "2024-12-31"
    in_sample_end: str = "2020-12-31"
    
    # === 初始資金 ===
    initial_capital: float = 500000
    
    # === 成本設定 ===
    spot_spread_pct: float = 0.002   # 現貨點差 0.2%
    futures_commission: float = 2.5   # 期貨手續費 (USD/合約)
    slippage_pct: float = 0.0001     # 滑價 0.01%
    
    # === 其他 ===
    risk_free_rate: float = 0.05     # 無風險利率（計算夏普用）


# 預設配置實例
DEFAULT_CONFIG = TriangularArbitrageConfig()
DEFAULT_BACKTEST_CONFIG = BacktestConfig()
