"""
Trading Modes 模組 - 交易模式配置

定義不同交易風格的專屬參數配置：
- 高頻交易 (High Frequency)
- 低頻日內交易 (Low Frequency)
- 波段交易 (Swing)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List


class TradingMode(Enum):
    """交易模式枚舉"""
    
    HIGH_FREQUENCY = "high_frequency"
    LOW_FREQUENCY = "low_frequency"
    SWING = "swing"


@dataclass
class HighFrequencyConfig:
    """
    高頻交易配置
    
    適用於：Tick 級別策略、剝頭皮、做市策略
    特點：低延遲、高吞吐量、嚴格風控
    """
    
    # ========== 數據設定 ==========
    # 使用 Tick 數據
    use_tick_data: bool = True
    tick_buffer_size: int = 10000  # Tick 緩存數量
    
    # Bar 聚合（從 Tick 生成）
    aggregate_bars: bool = True
    bar_sizes: Tuple[str, ...] = ("1s", "5s", "1min")  # 聚合的 Bar 週期
    
    # ========== 執行設定 ==========
    # 訂單類型
    use_limit_orders: bool = True  # 優先使用限價單減少滑點
    max_slippage_ticks: int = 2  # 最大滑點（tick 數）
    order_timeout_seconds: int = 5  # 訂單超時時間
    
    # 快速取消未成交訂單
    cancel_unfilled_after_seconds: int = 3
    
    # ========== 性能設定 ==========
    # 內存快取
    use_memory_cache: bool = True
    cache_size_mb: int = 256
    
    # 異步處理
    async_order_submission: bool = True
    async_data_processing: bool = True
    
    # ========== 頻率限制 ==========
    # IB 限制：50 條消息/秒
    max_messages_per_second: int = 45  # 預留緩衝
    max_orders_per_second: int = 10
    max_cancels_per_second: int = 10
    
    # ========== 持倉設定 ==========
    max_position_hold_seconds: int = 300  # 最大持倉時間（5分鐘）
    force_close_on_timeout: bool = True
    
    # ========== 風控設定 ==========
    stop_loss_ticks: int = 10
    take_profit_ticks: int = 20
    max_daily_trades: int = 500
    max_consecutive_losses: int = 5  # 連續虧損熔斷


@dataclass
class LowFrequencyConfig:
    """
    低頻日內交易配置
    
    適用於：分鐘級別策略、日內趨勢、動量策略
    特點：穩定性優先、適中的交易頻率
    """
    
    # ========== 數據設定 ==========
    use_tick_data: bool = False  # 不需要 Tick 數據
    
    # Bar 設定
    primary_bar_size: str = "5 mins"  # 主要 Bar 週期
    secondary_bar_sizes: Tuple[str, ...] = ("15 mins", "1 hour")  # 輔助週期
    
    # 歷史數據
    lookback_bars: int = 100  # 載入的歷史 Bar 數量
    
    # ========== 執行設定 ==========
    use_limit_orders: bool = False  # 市價單可接受
    max_slippage_percent: float = 0.1  # 最大滑點百分比
    order_timeout_seconds: int = 60
    
    # ========== 性能設定 ==========
    use_memory_cache: bool = False  # 不需要高頻快取
    async_order_submission: bool = False
    
    # ========== 持倉設定 ==========
    max_position_hold_minutes: int = 240  # 最大持倉 4 小時
    close_before_market_close: bool = True  # 收盤前平倉
    minutes_before_close: int = 15  # 收盤前 15 分鐘平倉
    
    # ========== 風控設定 ==========
    stop_loss_percent: float = 2.0  # 停損百分比
    take_profit_percent: float = 4.0  # 停利百分比
    trailing_stop_percent: Optional[float] = 1.5  # 移動停損
    max_daily_trades: int = 20


@dataclass
class SwingConfig:
    """
    波段交易配置
    
    適用於：日線級別策略、趨勢跟蹤、價值投資
    特點：較長持倉週期、較大停損空間
    """
    
    # ========== 數據設定 ==========
    use_tick_data: bool = False
    
    # Bar 設定
    primary_bar_size: str = "1 day"  # 日線為主
    secondary_bar_sizes: Tuple[str, ...] = ("1 hour", "4 hours")  # 輔助週期
    
    # 歷史數據
    lookback_days: int = 60  # 載入 60 天歷史
    
    # ========== 執行設定 ==========
    use_limit_orders: bool = True  # 使用限價單
    limit_order_offset_percent: float = 0.05  # 限價單偏移
    order_timeout_seconds: int = 300  # 訂單超時 5 分鐘
    
    # 允許訂單跨日
    allow_overnight_orders: bool = True
    
    # ========== 加減碼設定 ==========
    scale_in_enabled: bool = True  # 允許分批進場
    scale_in_levels: int = 3  # 分 3 批進場
    scale_in_percent: Tuple[float, ...] = (0.4, 0.3, 0.3)  # 每批比例
    
    scale_out_enabled: bool = True  # 允許分批出場
    scale_out_levels: int = 3
    scale_out_percent: Tuple[float, ...] = (0.3, 0.3, 0.4)
    
    # ========== 持倉設定 ==========
    max_hold_days: Optional[int] = 30  # 最大持倉天數
    allow_overnight: bool = True  # 允許隔夜持倉
    
    # ========== 風控設定 ==========
    stop_loss_percent: float = 5.0  # 較大的停損空間
    take_profit_percent: float = 15.0
    trailing_stop_percent: Optional[float] = 3.0
    
    # ATR 相關
    use_atr_stops: bool = True  # 使用 ATR 計算停損
    atr_period: int = 14
    atr_multiplier: float = 2.0  # 停損 = 2 倍 ATR
    
    max_daily_trades: int = 5


def get_mode_config(mode: TradingMode):
    """
    取得指定交易模式的配置
    
    Args:
        mode: 交易模式
        
    Returns:
        對應的配置實例
    """
    configs = {
        TradingMode.HIGH_FREQUENCY: HighFrequencyConfig(),
        TradingMode.LOW_FREQUENCY: LowFrequencyConfig(),
        TradingMode.SWING: SwingConfig(),
    }
    return configs[mode]


def get_default_bar_size(mode: TradingMode) -> str:
    """取得交易模式的預設 Bar 週期"""
    bar_sizes = {
        TradingMode.HIGH_FREQUENCY: "1s",
        TradingMode.LOW_FREQUENCY: "5 mins",
        TradingMode.SWING: "1 day",
    }
    return bar_sizes[mode]


def get_recommended_settings(mode: TradingMode) -> dict:
    """
    取得交易模式的建議設定
    
    Returns:
        包含建議參數的字典
    """
    recommendations = {
        TradingMode.HIGH_FREQUENCY: {
            "description": "高頻交易模式",
            "min_capital": 50000,
            "recommended_symbols": ["ES", "NQ", "SPY", "QQQ"],
            "network_requirement": "低延遲網路，建議 < 10ms",
            "hardware_requirement": "高性能 CPU，16GB+ RAM",
        },
        TradingMode.LOW_FREQUENCY: {
            "description": "日內低頻交易模式",
            "min_capital": 25000,
            "recommended_symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            "network_requirement": "穩定網路連線",
            "hardware_requirement": "一般電腦即可",
        },
        TradingMode.SWING: {
            "description": "波段交易模式",
            "min_capital": 10000,
            "recommended_symbols": ["SPY", "QQQ", "IWM", "DIA"],
            "network_requirement": "穩定網路連線",
            "hardware_requirement": "一般電腦即可",
        },
    }
    return recommendations[mode]