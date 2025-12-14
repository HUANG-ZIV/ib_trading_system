"""
Config 模組 - 系統配置管理

提供全局設定、交易模式配置、交易標的定義
"""

from .settings import Settings, get_settings
from .trading_modes import (
    TradingMode,
    HighFrequencyConfig,
    LowFrequencyConfig,
    SwingConfig,
    get_mode_config,
)
from .symbols import (
    SymbolConfig,
    SecurityType,
    create_stock,
    create_future,
    create_option,
    get_watchlist,
    US_STOCKS,
    US_FUTURES,
)

__all__ = [
    # 全局設定
    "Settings",
    "get_settings",
    # 交易模式
    "TradingMode",
    "HighFrequencyConfig",
    "LowFrequencyConfig",
    "SwingConfig",
    "get_mode_config",
    # 交易標的
    "SymbolConfig",
    "SecurityType",
    "create_stock",
    "create_future",
    "create_option",
    "get_watchlist",
    "US_STOCKS",
    "US_FUTURES",
]