"""
Backtest 模組 - 回測引擎

提供策略回測功能：
- BacktestEngine: 回測引擎
- DataLoader: 歷史數據載入
- 績效分析工具
"""

from .engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
)

from .data_loader import (
    DataLoader,
    DataSource,
    HistoricalBar,
)

__all__ = [
    # 回測引擎
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    # 數據載入
    "DataLoader",
    "DataSource",
    "HistoricalBar",
]