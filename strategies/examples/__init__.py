"""
Examples 模組 - 範例策略

提供可參考的策略實作範例：
- SMACrossStrategy: 簡單移動平均線交叉策略
- TickScalperStrategy: Tick 級別剝頭皮策略
"""

from .sma_cross import SMACrossStrategy
from .tick_scalper import TickScalperStrategy

__all__ = [
    "SMACrossStrategy",
    "TickScalperStrategy",
]