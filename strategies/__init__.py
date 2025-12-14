"""
Strategies 模組 - 交易策略層

提供策略開發的基礎設施：
- BaseStrategy: 策略基類
- StrategyRegistry: 策略註冊表
- 範例策略
"""

from .base import (
    BaseStrategy,
    StrategyState,
    StrategyConfig,
)

from .registry import (
    StrategyRegistry,
    get_registry,
    register_strategy,
)

__all__ = [
    # 策略基類
    "BaseStrategy",
    "StrategyState",
    "StrategyConfig",
    # 註冊表
    "StrategyRegistry",
    "get_registry",
    "register_strategy",
]