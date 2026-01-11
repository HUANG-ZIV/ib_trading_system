"""
三角套利策略模組
Triangular Arbitrage Strategy Module
"""

from .config import (
    TriangularArbitrageConfig,
    BacktestConfig,
    TriangleType,
    TriangleDefinition,
    AssetType,
    TRIANGLE_DEFINITIONS,
    SPOT_SYMBOLS,
    FUTURES_SYMBOLS,
    DEFAULT_CONFIG,
    DEFAULT_BACKTEST_CONFIG,
)

from .calculator import (
    TriangleCalculator,
    TrianglePositionManager,
    TriangleSignal,
    TriangleState,
)

from .strategy import (
    TriangularArbitrageStrategy,
    create_strategy,
)


__all__ = [
    # Config
    "TriangularArbitrageConfig",
    "BacktestConfig", 
    "TriangleType",
    "TriangleDefinition",
    "AssetType",
    "TRIANGLE_DEFINITIONS",
    "SPOT_SYMBOLS",
    "FUTURES_SYMBOLS",
    "DEFAULT_CONFIG",
    "DEFAULT_BACKTEST_CONFIG",
    
    # Calculator
    "TriangleCalculator",
    "TrianglePositionManager",
    "TriangleSignal",
    "TriangleState",
    
    # Strategy
    "TriangularArbitrageStrategy",
    "create_strategy",
]
