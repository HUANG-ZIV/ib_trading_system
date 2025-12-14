"""
Execution 模組 - 訂單執行層

提供進階訂單管理功能：
- OrderManager: 訂單管理器
- 複合訂單建立工具
"""

from .order_manager import (
    OrderManager,
    OrderInfo,
    OrderState,
)

from .order_types import (
    create_bracket_order,
    create_oco_order,
    create_trailing_stop,
    create_adaptive_order,
    create_twap_order,
)

__all__ = [
    # 訂單管理
    "OrderManager",
    "OrderInfo",
    "OrderState",
    # 訂單建立工具
    "create_bracket_order",
    "create_oco_order",
    "create_trailing_stop",
    "create_adaptive_order",
    "create_twap_order",
]