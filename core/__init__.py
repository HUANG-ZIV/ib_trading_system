"""
Core 模組 - 系統核心組件

提供事件驅動架構的基礎設施：
- 事件定義與事件總線
- IB 連接管理
- 合約工廠
"""

from .events import (
    # 枚舉
    EventType,
    OrderAction,
    OrderType,
    OrderStatus,
    # 事件類
    Event,
    TickEvent,
    BarEvent,
    SignalEvent,
    OrderEvent,
    FillEvent,
    PositionEvent,
    SystemEvent,
    RiskEvent,
)

from .event_bus import (
    EventBus,
    get_event_bus,
)

from .connection import (
    IBConnection,
    ConnectionState,
)

from .contracts import (
    ContractFactory,
    qualify_contract,
    qualify_contract_async,
)

__all__ = [
    # 事件枚舉
    "EventType",
    "OrderAction",
    "OrderType",
    "OrderStatus",
    # 事件類
    "Event",
    "TickEvent",
    "BarEvent",
    "SignalEvent",
    "OrderEvent",
    "FillEvent",
    "PositionEvent",
    "SystemEvent",
    "RiskEvent",
    # 事件總線
    "EventBus",
    "get_event_bus",
    # 連接
    "IBConnection",
    "ConnectionState",
    # 合約
    "ContractFactory",
    "qualify_contract",
    "qualify_contract_async",
]