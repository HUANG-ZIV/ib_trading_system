"""
Events 模組 - 事件驅動架構的事件定義

定義系統中所有事件類型，用於模組間通訊
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, Dict, Any
import uuid


# ============================================================
# 枚舉定義
# ============================================================

class EventType(Enum):
    """事件類型枚舉"""
    
    # 市場數據事件
    TICK = auto()           # Tick 數據
    BAR = auto()            # K 線數據
    
    # 交易事件
    SIGNAL = auto()         # 策略信號
    ORDER = auto()          # 訂單事件
    FILL = auto()           # 成交事件
    
    # 狀態事件
    POSITION = auto()       # 持倉變化
    SYSTEM = auto()         # 系統事件
    RISK = auto()           # 風控事件


class OrderAction(Enum):
    """訂單方向"""
    
    BUY = "BUY"
    SELL = "SELL"
    
    # 期權/期貨用
    SHORT = "SSHORT"        # 賣空


class OrderType(Enum):
    """訂單類型"""
    
    MARKET = "MKT"          # 市價單
    LIMIT = "LMT"           # 限價單
    STOP = "STP"            # 停損單
    STOP_LIMIT = "STP LMT"  # 停損限價單
    TRAILING = "TRAIL"      # 追蹤停損單
    MOC = "MOC"             # 收盤市價單
    LOC = "LOC"             # 收盤限價單


class OrderStatus(Enum):
    """訂單狀態"""
    
    PENDING = "pending"         # 等待提交
    SUBMITTED = "submitted"     # 已提交
    ACCEPTED = "accepted"       # 已接受
    PARTIAL = "partial"         # 部分成交
    FILLED = "filled"           # 完全成交
    CANCELLED = "cancelled"     # 已取消
    REJECTED = "rejected"       # 被拒絕
    ERROR = "error"             # 錯誤


class SystemEventType(Enum):
    """系統事件子類型"""
    
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    STARTED = "started"
    STOPPED = "stopped"


class RiskEventType(Enum):
    """風控事件子類型"""
    
    DAILY_LOSS_EXCEEDED = "daily_loss_exceeded"
    POSITION_LIMIT_EXCEEDED = "position_limit_exceeded"
    EXPOSURE_LIMIT_EXCEEDED = "exposure_limit_exceeded"
    CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker_triggered"
    CIRCUIT_BREAKER_RESET = "circuit_breaker_reset"
    SIGNAL_REJECTED = "signal_rejected"


# ============================================================
# 事件基類
# ============================================================

@dataclass
class Event:
    """
    事件基類
    
    所有事件的父類，包含共通屬性
    """
    
    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    def __post_init__(self):
        """確保 timestamp 有值"""
        if self.timestamp is None:
            self.timestamp = datetime.now()


# ============================================================
# 市場數據事件
# ============================================================

@dataclass
class TickEvent(Event):
    """
    Tick 事件
    
    即時報價數據，包含 bid/ask/last 價格
    """
    
    symbol: str = ""
    
    # 價格
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    
    # 數量
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    last_size: Optional[int] = None
    volume: Optional[int] = None
    
    # 其他
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    
    def __post_init__(self):
        self.event_type = EventType.TICK
        super().__post_init__()
    
    @property
    def mid(self) -> Optional[float]:
        """中間價"""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return self.last
    
    @property
    def spread(self) -> Optional[float]:
        """買賣價差"""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None


@dataclass
class BarEvent(Event):
    """
    K 線事件
    
    OHLCV 數據，可來自 IB 或由 Tick 聚合
    """
    
    symbol: str = ""
    
    # OHLCV
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: int = 0
    
    # 時間框架
    bar_size: str = ""  # 如 "1 min", "5 mins", "1 day"
    bar_start: Optional[datetime] = None
    bar_end: Optional[datetime] = None
    
    # 額外資訊
    vwap: Optional[float] = None  # 成交量加權平均價
    trade_count: Optional[int] = None  # 成交筆數
    
    def __post_init__(self):
        self.event_type = EventType.BAR
        super().__post_init__()
    
    @property
    def typical_price(self) -> float:
        """典型價格 (H+L+C)/3"""
        return (self.high + self.low + self.close) / 3
    
    @property
    def range(self) -> float:
        """價格範圍"""
        return self.high - self.low
    
    @property
    def body(self) -> float:
        """K 棒實體"""
        return abs(self.close - self.open)
    
    @property
    def is_bullish(self) -> bool:
        """是否為陽線"""
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        """是否為陰線"""
        return self.close < self.open


# ============================================================
# 交易事件
# ============================================================

@dataclass
class SignalEvent(Event):
    """
    策略信號事件
    
    策略產生的交易信號，傳遞給執行引擎
    """
    
    symbol: str = ""
    action: OrderAction = OrderAction.BUY
    
    # 信號強度 (0.0 - 1.0)
    strength: float = 1.0
    
    # 建議參數
    suggested_quantity: Optional[int] = None
    suggested_price: Optional[float] = None
    suggested_order_type: OrderType = OrderType.MARKET
    
    # 停損停利
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # 來源
    strategy_id: str = ""
    reason: str = ""
    
    # 額外數據
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.event_type = EventType.SIGNAL
        super().__post_init__()


@dataclass
class OrderEvent(Event):
    """
    訂單事件
    
    訂單狀態變化通知
    """
    
    # 訂單識別
    order_id: int = 0
    client_order_id: str = ""
    perm_id: int = 0
    
    # 訂單內容
    symbol: str = ""
    action: OrderAction = OrderAction.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: int = 0
    
    # 價格
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    
    # 狀態
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    remaining_quantity: int = 0
    avg_fill_price: float = 0.0
    
    # 時間戳
    submitted_time: Optional[datetime] = None
    filled_time: Optional[datetime] = None
    
    # 錯誤資訊
    error_code: Optional[int] = None
    error_message: str = ""
    
    # 關聯
    parent_id: Optional[int] = None  # 父訂單 ID（用於括號訂單）
    strategy_id: str = ""
    
    def __post_init__(self):
        self.event_type = EventType.ORDER
        super().__post_init__()
    
    @property
    def is_active(self) -> bool:
        """訂單是否活躍"""
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIAL,
        ]
    
    @property
    def is_done(self) -> bool:
        """訂單是否已結束"""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.ERROR,
        ]


@dataclass
class FillEvent(Event):
    """
    成交事件
    
    訂單成交回報
    """
    
    # 訂單識別
    order_id: int = 0
    execution_id: str = ""
    
    # 成交內容
    symbol: str = ""
    action: OrderAction = OrderAction.BUY
    quantity: int = 0
    price: float = 0.0
    
    # 成本
    commission: float = 0.0
    
    # 時間
    execution_time: Optional[datetime] = None
    
    # 交易所資訊
    exchange: str = ""
    
    # 關聯
    strategy_id: str = ""
    
    def __post_init__(self):
        self.event_type = EventType.FILL
        super().__post_init__()
    
    @property
    def value(self) -> float:
        """成交金額"""
        return self.quantity * self.price
    
    @property
    def net_value(self) -> float:
        """扣除手續費後金額"""
        if self.action == OrderAction.BUY:
            return -(self.value + self.commission)
        else:
            return self.value - self.commission


# ============================================================
# 狀態事件
# ============================================================

@dataclass
class PositionEvent(Event):
    """
    持倉事件
    
    持倉狀態變化通知
    """
    
    symbol: str = ""
    
    # 持倉數量（正數為多頭，負數為空頭）
    quantity: int = 0
    
    # 成本
    avg_cost: float = 0.0
    
    # 市值
    market_price: Optional[float] = None
    market_value: Optional[float] = None
    
    # 盈虧
    unrealized_pnl: Optional[float] = None
    realized_pnl: Optional[float] = None
    
    # 帳戶
    account: str = ""
    
    def __post_init__(self):
        self.event_type = EventType.POSITION
        super().__post_init__()
    
    @property
    def is_long(self) -> bool:
        """是否為多頭持倉"""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """是否為空頭持倉"""
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        """是否無持倉"""
        return self.quantity == 0


@dataclass
class SystemEvent(Event):
    """
    系統事件
    
    系統狀態、連接狀態、錯誤等通知
    """
    
    sub_type: SystemEventType = SystemEventType.INFO
    message: str = ""
    
    # 錯誤相關
    error_code: Optional[int] = None
    
    # 來源
    source: str = ""  # 如 "connection", "engine", "strategy"
    
    # 額外數據
    data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.event_type = EventType.SYSTEM
        super().__post_init__()
    
    @property
    def is_error(self) -> bool:
        """是否為錯誤事件"""
        return self.sub_type == SystemEventType.ERROR
    
    @property
    def is_connection_event(self) -> bool:
        """是否為連接相關事件"""
        return self.sub_type in [
            SystemEventType.CONNECTED,
            SystemEventType.DISCONNECTED,
            SystemEventType.RECONNECTING,
        ]


@dataclass
class RiskEvent(Event):
    """
    風控事件
    
    風控規則觸發通知
    """
    
    sub_type: RiskEventType = RiskEventType.SIGNAL_REJECTED
    message: str = ""
    
    # 觸發相關
    symbol: Optional[str] = None
    strategy_id: Optional[str] = None
    
    # 數值
    current_value: Optional[float] = None  # 當前值
    limit_value: Optional[float] = None    # 限制值
    
    # 動作
    action_taken: str = ""  # 如 "order_rejected", "trading_disabled"
    
    def __post_init__(self):
        self.event_type = EventType.RISK
        super().__post_init__()


# ============================================================
# 工廠函數
# ============================================================

def create_tick_event(
    symbol: str,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
    last: Optional[float] = None,
    **kwargs
) -> TickEvent:
    """建立 TickEvent 的便捷函數"""
    return TickEvent(
        event_type=EventType.TICK,
        symbol=symbol,
        bid=bid,
        ask=ask,
        last=last,
        **kwargs
    )


def create_bar_event(
    symbol: str,
    open: float,
    high: float,
    low: float,
    close: float,
    volume: int = 0,
    bar_size: str = "",
    **kwargs
) -> BarEvent:
    """建立 BarEvent 的便捷函數"""
    return BarEvent(
        event_type=EventType.BAR,
        symbol=symbol,
        open=open,
        high=high,
        low=low,
        close=close,
        volume=volume,
        bar_size=bar_size,
        **kwargs
    )


def create_signal_event(
    symbol: str,
    action: OrderAction,
    strategy_id: str,
    strength: float = 1.0,
    **kwargs
) -> SignalEvent:
    """建立 SignalEvent 的便捷函數"""
    return SignalEvent(
        event_type=EventType.SIGNAL,
        symbol=symbol,
        action=action,
        strategy_id=strategy_id,
        strength=strength,
        **kwargs
    )