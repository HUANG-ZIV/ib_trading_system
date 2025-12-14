"""
OrderManager 模組 - 訂單管理器

管理訂單生命週期和狀態追蹤
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, Dict, List, Any, Callable
import threading

from core.events import (
    Event,
    EventType,
    OrderEvent,
    FillEvent,
    OrderAction,
    OrderType,
    OrderStatus,
)
from core.event_bus import EventBus, get_event_bus


# 設定 logger
logger = logging.getLogger(__name__)


class OrderState(Enum):
    """訂單狀態（更詳細的內部狀態）"""
    
    CREATED = auto()          # 已建立（未提交）
    PENDING_SUBMIT = auto()   # 等待提交
    SUBMITTED = auto()        # 已提交
    ACKNOWLEDGED = auto()     # 已確認（交易所收到）
    PARTIAL_FILLED = auto()   # 部分成交
    FILLED = auto()           # 全部成交
    PENDING_CANCEL = auto()   # 等待取消
    CANCELLED = auto()        # 已取消
    REJECTED = auto()         # 已拒絕
    EXPIRED = auto()          # 已過期
    ERROR = auto()            # 錯誤


@dataclass
class OrderInfo:
    """訂單資訊"""
    
    # 基本資訊
    order_id: int
    client_order_id: str = ""
    symbol: str = ""
    action: OrderAction = OrderAction.BUY
    order_type: OrderType = OrderType.MARKET
    
    # 數量和價格
    quantity: int = 0
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    
    # 狀態
    state: OrderState = OrderState.CREATED
    status: OrderStatus = OrderStatus.PENDING
    
    # 成交資訊
    filled_quantity: int = 0
    remaining_quantity: int = 0
    avg_fill_price: float = 0.0
    last_fill_price: float = 0.0
    last_fill_quantity: int = 0
    
    # 費用
    commission: float = 0.0
    
    # 時間戳
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    first_fill_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    updated_at: datetime = field(default_factory=datetime.now)
    
    # 來源資訊
    strategy_id: str = ""
    signal_id: str = ""
    
    # 關聯訂單
    parent_order_id: Optional[int] = None
    child_order_ids: List[int] = field(default_factory=list)
    
    # 錯誤資訊
    error_code: Optional[int] = None
    error_message: str = ""
    
    # 額外資訊
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # ========== 屬性 ==========
    
    @property
    def is_active(self) -> bool:
        """是否活躍（未完成）"""
        return self.state in [
            OrderState.CREATED,
            OrderState.PENDING_SUBMIT,
            OrderState.SUBMITTED,
            OrderState.ACKNOWLEDGED,
            OrderState.PARTIAL_FILLED,
            OrderState.PENDING_CANCEL,
        ]
    
    @property
    def is_completed(self) -> bool:
        """是否已完成"""
        return self.state in [
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
            OrderState.ERROR,
        ]
    
    @property
    def is_filled(self) -> bool:
        """是否已成交"""
        return self.state == OrderState.FILLED
    
    @property
    def is_cancelled(self) -> bool:
        """是否已取消"""
        return self.state == OrderState.CANCELLED
    
    @property
    def is_rejected(self) -> bool:
        """是否被拒絕"""
        return self.state == OrderState.REJECTED
    
    @property
    def fill_ratio(self) -> float:
        """成交比例"""
        if self.quantity == 0:
            return 0.0
        return self.filled_quantity / self.quantity
    
    @property
    def total_value(self) -> float:
        """訂單總價值"""
        price = self.limit_price or self.avg_fill_price or 0
        return self.quantity * price
    
    @property
    def filled_value(self) -> float:
        """已成交價值"""
        return self.filled_quantity * self.avg_fill_price
    
    @property
    def net_value(self) -> float:
        """淨價值（扣除佣金）"""
        return self.filled_value - self.commission
    
    @property
    def duration(self) -> Optional[float]:
        """訂單持續時間（秒）"""
        if self.submitted_at is None:
            return None
        
        end_time = self.filled_at or self.cancelled_at or datetime.now()
        return (end_time - self.submitted_at).total_seconds()
    
    # ========== 方法 ==========
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "action": self.action.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "state": self.state.name,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "avg_fill_price": self.avg_fill_price,
            "commission": self.commission,
            "fill_ratio": f"{self.fill_ratio:.2%}",
            "strategy_id": self.strategy_id,
            "created_at": self.created_at.isoformat(),
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "duration": self.duration,
        }


@dataclass
class OrderStats:
    """訂單統計"""
    
    total_orders: int = 0
    active_orders: int = 0
    filled_orders: int = 0
    cancelled_orders: int = 0
    rejected_orders: int = 0
    
    total_filled_quantity: int = 0
    total_filled_value: float = 0.0
    total_commission: float = 0.0
    
    avg_fill_time: float = 0.0  # 平均成交時間（秒）


class OrderManager:
    """
    訂單管理器
    
    管理訂單的生命週期，追蹤訂單狀態變化
    
    使用方式:
        manager = OrderManager(event_bus)
        manager.start()
        
        # 添加訂單
        order_info = manager.add_order(
            order_id=12345,
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=100,
        )
        
        # 更新訂單狀態
        manager.update_order(12345, state=OrderState.FILLED)
        
        # 查詢訂單
        order = manager.get_order(12345)
        pending = manager.get_pending_orders()
    """
    
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        max_history: int = 1000,
    ):
        """
        初始化訂單管理器
        
        Args:
            event_bus: 事件總線
            max_history: 保留的歷史訂單數量
        """
        self._event_bus = event_bus or get_event_bus()
        self._max_history = max_history
        
        # 訂單存儲
        self._orders: Dict[int, OrderInfo] = {}
        self._completed_orders: List[OrderInfo] = []
        
        # 索引
        self._orders_by_symbol: Dict[str, List[int]] = {}
        self._orders_by_strategy: Dict[str, List[int]] = {}
        
        # 統計
        self._stats = OrderStats()
        
        # 運行狀態
        self._running = False
        
        # 線程安全
        self._lock = threading.RLock()
        
        # 回調
        self._on_state_change_callbacks: List[Callable[[OrderInfo, OrderState], None]] = []
        self._on_fill_callbacks: List[Callable[[OrderInfo, FillEvent], None]] = []
        
        logger.info("OrderManager 初始化完成")
    
    # ========== 控制 ==========
    
    def start(self) -> None:
        """啟動訂單管理器"""
        if self._running:
            logger.warning("OrderManager 已在運行中")
            return
        
        # 訂閱事件
        self._event_bus.subscribe(EventType.ORDER, self._on_order_event, priority=5)
        self._event_bus.subscribe(EventType.FILL, self._on_fill_event, priority=5)
        
        self._running = True
        logger.info("OrderManager 已啟動")
    
    def stop(self) -> None:
        """停止訂單管理器"""
        if not self._running:
            return
        
        self._event_bus.unsubscribe(EventType.ORDER, self._on_order_event)
        self._event_bus.unsubscribe(EventType.FILL, self._on_fill_event)
        
        self._running = False
        logger.info("OrderManager 已停止")
    
    # ========== 訂單管理 ==========
    
    def add_order(
        self,
        order_id: int,
        symbol: str,
        action: OrderAction,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        strategy_id: str = "",
        client_order_id: str = "",
        parent_order_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OrderInfo:
        """
        添加訂單
        
        Args:
            order_id: 訂單 ID
            symbol: 標的代碼
            action: 買賣方向
            quantity: 數量
            order_type: 訂單類型
            limit_price: 限價
            stop_price: 停損價
            strategy_id: 策略 ID
            client_order_id: 客戶訂單 ID
            parent_order_id: 父訂單 ID
            metadata: 額外資訊
            
        Returns:
            OrderInfo
        """
        with self._lock:
            # 建立訂單資訊
            order_info = OrderInfo(
                order_id=order_id,
                client_order_id=client_order_id or f"ord_{order_id}",
                symbol=symbol,
                action=action,
                order_type=order_type,
                quantity=quantity,
                remaining_quantity=quantity,
                limit_price=limit_price,
                stop_price=stop_price,
                strategy_id=strategy_id,
                parent_order_id=parent_order_id,
                metadata=metadata or {},
            )
            
            # 存儲訂單
            self._orders[order_id] = order_info
            
            # 建立索引
            if symbol not in self._orders_by_symbol:
                self._orders_by_symbol[symbol] = []
            self._orders_by_symbol[symbol].append(order_id)
            
            if strategy_id:
                if strategy_id not in self._orders_by_strategy:
                    self._orders_by_strategy[strategy_id] = []
                self._orders_by_strategy[strategy_id].append(order_id)
            
            # 更新統計
            self._stats.total_orders += 1
            self._stats.active_orders += 1
            
            logger.debug(f"添加訂單: {order_id} {symbol} {action.value} {quantity}")
            
            return order_info
    
    def update_order(
        self,
        order_id: int,
        state: Optional[OrderState] = None,
        status: Optional[OrderStatus] = None,
        filled_quantity: Optional[int] = None,
        avg_fill_price: Optional[float] = None,
        last_fill_price: Optional[float] = None,
        last_fill_quantity: Optional[int] = None,
        commission: Optional[float] = None,
        error_code: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> Optional[OrderInfo]:
        """
        更新訂單狀態
        
        Args:
            order_id: 訂單 ID
            state: 新狀態
            status: 新的外部狀態
            filled_quantity: 已成交數量
            avg_fill_price: 平均成交價
            last_fill_price: 最新成交價
            last_fill_quantity: 最新成交數量
            commission: 佣金
            error_code: 錯誤代碼
            error_message: 錯誤訊息
            
        Returns:
            更新後的 OrderInfo 或 None
        """
        with self._lock:
            order_info = self._orders.get(order_id)
            if order_info is None:
                logger.warning(f"訂單 {order_id} 不存在")
                return None
            
            old_state = order_info.state
            now = datetime.now()
            
            # 更新狀態
            if state is not None:
                order_info.state = state
                
                # 更新時間戳
                if state == OrderState.SUBMITTED:
                    order_info.submitted_at = now
                elif state == OrderState.ACKNOWLEDGED:
                    order_info.acknowledged_at = now
                elif state == OrderState.PARTIAL_FILLED:
                    if order_info.first_fill_at is None:
                        order_info.first_fill_at = now
                elif state == OrderState.FILLED:
                    order_info.filled_at = now
                    if order_info.first_fill_at is None:
                        order_info.first_fill_at = now
                elif state == OrderState.CANCELLED:
                    order_info.cancelled_at = now
            
            if status is not None:
                order_info.status = status
            
            # 更新成交資訊
            if filled_quantity is not None:
                order_info.filled_quantity = filled_quantity
                order_info.remaining_quantity = order_info.quantity - filled_quantity
            
            if avg_fill_price is not None:
                order_info.avg_fill_price = avg_fill_price
            
            if last_fill_price is not None:
                order_info.last_fill_price = last_fill_price
            
            if last_fill_quantity is not None:
                order_info.last_fill_quantity = last_fill_quantity
            
            if commission is not None:
                order_info.commission = commission
            
            # 更新錯誤資訊
            if error_code is not None:
                order_info.error_code = error_code
            
            if error_message is not None:
                order_info.error_message = error_message
            
            order_info.updated_at = now
            
            # 處理完成的訂單
            if order_info.is_completed and old_state != order_info.state:
                self._handle_completed_order(order_info)
            
            # 執行狀態變化回調
            if state is not None and state != old_state:
                for callback in self._on_state_change_callbacks:
                    try:
                        callback(order_info, old_state)
                    except Exception as e:
                        logger.error(f"狀態變化回調錯誤: {e}")
            
            logger.debug(
                f"更新訂單: {order_id} state={order_info.state.name} "
                f"filled={order_info.filled_quantity}/{order_info.quantity}"
            )
            
            return order_info
    
    def _handle_completed_order(self, order_info: OrderInfo) -> None:
        """處理已完成的訂單"""
        # 更新統計
        self._stats.active_orders -= 1
        
        if order_info.is_filled:
            self._stats.filled_orders += 1
            self._stats.total_filled_quantity += order_info.filled_quantity
            self._stats.total_filled_value += order_info.filled_value
            self._stats.total_commission += order_info.commission
            
            # 更新平均成交時間
            if order_info.duration:
                total_time = self._stats.avg_fill_time * (self._stats.filled_orders - 1)
                self._stats.avg_fill_time = (total_time + order_info.duration) / self._stats.filled_orders
        
        elif order_info.is_cancelled:
            self._stats.cancelled_orders += 1
        
        elif order_info.is_rejected:
            self._stats.rejected_orders += 1
        
        # 移動到已完成列表
        self._completed_orders.append(order_info)
        
        # 限制歷史數量
        if len(self._completed_orders) > self._max_history:
            self._completed_orders = self._completed_orders[-self._max_history:]
    
    def remove_order(self, order_id: int) -> bool:
        """
        移除訂單
        
        Args:
            order_id: 訂單 ID
            
        Returns:
            是否成功移除
        """
        with self._lock:
            if order_id not in self._orders:
                return False
            
            order_info = self._orders[order_id]
            
            # 從索引移除
            if order_info.symbol in self._orders_by_symbol:
                if order_id in self._orders_by_symbol[order_info.symbol]:
                    self._orders_by_symbol[order_info.symbol].remove(order_id)
            
            if order_info.strategy_id in self._orders_by_strategy:
                if order_id in self._orders_by_strategy[order_info.strategy_id]:
                    self._orders_by_strategy[order_info.strategy_id].remove(order_id)
            
            del self._orders[order_id]
            
            return True
    
    # ========== 事件處理 ==========
    
    def _on_order_event(self, event: OrderEvent) -> None:
        """處理 OrderEvent"""
        order_id = event.order_id
        
        with self._lock:
            # 如果訂單不存在，建立它
            if order_id not in self._orders:
                self.add_order(
                    order_id=order_id,
                    symbol=event.symbol,
                    action=event.action,
                    quantity=event.quantity,
                    order_type=event.order_type,
                    limit_price=event.limit_price,
                    stop_price=event.stop_price,
                    strategy_id=event.strategy_id,
                )
            
            # 映射狀態
            state = self._map_status_to_state(event.status)
            
            # 更新訂單
            self.update_order(
                order_id=order_id,
                state=state,
                status=event.status,
                filled_quantity=event.filled_quantity,
                avg_fill_price=event.avg_fill_price,
            )
    
    def _on_fill_event(self, event: FillEvent) -> None:
        """處理 FillEvent"""
        order_id = event.order_id
        
        with self._lock:
            order_info = self._orders.get(order_id)
            if order_info is None:
                return
            
            # 更新成交資訊
            order_info.last_fill_price = event.price
            order_info.last_fill_quantity = event.quantity
            order_info.commission += event.commission or 0
            
            # 計算新的平均成交價
            total_filled = order_info.filled_quantity + event.quantity
            if total_filled > 0:
                new_avg = (
                    order_info.avg_fill_price * order_info.filled_quantity
                    + event.price * event.quantity
                ) / total_filled
                order_info.avg_fill_price = new_avg
            
            order_info.filled_quantity = total_filled
            order_info.remaining_quantity = order_info.quantity - total_filled
            
            # 更新狀態
            if order_info.remaining_quantity <= 0:
                order_info.state = OrderState.FILLED
                order_info.filled_at = datetime.now()
                self._handle_completed_order(order_info)
            else:
                order_info.state = OrderState.PARTIAL_FILLED
                if order_info.first_fill_at is None:
                    order_info.first_fill_at = datetime.now()
            
            order_info.updated_at = datetime.now()
            
            # 執行成交回調
            for callback in self._on_fill_callbacks:
                try:
                    callback(order_info, event)
                except Exception as e:
                    logger.error(f"成交回調錯誤: {e}")
    
    def _map_status_to_state(self, status: OrderStatus) -> OrderState:
        """映射 OrderStatus 到 OrderState"""
        mapping = {
            OrderStatus.PENDING: OrderState.PENDING_SUBMIT,
            OrderStatus.SUBMITTED: OrderState.SUBMITTED,
            OrderStatus.PARTIAL: OrderState.PARTIAL_FILLED,
            OrderStatus.FILLED: OrderState.FILLED,
            OrderStatus.CANCELLED: OrderState.CANCELLED,
            OrderStatus.REJECTED: OrderState.REJECTED,
        }
        return mapping.get(status, OrderState.SUBMITTED)
    
    # ========== 查詢方法 ==========
    
    def get_order(self, order_id: int) -> Optional[OrderInfo]:
        """取得訂單"""
        return self._orders.get(order_id)
    
    def get_orders(self, order_ids: List[int]) -> List[OrderInfo]:
        """取得多個訂單"""
        return [
            self._orders[oid]
            for oid in order_ids
            if oid in self._orders
        ]
    
    def get_all_orders(self) -> List[OrderInfo]:
        """取得所有訂單"""
        return list(self._orders.values())
    
    def get_pending_orders(self) -> List[OrderInfo]:
        """取得待處理訂單"""
        return [
            order for order in self._orders.values()
            if order.is_active
        ]
    
    def get_filled_orders(self) -> List[OrderInfo]:
        """取得已成交訂單"""
        return [
            order for order in self._orders.values()
            if order.is_filled
        ]
    
    def get_cancelled_orders(self) -> List[OrderInfo]:
        """取得已取消訂單"""
        return [
            order for order in self._orders.values()
            if order.is_cancelled
        ]
    
    def get_orders_by_symbol(self, symbol: str) -> List[OrderInfo]:
        """取得指定標的的訂單"""
        order_ids = self._orders_by_symbol.get(symbol, [])
        return self.get_orders(order_ids)
    
    def get_orders_by_strategy(self, strategy_id: str) -> List[OrderInfo]:
        """取得指定策略的訂單"""
        order_ids = self._orders_by_strategy.get(strategy_id, [])
        return self.get_orders(order_ids)
    
    def get_active_orders_by_symbol(self, symbol: str) -> List[OrderInfo]:
        """取得指定標的的活躍訂單"""
        return [
            order for order in self.get_orders_by_symbol(symbol)
            if order.is_active
        ]
    
    def get_completed_history(self, limit: int = 100) -> List[OrderInfo]:
        """取得已完成訂單歷史"""
        return self._completed_orders[-limit:]
    
    # ========== 清理 ==========
    
    def clear_completed(self) -> int:
        """
        清理已完成的訂單
        
        Returns:
            清理的訂單數量
        """
        with self._lock:
            completed_ids = [
                order_id
                for order_id, order in self._orders.items()
                if order.is_completed
            ]
            
            for order_id in completed_ids:
                self.remove_order(order_id)
            
            count = len(completed_ids)
            if count > 0:
                logger.info(f"清理 {count} 個已完成訂單")
            
            return count
    
    def clear_all(self) -> int:
        """
        清理所有訂單
        
        Returns:
            清理的訂單數量
        """
        with self._lock:
            count = len(self._orders)
            self._orders.clear()
            self._orders_by_symbol.clear()
            self._orders_by_strategy.clear()
            self._completed_orders.clear()
            
            # 重置統計
            self._stats = OrderStats()
            
            logger.info(f"清理所有訂單: {count} 個")
            return count
    
    # ========== 統計 ==========
    
    def get_stats(self) -> OrderStats:
        """取得統計"""
        return self._stats
    
    def get_summary(self) -> Dict[str, Any]:
        """取得摘要"""
        return {
            "total_orders": self._stats.total_orders,
            "active_orders": self._stats.active_orders,
            "filled_orders": self._stats.filled_orders,
            "cancelled_orders": self._stats.cancelled_orders,
            "rejected_orders": self._stats.rejected_orders,
            "total_filled_quantity": self._stats.total_filled_quantity,
            "total_filled_value": self._stats.total_filled_value,
            "total_commission": self._stats.total_commission,
            "avg_fill_time": f"{self._stats.avg_fill_time:.2f}s",
            "pending_orders": [
                order.to_dict()
                for order in self.get_pending_orders()
            ],
        }
    
    # ========== 回調註冊 ==========
    
    def on_state_change(
        self,
        callback: Callable[[OrderInfo, OrderState], None],
    ) -> Callable:
        """註冊狀態變化回調"""
        self._on_state_change_callbacks.append(callback)
        return callback
    
    def on_fill(
        self,
        callback: Callable[[OrderInfo, FillEvent], None],
    ) -> Callable:
        """註冊成交回調"""
        self._on_fill_callbacks.append(callback)
        return callback