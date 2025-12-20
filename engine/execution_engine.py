"""
ExecutionEngine 模組 - 訂單執行引擎

處理信號轉換為訂單，管理訂單生命週期
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, Dict, List, Callable, Any
import threading

from ib_insync import (
    IB,
    Contract,
    Order,
    Trade,
    OrderStatus as IBOrderStatus,
    Execution,
    CommissionReport,
    LimitOrder,
    MarketOrder,
    StopOrder,
    StopLimitOrder,
)

from core.events import (
    Event,
    EventType,
    SignalEvent,
    OrderEvent,
    FillEvent,
    OrderAction,
    OrderType,
    OrderStatus,
)
from core.event_bus import EventBus, get_event_bus
from risk.manager import RiskManager
from core.connection import IBConnection
from core.contracts import ContractFactory, get_contract_factory


# 設定 logger
logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """執行模式"""
    
    LIVE = auto()        # 實盤
    PAPER = auto()       # 模擬
    SIMULATION = auto()  # 本地模擬（不發送到 IB）


@dataclass
class PendingOrder:
    """待處理訂單資訊"""
    
    order_id: int
    trade: Trade
    contract: Contract
    order: Order
    
    # 來源
    signal: Optional[SignalEvent] = None
    strategy_id: str = ""
    
    # 狀態
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    
    # 時間
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    
    # 關聯訂單（停損停利）
    stop_loss_order_id: Optional[int] = None
    take_profit_order_id: Optional[int] = None


class ExecutionEngine:
    """
    訂單執行引擎
    
    處理信號到訂單的轉換，管理訂單執行和狀態追蹤
    
    使用方式:
        engine = ExecutionEngine(connection, event_bus)
        
        # 啟動引擎
        engine.start()
        
        # 手動提交訂單
        order_id = engine.submit_order(
            contract=stock,
            action=OrderAction.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=150.0,
        )
        
        # 取消訂單
        engine.cancel_order(order_id)
    """
    
    def __init__(
        self,
        connection: IBConnection,
        event_bus: Optional[EventBus] = None,
        mode: ExecutionMode = ExecutionMode.PAPER,
        auto_process_signals: bool = True,
        risk_manager: Optional[RiskManager] = None,
    ):
        """
        初始化執行引擎
        
        Args:
            connection: IB 連接
            event_bus: 事件總線
            mode: 執行模式
            auto_process_signals: 是否自動處理 SignalEvent
            risk_manager: 風險管理器（可選，用於信號風控檢查）
        """
        self._connection = connection
        self._risk_manager = risk_manager
        self._ib = connection.ib
        self._event_bus = event_bus or get_event_bus()
        self._mode = mode
        self._auto_process_signals = auto_process_signals
        
        # 合約工廠
        self._contract_factory = get_contract_factory()
        
        # 訂單管理
        self._pending_orders: Dict[int, PendingOrder] = {}
        self._order_history: List[PendingOrder] = []
        
        # 訂單 ID 映射（IB perm_id -> local order_id）
        self._perm_id_map: Dict[int, int] = {}
        
        # 運行狀態
        self._running = False
        
        # 線程安全
        self._lock = threading.RLock()
        
        # 統計
        self._total_orders_submitted = 0
        self._total_orders_filled = 0
        self._total_orders_cancelled = 0
        self._total_orders_rejected = 0
        
        # 回調
        self._on_order_callbacks: List[Callable[[OrderEvent], None]] = []
        self._on_fill_callbacks: List[Callable[[FillEvent], None]] = []
        
        logger.debug(f"ExecutionEngine 初始化完成，模式: {mode.name}")
    
    # ========== 屬性 ==========
    
    @property
    def is_running(self) -> bool:
        """是否運行中"""
        return self._running
    
    @property
    def mode(self) -> ExecutionMode:
        """執行模式"""
        return self._mode
    
    @property
    def pending_order_count(self) -> int:
        """待處理訂單數量"""
        return len(self._pending_orders)
    
    # ========== 控制 ==========
    
    def start(self) -> None:
        """啟動執行引擎"""
        if self._running:
            logger.warning("ExecutionEngine 已在運行中")
            return
        
        # 設置 IB 事件監聽
        self._setup_ib_events()
        
        # 訂閱 SignalEvent
        if self._auto_process_signals:
            self._event_bus.subscribe(EventType.SIGNAL, self._on_signal, priority=5)
        
        self._running = True
        logger.info(f"ExecutionEngine 已啟動，模式: {self._mode.name}")
    
    def stop(self) -> None:
        """停止執行引擎"""
        if not self._running:
            return
        
        # 取消訂閱
        if self._auto_process_signals:
            self._event_bus.unsubscribe(EventType.SIGNAL, self._on_signal)
        
        # 移除 IB 事件監聽
        self._cleanup_ib_events()
        
        self._running = False
        logger.info("ExecutionEngine 已停止")
    
    def _setup_ib_events(self) -> None:
        """設置 IB 事件監聽"""
        self._ib.orderStatusEvent += self._on_order_status
        self._ib.execDetailsEvent += self._on_exec_details
        self._ib.commissionReportEvent += self._on_commission_report
        self._ib.errorEvent += self._on_ib_error
    
    def _cleanup_ib_events(self) -> None:
        """清理 IB 事件監聽"""
        self._ib.orderStatusEvent -= self._on_order_status
        self._ib.execDetailsEvent -= self._on_exec_details
        self._ib.commissionReportEvent -= self._on_commission_report
        self._ib.errorEvent -= self._on_ib_error
    
    # ========== 信號處理 ==========
    
    def _on_signal(self, signal: SignalEvent) -> None:
        """處理 SignalEvent"""
        if self._mode == ExecutionMode.SIMULATION:
            self._simulate_signal(signal)
            return
        
        # 風控檢查
        if self._risk_manager:
            result = self._risk_manager.check_signal(signal)
            
            if not result.passed:
                logger.warning(
                    f"風控拒絕信號: {signal.symbol} {signal.action.value} - {result.reason}"
                )
                return
            
            if result.has_warnings:
                for warning in result.warnings:
                    logger.warning(f"風控警告: {warning}")
        
        try:
            self.process_signal(signal)
        except Exception as e:
            logger.error(f"處理信號失敗: {e}")
    
    def process_signal(self, signal: SignalEvent) -> Optional[int]:
        """
        處理交易信號，轉換為訂單
        
        Args:
            signal: SignalEvent
            
        Returns:
            訂單 ID 或 None
        """
        # 建立合約
        contract = self._contract_factory.stock(signal.symbol)
        
        # 決定訂單類型和價格
        order_type = signal.suggested_order_type
        limit_price = signal.suggested_price
        
        # 決定數量
        quantity = signal.suggested_quantity or 1
        
        # 提交訂單
        order_id = self.submit_order(
            contract=contract,
            action=signal.action,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            strategy_id=signal.strategy_id,
            signal=signal,
        )
        
        # 處理停損停利
        if order_id and (signal.stop_loss or signal.take_profit):
            self._attach_bracket_orders(
                order_id=order_id,
                contract=contract,
                action=signal.action,
                quantity=quantity,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
            )
        
        return order_id
    
    def _simulate_signal(self, signal: SignalEvent) -> None:
        """模擬信號執行（不實際下單）"""
        logger.info(
            f"[SIMULATION] 信號: {signal.symbol} {signal.action.value} "
            f"qty={signal.suggested_quantity} @ {signal.suggested_price}"
        )
        
        # 發布模擬的 FillEvent
        fill = FillEvent(
            event_type=EventType.FILL,
            order_id=0,
            execution_id=f"SIM_{datetime.now().timestamp()}",
            symbol=signal.symbol,
            action=signal.action,
            quantity=signal.suggested_quantity or 1,
            price=signal.suggested_price or 0.0,
            commission=0.0,
            execution_time=datetime.now(),
            strategy_id=signal.strategy_id,
        )
        self._event_bus.publish(fill)
    
    # ========== 訂單提交 ==========
    
    def submit_order(
        self,
        contract: Contract,
        action: OrderAction,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        strategy_id: str = "",
        signal: Optional[SignalEvent] = None,
        **kwargs,
    ) -> Optional[int]:
        """
        提交訂單
        
        Args:
            contract: IB 合約
            action: 買賣方向
            quantity: 數量
            order_type: 訂單類型
            limit_price: 限價
            stop_price: 停損價
            strategy_id: 策略 ID
            signal: 來源信號
            **kwargs: 額外訂單參數
            
        Returns:
            訂單 ID 或 None
        """
        if self._mode == ExecutionMode.SIMULATION:
            logger.info(f"[SIMULATION] 訂單: {contract.symbol} {action.value} {quantity}")
            return None
        
        # 建立訂單物件
        order = self._create_order(
            action=action,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            **kwargs,
        )
        
        try:
            # 提交到 IB
            trade = self._ib.placeOrder(contract, order)
            order_id = trade.order.orderId
            
            # 建立 PendingOrder
            pending = PendingOrder(
                order_id=order_id,
                trade=trade,
                contract=contract,
                order=order,
                signal=signal,
                strategy_id=strategy_id,
                status=OrderStatus.SUBMITTED,
                submitted_at=datetime.now(),
            )
            
            with self._lock:
                self._pending_orders[order_id] = pending
                self._total_orders_submitted += 1
            
            # 發布 OrderEvent
            self._emit_order_event(pending)
            
            logger.info(
                f"提交訂單: {contract.symbol} {action.value} {quantity} "
                f"(order_id={order_id}, type={order_type.value})"
            )
            
            return order_id
            
        except Exception as e:
            logger.error(f"提交訂單失敗: {e}")
            return None
    
    def _create_order(
        self,
        action: OrderAction,
        quantity: int,
        order_type: OrderType,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        **kwargs,
    ) -> Order:
        """建立 IB Order 物件"""
        action_str = action.value
        
        if order_type == OrderType.MARKET:
            order = MarketOrder(action_str, quantity)
            
        elif order_type == OrderType.LIMIT:
            if limit_price is None:
                raise ValueError("限價單需要 limit_price")
            order = LimitOrder(action_str, quantity, limit_price)
            
        elif order_type == OrderType.STOP:
            if stop_price is None:
                raise ValueError("停損單需要 stop_price")
            order = StopOrder(action_str, quantity, stop_price)
            
        elif order_type == OrderType.STOP_LIMIT:
            if stop_price is None or limit_price is None:
                raise ValueError("停損限價單需要 stop_price 和 limit_price")
            order = StopLimitOrder(action_str, quantity, limit_price, stop_price)
            
        else:
            # 預設市價單
            order = MarketOrder(action_str, quantity)
        
        # 設置額外參數
        for key, value in kwargs.items():
            if hasattr(order, key):
                setattr(order, key, value)
        
        return order
    
    def _attach_bracket_orders(
        self,
        order_id: int,
        contract: Contract,
        action: OrderAction,
        quantity: int,
        stop_loss: Optional[float],
        take_profit: Optional[float],
    ) -> None:
        """附加停損停利訂單"""
        pending = self._pending_orders.get(order_id)
        if pending is None:
            return
        
        # 反向動作
        exit_action = OrderAction.SELL if action == OrderAction.BUY else OrderAction.BUY
        
        # 停損單
        if stop_loss:
            sl_order = StopOrder(exit_action.value, quantity, stop_loss)
            sl_order.parentId = order_id
            sl_order.transmit = False if take_profit else True
            
            sl_trade = self._ib.placeOrder(contract, sl_order)
            pending.stop_loss_order_id = sl_trade.order.orderId
            
            with self._lock:
                self._pending_orders[sl_trade.order.orderId] = PendingOrder(
                    order_id=sl_trade.order.orderId,
                    trade=sl_trade,
                    contract=contract,
                    order=sl_order,
                    strategy_id=pending.strategy_id,
                )
            
            logger.debug(f"附加停損單: {stop_loss}")
        
        # 停利單
        if take_profit:
            tp_order = LimitOrder(exit_action.value, quantity, take_profit)
            tp_order.parentId = order_id
            tp_order.transmit = True
            
            tp_trade = self._ib.placeOrder(contract, tp_order)
            pending.take_profit_order_id = tp_trade.order.orderId
            
            with self._lock:
                self._pending_orders[tp_trade.order.orderId] = PendingOrder(
                    order_id=tp_trade.order.orderId,
                    trade=tp_trade,
                    contract=contract,
                    order=tp_order,
                    strategy_id=pending.strategy_id,
                )
            
            logger.debug(f"附加停利單: {take_profit}")
    
    # ========== 訂單取消 ==========
    
    def cancel_order(self, order_id: int) -> bool:
        """
        取消訂單
        
        Args:
            order_id: 訂單 ID
            
        Returns:
            是否成功發送取消請求
        """
        pending = self._pending_orders.get(order_id)
        if pending is None:
            logger.warning(f"訂單 {order_id} 不存在")
            return False
        
        try:
            self._ib.cancelOrder(pending.order)
            logger.info(f"取消訂單: {order_id}")
            return True
        except Exception as e:
            logger.error(f"取消訂單失敗 ({order_id}): {e}")
            return False
    
    def cancel_all_orders(self) -> int:
        """
        取消所有待處理訂單
        
        Returns:
            取消的訂單數量
        """
        count = 0
        
        with self._lock:
            order_ids = list(self._pending_orders.keys())
        
        for order_id in order_ids:
            if self.cancel_order(order_id):
                count += 1
        
        logger.info(f"取消所有訂單: {count} 個")
        return count
    
    def cancel_strategy_orders(self, strategy_id: str) -> int:
        """
        取消指定策略的所有訂單
        
        Args:
            strategy_id: 策略 ID
            
        Returns:
            取消的訂單數量
        """
        count = 0
        
        with self._lock:
            for order_id, pending in list(self._pending_orders.items()):
                if pending.strategy_id == strategy_id:
                    if self.cancel_order(order_id):
                        count += 1
        
        return count
    
    # ========== IB 事件處理 ==========
    
    def _on_order_status(self, trade: Trade) -> None:
        """處理 IB orderStatusEvent"""
        order_id = trade.order.orderId
        pending = self._pending_orders.get(order_id)
        
        if pending is None:
            return
        
        # 更新狀態
        ib_status = trade.orderStatus.status
        new_status = self._map_order_status(ib_status)
        
        pending.status = new_status
        pending.filled_quantity = int(trade.orderStatus.filled)
        pending.avg_fill_price = trade.orderStatus.avgFillPrice
        
        # 發布 OrderEvent
        self._emit_order_event(pending)
        
        # 處理完成的訂單
        if new_status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            self._complete_order(pending, new_status)
    
    def _on_exec_details(self, trade: Trade, fill: Execution) -> None:
        """處理 IB execDetailsEvent"""
        order_id = trade.order.orderId
        pending = self._pending_orders.get(order_id)
        
        # 建立 FillEvent
        fill_event = FillEvent(
            event_type=EventType.FILL,
            order_id=order_id,
            execution_id=fill.execId,
            symbol=trade.contract.symbol,
            action=OrderAction.BUY if fill.side == "BOT" else OrderAction.SELL,
            quantity=int(fill.shares),
            price=fill.price,
            commission=0.0,  # 會在 commissionReportEvent 更新
            execution_time=fill.time,
            exchange=fill.exchange,
            strategy_id=pending.strategy_id if pending else "",
        )
        
        # 發布事件
        self._event_bus.publish(fill_event)
        
        # 執行回調
        for callback in self._on_fill_callbacks:
            try:
                callback(fill_event)
            except Exception as e:
                logger.error(f"Fill 回調錯誤: {e}")
        
        logger.info(
            f"成交: {trade.contract.symbol} {fill.side} {fill.shares} @ {fill.price}"
        )
    
    def _on_commission_report(self, trade: Trade, fill: Execution, report: CommissionReport) -> None:
        """處理 IB commissionReportEvent"""
        logger.debug(
            f"佣金報告: order_id={trade.order.orderId}, "
            f"commission={report.commission}, currency={report.currency}"
        )
    
    def _on_ib_error(
        self,
        reqId: int,
        errorCode: int,
        errorString: str,
        contract: Optional[Contract] = None,
    ) -> None:
        """處理 IB 錯誤"""
        # 訂單相關錯誤
        order_error_codes = {201, 202, 203, 321, 322, 323, 324, 325}
        
        if errorCode in order_error_codes:
            pending = self._pending_orders.get(reqId)
            if pending:
                pending.status = OrderStatus.REJECTED
                self._complete_order(pending, OrderStatus.REJECTED)
                logger.warning(f"訂單錯誤 [{errorCode}]: {errorString}")
    
    # ========== 輔助方法 ==========
    
    def _map_order_status(self, ib_status: str) -> OrderStatus:
        """映射 IB 訂單狀態到內部狀態"""
        mapping = {
            "PendingSubmit": OrderStatus.PENDING,
            "PendingCancel": OrderStatus.PENDING,
            "PreSubmitted": OrderStatus.SUBMITTED,
            "Submitted": OrderStatus.SUBMITTED,
            "ApiPending": OrderStatus.PENDING,
            "ApiCancelled": OrderStatus.CANCELLED,
            "Cancelled": OrderStatus.CANCELLED,
            "Filled": OrderStatus.FILLED,
            "Inactive": OrderStatus.REJECTED,
        }
        return mapping.get(ib_status, OrderStatus.PENDING)
    
    def _emit_order_event(self, pending: PendingOrder) -> None:
        """發布 OrderEvent"""
        order = pending.order
        
        order_event = OrderEvent(
            event_type=EventType.ORDER,
            order_id=pending.order_id,
            symbol=pending.contract.symbol,
            action=OrderAction.BUY if order.action == "BUY" else OrderAction.SELL,
            order_type=self._get_order_type(order),
            quantity=int(order.totalQuantity),
            limit_price=getattr(order, "lmtPrice", None),
            stop_price=getattr(order, "auxPrice", None),
            status=pending.status,
            filled_quantity=pending.filled_quantity,
            remaining_quantity=int(order.totalQuantity) - pending.filled_quantity,
            avg_fill_price=pending.avg_fill_price,
            submitted_time=pending.submitted_at,
            filled_time=pending.filled_at,
            strategy_id=pending.strategy_id,
        )
        
        self._event_bus.publish(order_event)
        
        # 執行回調
        for callback in self._on_order_callbacks:
            try:
                callback(order_event)
            except Exception as e:
                logger.error(f"Order 回調錯誤: {e}")
    
    def _get_order_type(self, order: Order) -> OrderType:
        """取得訂單類型"""
        order_type_str = order.orderType
        mapping = {
            "MKT": OrderType.MARKET,
            "LMT": OrderType.LIMIT,
            "STP": OrderType.STOP,
            "STP LMT": OrderType.STOP_LIMIT,
        }
        return mapping.get(order_type_str, OrderType.MARKET)
    
    def _complete_order(self, pending: PendingOrder, status: OrderStatus) -> None:
        """完成訂單處理"""
        with self._lock:
            if pending.order_id in self._pending_orders:
                del self._pending_orders[pending.order_id]
                self._order_history.append(pending)
            
            if status == OrderStatus.FILLED:
                pending.filled_at = datetime.now()
                self._total_orders_filled += 1
            elif status == OrderStatus.CANCELLED:
                self._total_orders_cancelled += 1
            elif status == OrderStatus.REJECTED:
                self._total_orders_rejected += 1
    
    # ========== 回調註冊 ==========
    
    def on_order(self, callback: Callable[[OrderEvent], None]) -> Callable:
        """註冊訂單回調"""
        self._on_order_callbacks.append(callback)
        return callback
    
    def on_fill(self, callback: Callable[[FillEvent], None]) -> Callable:
        """註冊成交回調"""
        self._on_fill_callbacks.append(callback)
        return callback
    
    # ========== 查詢 ==========
    
    def get_pending_orders(self) -> List[PendingOrder]:
        """取得所有待處理訂單"""
        with self._lock:
            return list(self._pending_orders.values())
    
    def get_order(self, order_id: int) -> Optional[PendingOrder]:
        """取得指定訂單"""
        return self._pending_orders.get(order_id)
    
    def get_order_history(self, limit: int = 100) -> List[PendingOrder]:
        """取得訂單歷史"""
        return self._order_history[-limit:]
    
    # ========== 統計 ==========
    
    def get_stats(self) -> Dict[str, Any]:
        """取得統計資訊"""
        return {
            "running": self._running,
            "mode": self._mode.name,
            "pending_orders": len(self._pending_orders),
            "total_submitted": self._total_orders_submitted,
            "total_filled": self._total_orders_filled,
            "total_cancelled": self._total_orders_cancelled,
            "total_rejected": self._total_orders_rejected,
            "history_count": len(self._order_history),
        }
    
    def reset_stats(self) -> None:
        """重置統計"""
        self._total_orders_submitted = 0
        self._total_orders_filled = 0
        self._total_orders_cancelled = 0
        self._total_orders_rejected = 0