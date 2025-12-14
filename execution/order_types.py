"""
OrderTypes 模組 - 訂單類型工具

提供各種複合訂單的建立工具
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, List
import uuid

from ib_insync import (
    Order,
    LimitOrder,
    MarketOrder,
    StopOrder,
    StopLimitOrder,
    Contract,
)

from core.events import OrderAction


# 設定 logger
logger = logging.getLogger(__name__)


@dataclass
class BracketOrders:
    """括號訂單組"""
    
    parent: Order         # 主訂單（進場）
    take_profit: Order    # 止盈訂單
    stop_loss: Order      # 止損訂單
    
    def to_list(self) -> List[Order]:
        """轉換為列表"""
        return [self.parent, self.take_profit, self.stop_loss]


@dataclass
class OCOOrders:
    """OCO 訂單組"""
    
    order1: Order
    order2: Order
    oca_group: str
    
    def to_list(self) -> List[Order]:
        """轉換為列表"""
        return [self.order1, self.order2]


def create_bracket_order(
    action: str,
    quantity: int,
    entry_price: Optional[float] = None,
    take_profit_price: float = 0.0,
    stop_loss_price: float = 0.0,
    entry_order_type: str = "LMT",
    parent_order_id: int = 0,
    transmit: bool = True,
    tif: str = "GTC",
    oca_group: Optional[str] = None,
) -> BracketOrders:
    """
    建立括號訂單（進場 + 止盈 + 止損）
    
    Args:
        action: 進場方向 ("BUY" 或 "SELL")
        quantity: 數量
        entry_price: 進場價格（限價單時必填）
        take_profit_price: 止盈價格
        stop_loss_price: 止損價格
        entry_order_type: 進場訂單類型 ("MKT" 或 "LMT")
        parent_order_id: 父訂單 ID（0 表示自動分配）
        transmit: 是否立即傳送
        tif: 有效期 ("DAY", "GTC", "IOC", "FOK")
        oca_group: OCA 群組名稱
        
    Returns:
        BracketOrders 包含三個訂單
        
    使用方式:
        bracket = create_bracket_order(
            action="BUY",
            quantity=100,
            entry_price=150.0,
            take_profit_price=160.0,
            stop_loss_price=145.0,
        )
        
        # 提交訂單
        for order in bracket.to_list():
            ib.placeOrder(contract, order)
    """
    # 反向動作（用於止盈止損）
    exit_action = "SELL" if action == "BUY" else "BUY"
    
    # 產生 OCA 群組名稱
    if oca_group is None:
        oca_group = f"bracket_{uuid.uuid4().hex[:8]}"
    
    # 建立進場訂單（父訂單）
    if entry_order_type == "MKT":
        parent = MarketOrder(action, quantity)
    else:
        if entry_price is None:
            raise ValueError("限價單需要 entry_price")
        parent = LimitOrder(action, quantity, entry_price)
    
    parent.orderId = parent_order_id
    parent.transmit = False  # 暫不傳送，等所有訂單都準備好
    parent.tif = tif
    
    # 建立止盈訂單（限價單）
    take_profit = LimitOrder(exit_action, quantity, take_profit_price)
    take_profit.parentId = parent.orderId
    take_profit.transmit = False
    take_profit.tif = tif
    take_profit.ocaGroup = oca_group
    take_profit.ocaType = 1  # Cancel other orders in group
    
    # 建立止損訂單（停損單）
    stop_loss = StopOrder(exit_action, quantity, stop_loss_price)
    stop_loss.parentId = parent.orderId
    stop_loss.transmit = transmit  # 最後一個訂單設為 transmit
    stop_loss.tif = tif
    stop_loss.ocaGroup = oca_group
    stop_loss.ocaType = 1  # Cancel other orders in group
    
    logger.debug(
        f"建立括號訂單: {action} {quantity} @ {entry_price}, "
        f"TP={take_profit_price}, SL={stop_loss_price}"
    )
    
    return BracketOrders(
        parent=parent,
        take_profit=take_profit,
        stop_loss=stop_loss,
    )


def create_oco_order(
    action: str,
    quantity: int,
    limit_price: float,
    stop_price: float,
    oca_group: Optional[str] = None,
    oca_type: int = 1,
    transmit: bool = True,
    tif: str = "GTC",
) -> OCOOrders:
    """
    建立 OCO 訂單（One-Cancels-Other，二擇一）
    
    當其中一個訂單成交時，自動取消另一個訂單
    
    Args:
        action: 交易方向 ("BUY" 或 "SELL")
        quantity: 數量
        limit_price: 限價單價格
        stop_price: 停損單價格
        oca_group: OCA 群組名稱
        oca_type: OCA 類型 (1=取消其他, 2=減少其他, 3=減少並取消)
        transmit: 是否立即傳送
        tif: 有效期
        
    Returns:
        OCOOrders 包含兩個關聯訂單
        
    使用方式:
        # 建立 OCO 訂單：限價買入或停損買入
        oco = create_oco_order(
            action="BUY",
            quantity=100,
            limit_price=145.0,  # 限價買入
            stop_price=155.0,   # 突破買入
        )
        
        # 提交訂單
        for order in oco.to_list():
            ib.placeOrder(contract, order)
    """
    # 產生 OCA 群組名稱
    if oca_group is None:
        oca_group = f"oco_{uuid.uuid4().hex[:8]}"
    
    # 建立限價單
    limit_order = LimitOrder(action, quantity, limit_price)
    limit_order.transmit = False
    limit_order.tif = tif
    limit_order.ocaGroup = oca_group
    limit_order.ocaType = oca_type
    
    # 建立停損單
    stop_order = StopOrder(action, quantity, stop_price)
    stop_order.transmit = transmit  # 最後一個設為 transmit
    stop_order.tif = tif
    stop_order.ocaGroup = oca_group
    stop_order.ocaType = oca_type
    
    logger.debug(
        f"建立 OCO 訂單: {action} {quantity}, "
        f"LMT={limit_price}, STP={stop_price}, group={oca_group}"
    )
    
    return OCOOrders(
        order1=limit_order,
        order2=stop_order,
        oca_group=oca_group,
    )


def create_trailing_stop(
    action: str,
    quantity: int,
    trailing_amount: Optional[float] = None,
    trailing_percent: Optional[float] = None,
    stop_price: Optional[float] = None,
    tif: str = "GTC",
    transmit: bool = True,
) -> Order:
    """
    建立追蹤停損單
    
    Args:
        action: 交易方向 ("BUY" 或 "SELL")
        quantity: 數量
        trailing_amount: 追蹤金額（與 trailing_percent 二擇一）
        trailing_percent: 追蹤百分比（與 trailing_amount 二擇一）
        stop_price: 初始停損價格（可選）
        tif: 有效期
        transmit: 是否立即傳送
        
    Returns:
        Order 追蹤停損訂單
        
    使用方式:
        # 追蹤金額：價格下跌 $2 時觸發
        order = create_trailing_stop(
            action="SELL",
            quantity=100,
            trailing_amount=2.0,
        )
        
        # 追蹤百分比：價格下跌 5% 時觸發
        order = create_trailing_stop(
            action="SELL",
            quantity=100,
            trailing_percent=5.0,
        )
    """
    order = Order()
    order.action = action
    order.totalQuantity = quantity
    order.orderType = "TRAIL"
    order.tif = tif
    order.transmit = transmit
    
    if trailing_amount is not None:
        order.auxPrice = trailing_amount  # 追蹤金額
    elif trailing_percent is not None:
        order.trailingPercent = trailing_percent  # 追蹤百分比
    else:
        raise ValueError("需要 trailing_amount 或 trailing_percent")
    
    if stop_price is not None:
        order.trailStopPrice = stop_price
    
    logger.debug(
        f"建立追蹤停損單: {action} {quantity}, "
        f"amount={trailing_amount}, pct={trailing_percent}"
    )
    
    return order


def create_adaptive_order(
    action: str,
    quantity: int,
    limit_price: float,
    priority: str = "Normal",
    tif: str = "DAY",
    transmit: bool = True,
) -> Order:
    """
    建立自適應訂單
    
    IB 的自適應演算法會根據市場狀況自動調整訂單
    
    Args:
        action: 交易方向 ("BUY" 或 "SELL")
        quantity: 數量
        limit_price: 限價
        priority: 優先級 ("Urgent", "Normal", "Patient")
        tif: 有效期
        transmit: 是否立即傳送
        
    Returns:
        Order 自適應訂單
    """
    order = LimitOrder(action, quantity, limit_price)
    order.tif = tif
    order.transmit = transmit
    
    # 設定自適應演算法
    order.algoStrategy = "Adaptive"
    order.algoParams = [
        ("adaptivePriority", priority),
    ]
    
    logger.debug(
        f"建立自適應訂單: {action} {quantity} @ {limit_price}, priority={priority}"
    )
    
    return order


def create_twap_order(
    action: str,
    quantity: int,
    start_time: str = "",
    end_time: str = "",
    allow_past_end_time: bool = True,
    strategy_type: str = "Midpoint",
    transmit: bool = True,
) -> Order:
    """
    建立 TWAP 訂單（時間加權平均價格）
    
    在指定時間範圍內平均分配訂單執行
    
    Args:
        action: 交易方向 ("BUY" 或 "SELL")
        quantity: 數量
        start_time: 開始時間 (格式: "HH:MM:SS TMZ")
        end_time: 結束時間 (格式: "HH:MM:SS TMZ")
        allow_past_end_time: 是否允許超過結束時間
        strategy_type: 策略類型 ("Midpoint", "Matchpoint")
        transmit: 是否立即傳送
        
    Returns:
        Order TWAP 訂單
        
    使用方式:
        order = create_twap_order(
            action="BUY",
            quantity=1000,
            start_time="09:30:00 US/Eastern",
            end_time="16:00:00 US/Eastern",
        )
    """
    order = Order()
    order.action = action
    order.totalQuantity = quantity
    order.orderType = "MKT"  # TWAP 使用市價單
    order.transmit = transmit
    
    # 設定 TWAP 演算法
    order.algoStrategy = "Twap"
    order.algoParams = []
    
    if start_time:
        order.algoParams.append(("startTime", start_time))
    if end_time:
        order.algoParams.append(("endTime", end_time))
    
    order.algoParams.append(("allowPastEndTime", "1" if allow_past_end_time else "0"))
    order.algoParams.append(("strategyType", strategy_type))
    
    logger.debug(
        f"建立 TWAP 訂單: {action} {quantity}, "
        f"start={start_time}, end={end_time}"
    )
    
    return order


def create_vwap_order(
    action: str,
    quantity: int,
    start_time: str = "",
    end_time: str = "",
    max_pct_volume: float = 0.1,
    no_take_liq: bool = False,
    transmit: bool = True,
) -> Order:
    """
    建立 VWAP 訂單（成交量加權平均價格）
    
    根據市場成交量分配訂單執行
    
    Args:
        action: 交易方向 ("BUY" 或 "SELL")
        quantity: 數量
        start_time: 開始時間
        end_time: 結束時間
        max_pct_volume: 最大成交量佔比 (0.1 = 10%)
        no_take_liq: 是否禁止吃單
        transmit: 是否立即傳送
        
    Returns:
        Order VWAP 訂單
    """
    order = Order()
    order.action = action
    order.totalQuantity = quantity
    order.orderType = "MKT"
    order.transmit = transmit
    
    # 設定 VWAP 演算法
    order.algoStrategy = "Vwap"
    order.algoParams = []
    
    if start_time:
        order.algoParams.append(("startTime", start_time))
    if end_time:
        order.algoParams.append(("endTime", end_time))
    
    order.algoParams.append(("maxPctVol", str(max_pct_volume)))
    order.algoParams.append(("noTakeLiq", "1" if no_take_liq else "0"))
    
    logger.debug(
        f"建立 VWAP 訂單: {action} {quantity}, "
        f"maxPctVol={max_pct_volume}"
    )
    
    return order


def create_iceberg_order(
    action: str,
    quantity: int,
    limit_price: float,
    display_size: int,
    tif: str = "DAY",
    transmit: bool = True,
) -> Order:
    """
    建立冰山訂單
    
    只顯示部分訂單量，隱藏真實訂單大小
    
    Args:
        action: 交易方向 ("BUY" 或 "SELL")
        quantity: 總數量
        limit_price: 限價
        display_size: 顯示數量
        tif: 有效期
        transmit: 是否立即傳送
        
    Returns:
        Order 冰山訂單
    """
    order = LimitOrder(action, quantity, limit_price)
    order.displaySize = display_size
    order.tif = tif
    order.transmit = transmit
    
    logger.debug(
        f"建立冰山訂單: {action} {quantity} @ {limit_price}, "
        f"display={display_size}"
    )
    
    return order


def create_peg_to_midpoint_order(
    action: str,
    quantity: int,
    offset: float = 0.0,
    tif: str = "DAY",
    transmit: bool = True,
) -> Order:
    """
    建立中間價掛單
    
    訂單價格自動追蹤買賣價中間值
    
    Args:
        action: 交易方向 ("BUY" 或 "SELL")
        quantity: 數量
        offset: 偏移量（相對於中間價）
        tif: 有效期
        transmit: 是否立即傳送
        
    Returns:
        Order 中間價掛單
    """
    order = Order()
    order.action = action
    order.totalQuantity = quantity
    order.orderType = "PEG MID"
    order.auxPrice = offset
    order.tif = tif
    order.transmit = transmit
    
    logger.debug(
        f"建立中間價掛單: {action} {quantity}, offset={offset}"
    )
    
    return order


def create_stop_limit_order(
    action: str,
    quantity: int,
    stop_price: float,
    limit_price: float,
    tif: str = "GTC",
    transmit: bool = True,
) -> Order:
    """
    建立停損限價單
    
    當價格觸及停損價時，轉為限價單
    
    Args:
        action: 交易方向 ("BUY" 或 "SELL")
        quantity: 數量
        stop_price: 停損觸發價
        limit_price: 限價
        tif: 有效期
        transmit: 是否立即傳送
        
    Returns:
        Order 停損限價單
    """
    order = StopLimitOrder(action, quantity, limit_price, stop_price)
    order.tif = tif
    order.transmit = transmit
    
    logger.debug(
        f"建立停損限價單: {action} {quantity}, "
        f"stop={stop_price}, limit={limit_price}"
    )
    
    return order


def create_conditional_order(
    action: str,
    quantity: int,
    order_type: str,
    price: Optional[float] = None,
    conditions: Optional[List] = None,
    conditions_logic: str = "AND",
    transmit: bool = True,
) -> Order:
    """
    建立條件訂單
    
    當滿足指定條件時才執行訂單
    
    Args:
        action: 交易方向 ("BUY" 或 "SELL")
        quantity: 數量
        order_type: 訂單類型 ("MKT", "LMT", "STP")
        price: 價格（限價單或停損單需要）
        conditions: 條件列表
        conditions_logic: 條件邏輯 ("AND" 或 "OR")
        transmit: 是否立即傳送
        
    Returns:
        Order 條件訂單
        
    注意:
        conditions 應該是 ib_insync 的 OrderCondition 物件列表
        例如：PriceCondition, TimeCondition, VolumeCondition 等
    """
    if order_type == "MKT":
        order = MarketOrder(action, quantity)
    elif order_type == "LMT":
        if price is None:
            raise ValueError("限價單需要 price")
        order = LimitOrder(action, quantity, price)
    elif order_type == "STP":
        if price is None:
            raise ValueError("停損單需要 price")
        order = StopOrder(action, quantity, price)
    else:
        order = MarketOrder(action, quantity)
    
    order.transmit = transmit
    
    # 設定條件
    if conditions:
        order.conditions = conditions
        # 設定條件邏輯：True = AND, False = OR
        order.conditionsIgnoreRth = True
        order.conditionsCancelOrder = False
        
        # 注意：實際的 conditionsLogic 需要在 conditions 中設定
        # 每個 condition 有 connector 屬性來設定 AND/OR
    
    logger.debug(
        f"建立條件訂單: {action} {quantity} {order_type}, "
        f"conditions={len(conditions) if conditions else 0}"
    )
    
    return order


# ============================================================
# 便捷函數
# ============================================================

def market_order(action: str, quantity: int, transmit: bool = True) -> Order:
    """建立市價單"""
    order = MarketOrder(action, quantity)
    order.transmit = transmit
    return order


def limit_order(
    action: str,
    quantity: int,
    price: float,
    tif: str = "DAY",
    transmit: bool = True,
) -> Order:
    """建立限價單"""
    order = LimitOrder(action, quantity, price)
    order.tif = tif
    order.transmit = transmit
    return order


def stop_order(
    action: str,
    quantity: int,
    price: float,
    tif: str = "GTC",
    transmit: bool = True,
) -> Order:
    """建立停損單"""
    order = StopOrder(action, quantity, price)
    order.tif = tif
    order.transmit = transmit
    return order