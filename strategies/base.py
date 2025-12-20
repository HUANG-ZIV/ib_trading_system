"""
Base 模組 - 策略基類

提供所有交易策略的基礎抽象類
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, Dict, List, Set, Any
import uuid

from core.events import (
    Event,
    EventType,
    TickEvent,
    BarEvent,
    SignalEvent,
    FillEvent,
    PositionEvent,
    OrderAction,
    OrderType,
)
from core.event_bus import EventBus, get_event_bus


# 設定 logger
logger = logging.getLogger(__name__)


class StrategyState(Enum):
    """策略狀態枚舉"""
    
    IDLE = auto()       # 閒置（未啟動）
    RUNNING = auto()    # 運行中
    PAUSED = auto()     # 暫停中
    STOPPED = auto()    # 已停止
    ERROR = auto()      # 錯誤狀態


@dataclass
class StrategyConfig:
    """
    策略配置
    
    可由子類繼承擴展
    """
    
    # 基本設定
    name: str = ""
    symbols: List[str] = field(default_factory=list)
    
    # 交易設定
    max_position_size: int = 100
    default_quantity: int = 1
    
    # 風控設定
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    max_daily_trades: int = 100
    
    # 時間設定
    trading_start_time: Optional[str] = None  # "09:30"
    trading_end_time: Optional[str] = None    # "16:00"
    
    # 額外參數
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyStats:
    """策略統計資訊"""
    
    # 事件計數
    tick_count: int = 0
    bar_count: int = 0
    signal_count: int = 0
    
    # 交易統計
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # 盈虧統計
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    max_drawdown: float = 0.0
    
    # 時間戳
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    last_signal_time: Optional[datetime] = None
    last_trade_time: Optional[datetime] = None
    
    @property
    def win_rate(self) -> float:
        """勝率"""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades
    
    @property
    def loss_rate(self) -> float:
        """敗率"""
        if self.total_trades == 0:
            return 0.0
        return self.losing_trades / self.total_trades
    
    def reset(self) -> None:
        """重置統計"""
        self.tick_count = 0
        self.bar_count = 0
        self.signal_count = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.max_drawdown = 0.0


@dataclass
class Position:
    """持倉資訊"""
    
    symbol: str
    quantity: int = 0           # 正數多頭，負數空頭
    avg_cost: float = 0.0       # 平均成本
    market_price: float = 0.0   # 市場價格
    
    @property
    def is_long(self) -> bool:
        """是否多頭"""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """是否空頭"""
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        """是否無持倉"""
        return self.quantity == 0
    
    @property
    def market_value(self) -> float:
        """市值"""
        return abs(self.quantity) * self.market_price
    
    @property
    def unrealized_pnl(self) -> float:
        """未實現盈虧"""
        if self.quantity == 0:
            return 0.0
        return self.quantity * (self.market_price - self.avg_cost)
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """未實現盈虧百分比"""
        if self.avg_cost == 0:
            return 0.0
        return (self.market_price - self.avg_cost) / self.avg_cost * 100
    
    def update_price(self, price: float) -> None:
        """更新市場價格"""
        self.market_price = price
    
    def update_from_fill(self, quantity: int, price: float) -> float:
        """
        根據成交更新持倉
        
        Args:
            quantity: 成交數量（正數買入，負數賣出）
            price: 成交價格
            
        Returns:
            已實現盈虧
        """
        realized_pnl = 0.0
        
        if self.quantity == 0:
            # 新建倉位
            self.quantity = quantity
            self.avg_cost = price
            
        elif (self.quantity > 0 and quantity > 0) or (self.quantity < 0 and quantity < 0):
            # 加倉
            total_cost = self.avg_cost * abs(self.quantity) + price * abs(quantity)
            self.quantity += quantity
            self.avg_cost = total_cost / abs(self.quantity)
            
        else:
            # 減倉或反向
            close_qty = min(abs(self.quantity), abs(quantity))
            realized_pnl = close_qty * (price - self.avg_cost) * (1 if self.quantity > 0 else -1)
            
            self.quantity += quantity
            
            if self.quantity != 0 and (self.quantity > 0) != (self.quantity - quantity > 0):
                # 反向了，重新計算成本
                self.avg_cost = price
        
        self.market_price = price
        return realized_pnl


class BaseStrategy(ABC):
    """
    策略基類
    
    所有交易策略必須繼承此類並實現 on_bar() 方法
    
    使用方式:
        class MyStrategy(BaseStrategy):
            def __init__(self):
                super().__init__(
                    strategy_id="my_strategy",
                    symbols=["AAPL", "MSFT"],
                )
            
            def on_bar(self, event: BarEvent) -> Optional[SignalEvent]:
                # 策略邏輯
                if some_condition:
                    return self.emit_signal(
                        symbol=event.symbol,
                        action=OrderAction.BUY,
                        quantity=100,
                    )
                return None
    """
    
    def __init__(
        self,
        strategy_id: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        config: Optional[StrategyConfig] = None,
        event_bus: Optional[EventBus] = None,
    ):
        """
        初始化策略
        
        Args:
            strategy_id: 策略唯一 ID，None 則自動生成
            symbols: 策略訂閱的標的列表
            config: 策略配置
            event_bus: 事件總線
        """
        # 基本屬性
        self._strategy_id = strategy_id or f"strategy_{uuid.uuid4().hex[:8]}"
        self._symbols: Set[str] = set(symbols or [])
        self._config = config or StrategyConfig()
        self._event_bus = event_bus or get_event_bus()
        
        # 從 config 補充 symbols
        if self._config.symbols:
            self._symbols.update(self._config.symbols)
        
        # 狀態
        self._state = StrategyState.IDLE
        
        # 持倉管理: {symbol: Position}
        self._positions: Dict[str, Position] = {}
        
        # 統計
        self._stats = StrategyStats()
        
        # 數據緩存（子類可使用）
        self._data_cache: Dict[str, Any] = {}
        
        # Logger
        self._logger = logging.getLogger(f"strategy.{self._strategy_id}")
        
        self._logger.debug(f"策略 {self._strategy_id} 初始化，symbols={list(self._symbols)}")
    
    # ========== 屬性 ==========
    
    @property
    def strategy_id(self) -> str:
        """策略 ID"""
        return self._strategy_id
    
    @property
    def symbols(self) -> Set[str]:
        """訂閱的標的集合"""
        return self._symbols
    
    @property
    def state(self) -> StrategyState:
        """策略狀態"""
        return self._state
    
    @property
    def config(self) -> StrategyConfig:
        """策略配置"""
        return self._config
    
    @property
    def stats(self) -> StrategyStats:
        """策略統計"""
        return self._stats
    
    @property
    def positions(self) -> Dict[str, Position]:
        """持倉字典"""
        return self._positions
    
    @property
    def is_running(self) -> bool:
        """是否運行中"""
        return self._state == StrategyState.RUNNING
    
    @property
    def event_bus(self) -> EventBus:
        """事件總線"""
        return self._event_bus
    
    # ========== 生命週期方法 ==========
    
    def initialize(self) -> None:
        """
        初始化策略
        
        在策略添加到引擎時調用，子類可覆寫
        """
        self._logger.info(f"策略 {self._strategy_id} 初始化")
    
    def on_start(self) -> None:
        """
        策略啟動時調用
        
        子類可覆寫以執行啟動邏輯
        """
        self._state = StrategyState.RUNNING
        self._stats.started_at = datetime.now()
        self._logger.info(f"策略 {self._strategy_id} 啟動")
    
    def on_stop(self) -> None:
        """
        策略停止時調用
        
        子類可覆寫以執行停止邏輯
        """
        self._state = StrategyState.STOPPED
        self._stats.stopped_at = datetime.now()
        self._logger.info(f"策略 {self._strategy_id} 停止")
    
    def start(self) -> None:
        """啟動策略"""
        if self._state == StrategyState.RUNNING:
            return
        self.on_start()
    
    def stop(self) -> None:
        """停止策略"""
        if self._state == StrategyState.STOPPED:
            return
        self.on_stop()
    
    def pause(self) -> None:
        """暫停策略"""
        if self._state == StrategyState.RUNNING:
            self._state = StrategyState.PAUSED
            self._logger.info(f"策略 {self._strategy_id} 暫停")
    
    def resume(self) -> None:
        """恢復策略"""
        if self._state == StrategyState.PAUSED:
            self._state = StrategyState.RUNNING
            self._logger.info(f"策略 {self._strategy_id} 恢復")
    
    def cleanup(self) -> None:
        """
        清理策略資源
        
        在策略從引擎移除時調用，子類可覆寫
        """
        self._data_cache.clear()
        self._logger.info(f"策略 {self._strategy_id} 清理完成")
    
    # ========== 事件處理方法 ==========
    
    @abstractmethod
    def on_bar(self, event: BarEvent) -> Optional[SignalEvent]:
        """
        處理 Bar 事件（必須實現）
        
        Args:
            event: BarEvent
            
        Returns:
            SignalEvent 或 None
        """
        pass
    
    def on_tick(self, event: TickEvent) -> Optional[SignalEvent]:
        """
        處理 Tick 事件（可選實現）
        
        Args:
            event: TickEvent
            
        Returns:
            SignalEvent 或 None
        """
        self._stats.tick_count += 1
        
        # 更新持倉價格
        symbol = event.symbol
        if symbol in self._positions:
            price = event.last or event.mid
            if price:
                self._positions[symbol].update_price(price)
        
        return None
    
    def on_fill(self, event: FillEvent) -> None:
        """
        處理成交事件
        
        Args:
            event: FillEvent
        """
        symbol = event.symbol
        
        # 計算數量（買入為正，賣出為負）
        quantity = event.quantity
        if event.action == OrderAction.SELL:
            quantity = -quantity
        
        # 更新持倉
        if symbol not in self._positions:
            self._positions[symbol] = Position(symbol=symbol)
        
        realized_pnl = self._positions[symbol].update_from_fill(quantity, event.price)
        
        # 更新統計
        self._stats.total_trades += 1
        self._stats.realized_pnl += realized_pnl
        self._stats.last_trade_time = datetime.now()
        
        if realized_pnl > 0:
            self._stats.winning_trades += 1
        elif realized_pnl < 0:
            self._stats.losing_trades += 1
        
        self._logger.debug(
            f"成交: {symbol} {event.action.value} {event.quantity} @ {event.price}, "
            f"realized_pnl={realized_pnl:.2f}"
        )
    
    def on_position(self, event: PositionEvent) -> None:
        """
        處理持倉更新事件
        
        Args:
            event: PositionEvent
        """
        symbol = event.symbol
        
        if symbol not in self._positions:
            self._positions[symbol] = Position(symbol=symbol)
        
        pos = self._positions[symbol]
        pos.quantity = event.quantity
        pos.avg_cost = event.avg_cost
        if event.market_price:
            pos.market_price = event.market_price
    
    # ========== 信號發送 ==========
    
    def emit_signal(
        self,
        symbol: str,
        action: OrderAction,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
        order_type: OrderType = OrderType.MARKET,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        strength: float = 1.0,
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SignalEvent:
        """
        發送交易信號
        
        Args:
            symbol: 標的代碼
            action: 買賣方向
            quantity: 數量，None 使用預設值
            price: 建議價格
            order_type: 訂單類型
            stop_loss: 停損價
            take_profit: 停利價
            strength: 信號強度 (0.0-1.0)
            reason: 信號原因說明
            metadata: 額外數據
            
        Returns:
            SignalEvent
        """
        # 使用預設數量
        if quantity is None:
            quantity = self._config.default_quantity
        
        signal = SignalEvent(
            event_type=EventType.SIGNAL,
            symbol=symbol,
            action=action,
            strength=strength,
            suggested_quantity=quantity,
            suggested_price=price,
            suggested_order_type=order_type,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_id=self._strategy_id,
            reason=reason,
            metadata=metadata or {},
        )
        
        # 更新統計
        self._stats.signal_count += 1
        self._stats.last_signal_time = datetime.now()
        
        self._logger.info(
            f"發送信號: {symbol} {action.value} qty={quantity} "
            f"strength={strength:.2f} reason={reason}"
        )
        
        return signal
    
    def buy(
        self,
        symbol: str,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        reason: str = "",
    ) -> SignalEvent:
        """買入信號的便捷方法"""
        return self.emit_signal(
            symbol=symbol,
            action=OrderAction.BUY,
            quantity=quantity,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason,
        )
    
    def sell(
        self,
        symbol: str,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        reason: str = "",
    ) -> SignalEvent:
        """賣出信號的便捷方法"""
        return self.emit_signal(
            symbol=symbol,
            action=OrderAction.SELL,
            quantity=quantity,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason,
        )
    
    # ========== 持倉查詢 ==========
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """取得指定標的持倉"""
        return self._positions.get(symbol)
    
    def get_position_quantity(self, symbol: str) -> int:
        """取得指定標的持倉數量"""
        pos = self._positions.get(symbol)
        return pos.quantity if pos else 0
    
    def has_position(self, symbol: str) -> bool:
        """是否有持倉"""
        pos = self._positions.get(symbol)
        return pos is not None and not pos.is_flat
    
    def is_long(self, symbol: str) -> bool:
        """是否多頭持倉"""
        pos = self._positions.get(symbol)
        return pos is not None and pos.is_long
    
    def is_short(self, symbol: str) -> bool:
        """是否空頭持倉"""
        pos = self._positions.get(symbol)
        return pos is not None and pos.is_short
    
    def is_flat(self, symbol: str) -> bool:
        """是否無持倉"""
        pos = self._positions.get(symbol)
        return pos is None or pos.is_flat
    
    def get_total_unrealized_pnl(self) -> float:
        """取得總未實現盈虧"""
        return sum(pos.unrealized_pnl for pos in self._positions.values())
    
    # ========== 持倉恢復 ==========
    
    def restore_position(
        self,
        symbol: str,
        quantity: int,
        avg_cost: float = 0.0,
        strategy_id: str = "",
    ) -> bool:
        """
        恢復持倉狀態（用於系統重啟後）
        
        Args:
            symbol: 標的代碼
            quantity: 持倉數量（正數多頭，負數空頭）
            avg_cost: 平均成本
            strategy_id: 策略 ID（用於驗證）
            
        Returns:
            是否成功恢復
        """
        # 驗證是否屬於此策略
        if strategy_id and strategy_id != self._strategy_id:
            self._logger.debug(
                f"持倉 {symbol} 屬於策略 {strategy_id}，非本策略 {self._strategy_id}"
            )
            return False
        
        # 驗證是否在本策略的交易標的中
        if symbol not in self._symbols:
            self._logger.debug(
                f"持倉 {symbol} 不在本策略的交易標的中"
            )
            return False
        
        # 恢復持倉
        if symbol not in self._positions:
            self._positions[symbol] = Position(symbol=symbol)
        
        pos = self._positions[symbol]
        pos.quantity = quantity
        pos.avg_cost = avg_cost
        
        self._logger.info(
            f"已恢復持倉: {symbol} = {quantity} @ ${avg_cost:.2f}"
        )
        
        return True
    
    def get_strategy_id(self) -> str:
        """取得策略 ID"""
        return self._strategy_id
    
    # ========== 工具方法 ==========
    
    def log(self, message: str, level: str = "info") -> None:
        """記錄日誌"""
        log_func = getattr(self._logger, level, self._logger.info)
        log_func(message)
    
    def get_param(self, key: str, default: Any = None) -> Any:
        """取得配置參數"""
        return self._config.params.get(key, default)
    
    def set_param(self, key: str, value: Any) -> None:
        """設置配置參數"""
        self._config.params[key] = value
    
    def cache_data(self, key: str, value: Any) -> None:
        """緩存數據"""
        self._data_cache[key] = value
    
    def get_cached_data(self, key: str, default: Any = None) -> Any:
        """取得緩存數據"""
        return self._data_cache.get(key, default)
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """取得統計摘要"""
        return {
            "strategy_id": self._strategy_id,
            "state": self._state.name,
            "symbols": list(self._symbols),
            "tick_count": self._stats.tick_count,
            "bar_count": self._stats.bar_count,
            "signal_count": self._stats.signal_count,
            "total_trades": self._stats.total_trades,
            "win_rate": f"{self._stats.win_rate:.2%}",
            "realized_pnl": self._stats.realized_pnl,
            "unrealized_pnl": self.get_total_unrealized_pnl(),
            "positions": {
                symbol: {
                    "quantity": pos.quantity,
                    "avg_cost": pos.avg_cost,
                    "market_price": pos.market_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                }
                for symbol, pos in self._positions.items()
                if not pos.is_flat
            },
        }