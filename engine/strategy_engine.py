"""
StrategyEngine 模組 - 策略執行引擎

管理和執行多個交易策略，處理事件分發
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, Dict, List, Set, Type, Any, TYPE_CHECKING
import threading
import traceback

from core.events import (
    Event,
    EventType,
    TickEvent,
    BarEvent,
    SignalEvent,
    FillEvent,
    PositionEvent,
    SystemEvent,
    SystemEventType,
)
from core.event_bus import EventBus, get_event_bus

if TYPE_CHECKING:
    from strategies.base import BaseStrategy


# 設定 logger
logger = logging.getLogger(__name__)


class StrategyState(Enum):
    """策略狀態枚舉"""
    
    CREATED = auto()      # 已建立
    INITIALIZED = auto()  # 已初始化
    RUNNING = auto()      # 運行中
    PAUSED = auto()       # 暫停中
    STOPPED = auto()      # 已停止
    ERROR = auto()        # 錯誤狀態


@dataclass
class StrategyInfo:
    """策略資訊"""
    
    strategy: "BaseStrategy"
    strategy_id: str
    symbols: Set[str]
    state: StrategyState = StrategyState.CREATED
    
    # 統計
    tick_count: int = 0
    bar_count: int = 0
    signal_count: int = 0
    error_count: int = 0
    
    # 時間戳
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    last_event_time: Optional[datetime] = None
    
    # 錯誤追蹤
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None


class StrategyEngine:
    """
    策略執行引擎
    
    管理多個策略實例，處理事件分發和生命週期管理
    
    使用方式:
        engine = StrategyEngine(event_bus)
        
        # 添加策略
        engine.add_strategy(my_strategy, ["AAPL", "MSFT"])
        
        # 啟動引擎
        engine.start()
        
        # 停止引擎
        engine.stop()
    """
    
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        max_errors_before_pause: int = 5,
        error_cooldown_seconds: int = 60,
    ):
        """
        初始化策略引擎
        
        Args:
            event_bus: 事件總線
            max_errors_before_pause: 策略錯誤多少次後自動暫停
            error_cooldown_seconds: 錯誤冷卻時間（秒）
        """
        self._event_bus = event_bus or get_event_bus()
        self._max_errors_before_pause = max_errors_before_pause
        self._error_cooldown_seconds = error_cooldown_seconds
        
        # 策略管理: {strategy_id: StrategyInfo}
        self._strategies: Dict[str, StrategyInfo] = {}
        
        # Symbol 到策略的映射: {symbol: [strategy_id, ...]}
        self._symbol_strategies: Dict[str, List[str]] = {}
        
        # 運行狀態
        self._running = False
        self._paused = False
        
        # 線程安全
        self._lock = threading.RLock()
        
        # 統計
        self._total_events_processed = 0
        self._total_signals_generated = 0
        
        logger.debug("StrategyEngine 初始化完成")
    
    # ========== 屬性 ==========
    
    @property
    def is_running(self) -> bool:
        """是否運行中"""
        return self._running
    
    @property
    def is_paused(self) -> bool:
        """是否暫停中"""
        return self._paused
    
    @property
    def strategy_count(self) -> int:
        """策略數量"""
        return len(self._strategies)
    
    @property
    def active_strategy_count(self) -> int:
        """運行中的策略數量"""
        return sum(
            1 for info in self._strategies.values()
            if info.state == StrategyState.RUNNING
        )
    
    # ========== 策略管理 ==========
    
    def add_strategy(
        self,
        strategy: "BaseStrategy",
        symbols: Optional[List[str]] = None,
        auto_start: bool = False,
    ) -> str:
        """
        添加策略
        
        Args:
            strategy: 策略實例
            symbols: 策略訂閱的標的列表
            auto_start: 是否自動啟動策略
            
        Returns:
            策略 ID
        """
        strategy_id = strategy.strategy_id
        
        with self._lock:
            if strategy_id in self._strategies:
                logger.warning(f"策略 {strategy_id} 已存在，跳過")
                return strategy_id
            
            # 取得策略訂閱的標的
            if symbols is None:
                symbols = list(strategy.symbols) if hasattr(strategy, 'symbols') else []
            
            # 建立策略資訊
            info = StrategyInfo(
                strategy=strategy,
                strategy_id=strategy_id,
                symbols=set(symbols),
                state=StrategyState.CREATED,
            )
            
            self._strategies[strategy_id] = info
            
            # 建立 symbol 映射
            for symbol in symbols:
                if symbol not in self._symbol_strategies:
                    self._symbol_strategies[symbol] = []
                self._symbol_strategies[symbol].append(strategy_id)
            
            # 初始化策略
            try:
                if hasattr(strategy, 'initialize'):
                    strategy.initialize()
                info.state = StrategyState.INITIALIZED
                logger.info(f"添加策略: {strategy_id}, symbols={symbols}")
            except Exception as e:
                info.state = StrategyState.ERROR
                info.last_error = str(e)
                info.last_error_time = datetime.now()
                logger.error(f"策略 {strategy_id} 初始化失敗: {e}")
            
            # 自動啟動
            if auto_start and info.state == StrategyState.INITIALIZED:
                self._start_strategy(strategy_id)
            
            return strategy_id
    
    def remove_strategy(self, strategy_id: str) -> bool:
        """
        移除策略
        
        Args:
            strategy_id: 策略 ID
            
        Returns:
            是否成功移除
        """
        with self._lock:
            info = self._strategies.get(strategy_id)
            if info is None:
                logger.warning(f"策略 {strategy_id} 不存在")
                return False
            
            # 停止策略
            if info.state == StrategyState.RUNNING:
                self._stop_strategy(strategy_id)
            
            # 移除 symbol 映射
            for symbol in info.symbols:
                if symbol in self._symbol_strategies:
                    if strategy_id in self._symbol_strategies[symbol]:
                        self._symbol_strategies[symbol].remove(strategy_id)
                    if not self._symbol_strategies[symbol]:
                        del self._symbol_strategies[symbol]
            
            # 清理策略
            try:
                if hasattr(info.strategy, 'cleanup'):
                    info.strategy.cleanup()
            except Exception as e:
                logger.error(f"策略 {strategy_id} 清理失敗: {e}")
            
            del self._strategies[strategy_id]
            logger.info(f"移除策略: {strategy_id}")
            
            return True
    
    def get_strategy(self, strategy_id: str) -> Optional["BaseStrategy"]:
        """取得策略實例"""
        info = self._strategies.get(strategy_id)
        return info.strategy if info else None
    
    def get_strategy_info(self, strategy_id: str) -> Optional[StrategyInfo]:
        """取得策略資訊"""
        return self._strategies.get(strategy_id)
    
    def get_all_strategies(self) -> List[str]:
        """取得所有策略 ID"""
        return list(self._strategies.keys())
    
    # ========== 策略控制 ==========
    
    def _start_strategy(self, strategy_id: str) -> bool:
        """內部方法：啟動單一策略"""
        info = self._strategies.get(strategy_id)
        if info is None:
            return False
        
        if info.state == StrategyState.RUNNING:
            return True
        
        try:
            if hasattr(info.strategy, 'on_start'):
                info.strategy.on_start()
            
            info.state = StrategyState.RUNNING
            info.started_at = datetime.now()
            logger.info(f"策略 {strategy_id} 已啟動")
            return True
            
        except Exception as e:
            info.state = StrategyState.ERROR
            info.last_error = str(e)
            info.last_error_time = datetime.now()
            logger.error(f"策略 {strategy_id} 啟動失敗: {e}")
            return False
    
    def _stop_strategy(self, strategy_id: str) -> bool:
        """內部方法：停止單一策略"""
        info = self._strategies.get(strategy_id)
        if info is None:
            return False
        
        if info.state != StrategyState.RUNNING:
            return True
        
        try:
            if hasattr(info.strategy, 'on_stop'):
                info.strategy.on_stop()
            
            info.state = StrategyState.STOPPED
            info.stopped_at = datetime.now()
            logger.info(f"策略 {strategy_id} 已停止")
            return True
            
        except Exception as e:
            info.state = StrategyState.ERROR
            info.last_error = str(e)
            info.last_error_time = datetime.now()
            logger.error(f"策略 {strategy_id} 停止失敗: {e}")
            return False
    
    def start_strategy(self, strategy_id: str) -> bool:
        """啟動指定策略"""
        with self._lock:
            return self._start_strategy(strategy_id)
    
    def stop_strategy(self, strategy_id: str) -> bool:
        """停止指定策略"""
        with self._lock:
            return self._stop_strategy(strategy_id)
    
    def pause_strategy(self, strategy_id: str) -> bool:
        """暫停指定策略"""
        with self._lock:
            info = self._strategies.get(strategy_id)
            if info is None:
                return False
            
            if info.state == StrategyState.RUNNING:
                info.state = StrategyState.PAUSED
                logger.info(f"策略 {strategy_id} 已暫停")
                return True
            return False
    
    def resume_strategy(self, strategy_id: str) -> bool:
        """恢復指定策略"""
        with self._lock:
            info = self._strategies.get(strategy_id)
            if info is None:
                return False
            
            if info.state == StrategyState.PAUSED:
                info.state = StrategyState.RUNNING
                logger.info(f"策略 {strategy_id} 已恢復")
                return True
            return False
    
    # ========== 引擎控制 ==========
    
    def start(self) -> None:
        """啟動策略引擎"""
        if self._running:
            logger.warning("StrategyEngine 已在運行中")
            return
        
        # 訂閱事件
        self._event_bus.subscribe(EventType.TICK, self._on_tick, priority=10)
        self._event_bus.subscribe(EventType.BAR, self._on_bar, priority=10)
        self._event_bus.subscribe(EventType.FILL, self._on_fill, priority=10)
        self._event_bus.subscribe(EventType.POSITION, self._on_position, priority=10)
        
        self._running = True
        self._paused = False
        
        # 啟動所有已初始化的策略
        with self._lock:
            for strategy_id, info in self._strategies.items():
                if info.state == StrategyState.INITIALIZED:
                    self._start_strategy(strategy_id)
        
        logger.info("StrategyEngine 已啟動")
    
    def stop(self) -> None:
        """停止策略引擎"""
        if not self._running:
            return
        
        # 停止所有策略
        with self._lock:
            for strategy_id in list(self._strategies.keys()):
                self._stop_strategy(strategy_id)
        
        # 取消訂閱
        self._event_bus.unsubscribe(EventType.TICK, self._on_tick)
        self._event_bus.unsubscribe(EventType.BAR, self._on_bar)
        self._event_bus.unsubscribe(EventType.FILL, self._on_fill)
        self._event_bus.unsubscribe(EventType.POSITION, self._on_position)
        
        self._running = False
        logger.info("StrategyEngine 已停止")
    
    def pause(self) -> None:
        """暫停引擎（不處理事件）"""
        self._paused = True
        logger.info("StrategyEngine 已暫停")
    
    def resume(self) -> None:
        """恢復引擎"""
        self._paused = False
        logger.info("StrategyEngine 已恢復")
    
    # ========== 事件處理 ==========
    
    def _on_tick(self, event: TickEvent) -> None:
        """處理 TickEvent"""
        if self._paused:
            return
        
        symbol = event.symbol
        self._dispatch_event(symbol, event, "on_tick")
    
    def _on_bar(self, event: BarEvent) -> None:
        """處理 BarEvent"""
        if self._paused:
            return
        
        symbol = event.symbol
        self._dispatch_event(symbol, event, "on_bar")
    
    def _on_fill(self, event: FillEvent) -> None:
        """處理 FillEvent"""
        if self._paused:
            return
        
        symbol = event.symbol
        self._dispatch_event(symbol, event, "on_fill")
    
    def _on_position(self, event: PositionEvent) -> None:
        """處理 PositionEvent"""
        if self._paused:
            return
        
        symbol = event.symbol
        self._dispatch_event(symbol, event, "on_position")
    
    def _dispatch_event(
        self,
        symbol: str,
        event: Event,
        handler_name: str,
    ) -> None:
        """
        分發事件給對應的策略
        
        Args:
            symbol: 標的代碼
            event: 事件
            handler_name: 處理方法名稱
        """
        self._total_events_processed += 1
        
        # 取得訂閱該 symbol 的策略
        strategy_ids = self._symbol_strategies.get(symbol, [])
        
        for strategy_id in strategy_ids:
            info = self._strategies.get(strategy_id)
            if info is None:
                continue
            
            # 只處理運行中的策略
            if info.state != StrategyState.RUNNING:
                continue
            
            # 更新統計
            info.last_event_time = datetime.now()
            if isinstance(event, TickEvent):
                info.tick_count += 1
            elif isinstance(event, BarEvent):
                info.bar_count += 1
            
            # 調用策略處理方法
            try:
                handler = getattr(info.strategy, handler_name, None)
                if handler is not None:
                    result = handler(event)
                    
                    # 如果返回 SignalEvent，發布到事件總線
                    if isinstance(result, SignalEvent):
                        self._handle_signal(info, result)
                    elif isinstance(result, list):
                        for item in result:
                            if isinstance(item, SignalEvent):
                                self._handle_signal(info, item)
                                
            except Exception as e:
                self._handle_strategy_error(info, e, handler_name)
    
    def _handle_signal(self, info: StrategyInfo, signal: SignalEvent) -> None:
        """處理策略產生的信號"""
        # 確保信號包含策略 ID
        if not signal.strategy_id:
            signal.strategy_id = info.strategy_id
        
        info.signal_count += 1
        self._total_signals_generated += 1
        
        # 發布信號事件
        self._event_bus.publish(signal)
        
        logger.debug(
            f"策略 {info.strategy_id} 產生信號: "
            f"{signal.symbol} {signal.action.value} strength={signal.strength}"
        )
    
    def _handle_strategy_error(
        self,
        info: StrategyInfo,
        error: Exception,
        context: str,
    ) -> None:
        """處理策略錯誤"""
        info.error_count += 1
        info.last_error = f"{context}: {str(error)}"
        info.last_error_time = datetime.now()
        
        logger.error(
            f"策略 {info.strategy_id} 錯誤 ({context}): {error}\n"
            f"{traceback.format_exc()}"
        )
        
        # 發布系統事件
        self._event_bus.publish(SystemEvent(
            event_type=EventType.SYSTEM,
            sub_type=SystemEventType.ERROR,
            message=f"策略 {info.strategy_id} 錯誤: {error}",
            source="strategy_engine",
            data={
                "strategy_id": info.strategy_id,
                "error": str(error),
                "context": context,
            },
        ))
        
        # 檢查是否需要暫停策略
        if info.error_count >= self._max_errors_before_pause:
            info.state = StrategyState.PAUSED
            logger.warning(
                f"策略 {info.strategy_id} 錯誤次數達到 {info.error_count}，已自動暫停"
            )
    
    # ========== 統計 ==========
    
    def get_stats(self) -> Dict[str, Any]:
        """取得引擎統計"""
        with self._lock:
            strategy_stats = {}
            for strategy_id, info in self._strategies.items():
                strategy_stats[strategy_id] = {
                    "state": info.state.name,
                    "symbols": list(info.symbols),
                    "tick_count": info.tick_count,
                    "bar_count": info.bar_count,
                    "signal_count": info.signal_count,
                    "error_count": info.error_count,
                    "last_error": info.last_error,
                }
            
            return {
                "running": self._running,
                "paused": self._paused,
                "strategy_count": len(self._strategies),
                "active_count": self.active_strategy_count,
                "total_events": self._total_events_processed,
                "total_signals": self._total_signals_generated,
                "strategies": strategy_stats,
            }
    
    def reset_stats(self) -> None:
        """重置統計"""
        self._total_events_processed = 0
        self._total_signals_generated = 0
        
        with self._lock:
            for info in self._strategies.values():
                info.tick_count = 0
                info.bar_count = 0
                info.signal_count = 0
                info.error_count = 0