"""
Cache 模組 - 市場數據內存快取

提供高效的內存快取，用於即時市場數據存取
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Deque, Any
import threading

from core.events import TickEvent, BarEvent, EventType
from core.event_bus import EventBus, get_event_bus


# 設定 logger
logger = logging.getLogger(__name__)


@dataclass
class TickCache:
    """
    單一標的的 Tick 快取
    """
    
    symbol: str
    max_size: int = 1000
    
    # 快取數據
    ticks: Deque[TickEvent] = field(default_factory=lambda: deque(maxlen=1000))
    
    # 最新數據（快速存取）
    latest: Optional[TickEvent] = None
    
    # 統計
    update_count: int = 0
    last_update_time: Optional[datetime] = None
    
    def __post_init__(self):
        """初始化時設定 deque maxlen"""
        self.ticks = deque(maxlen=self.max_size)
    
    def update(self, tick: TickEvent) -> None:
        """更新快取"""
        self.ticks.append(tick)
        self.latest = tick
        self.update_count += 1
        self.last_update_time = datetime.now()
    
    def get_latest(self) -> Optional[TickEvent]:
        """取得最新 Tick"""
        return self.latest
    
    def get_history(self, count: Optional[int] = None) -> List[TickEvent]:
        """取得歷史 Tick"""
        if count is None:
            return list(self.ticks)
        return list(self.ticks)[-count:]
    
    def clear(self) -> None:
        """清除快取"""
        self.ticks.clear()
        self.latest = None


@dataclass
class BarCache:
    """
    單一標的單一週期的 Bar 快取
    """
    
    symbol: str
    bar_size: str
    max_size: int = 500
    
    # 快取數據
    bars: Deque[BarEvent] = field(default_factory=lambda: deque(maxlen=500))
    
    # 最新數據
    latest: Optional[BarEvent] = None
    
    # 當前未完成的 Bar
    current: Optional[BarEvent] = None
    
    # 統計
    update_count: int = 0
    last_update_time: Optional[datetime] = None
    
    def __post_init__(self):
        """初始化時設定 deque maxlen"""
        self.bars = deque(maxlen=self.max_size)
    
    def update(self, bar: BarEvent, is_complete: bool = True) -> None:
        """
        更新快取
        
        Args:
            bar: BarEvent
            is_complete: 是否為完成的 Bar
        """
        if is_complete:
            self.bars.append(bar)
            self.latest = bar
            self.current = None
        else:
            self.current = bar
        
        self.update_count += 1
        self.last_update_time = datetime.now()
    
    def get_latest(self) -> Optional[BarEvent]:
        """取得最新完成的 Bar"""
        return self.latest
    
    def get_current(self) -> Optional[BarEvent]:
        """取得當前未完成的 Bar"""
        return self.current
    
    def get_history(self, count: Optional[int] = None) -> List[BarEvent]:
        """取得歷史 Bar"""
        if count is None:
            return list(self.bars)
        return list(self.bars)[-count:]
    
    def clear(self) -> None:
        """清除快取"""
        self.bars.clear()
        self.latest = None
        self.current = None


class MarketDataCache:
    """
    市場數據快取管理器
    
    提供高效的內存快取，用於即時數據存取
    
    使用方式:
        cache = MarketDataCache()
        
        # 更新數據
        cache.update_tick(tick_event)
        cache.update_bar(bar_event)
        
        # 查詢數據
        latest = cache.get_latest_tick("AAPL")
        bars = cache.get_bars("AAPL", "1min", count=20)
        
        # 自動訂閱事件總線
        cache.start()
    """
    
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        tick_cache_size: int = 1000,
        bar_cache_size: int = 500,
        auto_subscribe: bool = False,
    ):
        """
        初始化快取管理器
        
        Args:
            event_bus: 事件總線
            tick_cache_size: 每個標的的 Tick 快取大小
            bar_cache_size: 每個週期的 Bar 快取大小
            auto_subscribe: 是否自動訂閱事件總線
        """
        self._event_bus = event_bus or get_event_bus()
        self._tick_cache_size = tick_cache_size
        self._bar_cache_size = bar_cache_size
        
        # 快取存儲
        # Tick: {symbol: TickCache}
        self._tick_caches: Dict[str, TickCache] = {}
        
        # Bar: {symbol: {bar_size: BarCache}}
        self._bar_caches: Dict[str, Dict[str, BarCache]] = {}
        
        # 線程安全鎖
        self._lock = threading.RLock()
        
        # 運行狀態
        self._running = False
        
        # 統計
        self._total_tick_updates = 0
        self._total_bar_updates = 0
        
        if auto_subscribe:
            self.start()
        
        logger.debug("MarketDataCache 初始化完成")
    
    # ========== 控制 ==========
    
    def start(self) -> None:
        """啟動快取，訂閱事件"""
        if self._running:
            logger.warning("MarketDataCache 已在運行中")
            return
        
        self._event_bus.subscribe(EventType.TICK, self._on_tick)
        self._event_bus.subscribe(EventType.BAR, self._on_bar)
        self._running = True
        logger.info("MarketDataCache 已啟動")
    
    def stop(self) -> None:
        """停止快取，取消訂閱"""
        if not self._running:
            return
        
        self._event_bus.unsubscribe(EventType.TICK, self._on_tick)
        self._event_bus.unsubscribe(EventType.BAR, self._on_bar)
        self._running = False
        logger.info("MarketDataCache 已停止")
    
    # ========== 事件處理 ==========
    
    def _on_tick(self, event: TickEvent) -> None:
        """處理 TickEvent"""
        self.update_tick(event)
    
    def _on_bar(self, event: BarEvent) -> None:
        """處理 BarEvent"""
        self.update_bar(event)
    
    # ========== Tick 操作 ==========
    
    def update_tick(self, tick: TickEvent) -> None:
        """
        更新 Tick 快取
        
        Args:
            tick: TickEvent
        """
        symbol = tick.symbol
        
        with self._lock:
            # 確保快取存在
            if symbol not in self._tick_caches:
                self._tick_caches[symbol] = TickCache(
                    symbol=symbol,
                    max_size=self._tick_cache_size,
                )
            
            self._tick_caches[symbol].update(tick)
            self._total_tick_updates += 1
    
    def get_latest_tick(self, symbol: str) -> Optional[TickEvent]:
        """
        取得最新 Tick
        
        Args:
            symbol: 標的代碼
            
        Returns:
            最新的 TickEvent 或 None
        """
        with self._lock:
            cache = self._tick_caches.get(symbol)
            if cache is None:
                return None
            return cache.get_latest()
    
    def get_ticks(
        self,
        symbol: str,
        count: Optional[int] = None,
    ) -> List[TickEvent]:
        """
        取得歷史 Tick
        
        Args:
            symbol: 標的代碼
            count: 數量，None 為全部
            
        Returns:
            TickEvent 列表（從舊到新）
        """
        with self._lock:
            cache = self._tick_caches.get(symbol)
            if cache is None:
                return []
            return cache.get_history(count)
    
    def get_tick_stats(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        取得 Tick 快取統計
        
        Args:
            symbol: 標的代碼
            
        Returns:
            統計資訊
        """
        with self._lock:
            cache = self._tick_caches.get(symbol)
            if cache is None:
                return None
            
            return {
                "symbol": symbol,
                "count": len(cache.ticks),
                "max_size": cache.max_size,
                "update_count": cache.update_count,
                "last_update": cache.last_update_time,
            }
    
    # ========== Bar 操作 ==========
    
    def update_bar(self, bar: BarEvent, is_complete: bool = True) -> None:
        """
        更新 Bar 快取
        
        Args:
            bar: BarEvent
            is_complete: 是否為完成的 Bar
        """
        symbol = bar.symbol
        bar_size = bar.bar_size
        
        with self._lock:
            # 確保標的快取存在
            if symbol not in self._bar_caches:
                self._bar_caches[symbol] = {}
            
            # 確保週期快取存在
            if bar_size not in self._bar_caches[symbol]:
                self._bar_caches[symbol][bar_size] = BarCache(
                    symbol=symbol,
                    bar_size=bar_size,
                    max_size=self._bar_cache_size,
                )
            
            self._bar_caches[symbol][bar_size].update(bar, is_complete)
            self._total_bar_updates += 1
    
    def get_latest_bar(
        self,
        symbol: str,
        bar_size: str,
    ) -> Optional[BarEvent]:
        """
        取得最新完成的 Bar
        
        Args:
            symbol: 標的代碼
            bar_size: K 棒週期
            
        Returns:
            最新的 BarEvent 或 None
        """
        with self._lock:
            symbol_caches = self._bar_caches.get(symbol)
            if symbol_caches is None:
                return None
            
            cache = symbol_caches.get(bar_size)
            if cache is None:
                return None
            
            return cache.get_latest()
    
    def get_current_bar(
        self,
        symbol: str,
        bar_size: str,
    ) -> Optional[BarEvent]:
        """
        取得當前未完成的 Bar
        
        Args:
            symbol: 標的代碼
            bar_size: K 棒週期
            
        Returns:
            當前 BarEvent 或 None
        """
        with self._lock:
            symbol_caches = self._bar_caches.get(symbol)
            if symbol_caches is None:
                return None
            
            cache = symbol_caches.get(bar_size)
            if cache is None:
                return None
            
            return cache.get_current()
    
    def get_bars(
        self,
        symbol: str,
        bar_size: str,
        count: Optional[int] = None,
    ) -> List[BarEvent]:
        """
        取得歷史 Bar
        
        Args:
            symbol: 標的代碼
            bar_size: K 棒週期
            count: 數量，None 為全部
            
        Returns:
            BarEvent 列表（從舊到新）
        """
        with self._lock:
            symbol_caches = self._bar_caches.get(symbol)
            if symbol_caches is None:
                return []
            
            cache = symbol_caches.get(bar_size)
            if cache is None:
                return []
            
            return cache.get_history(count)
    
    def get_bar_stats(
        self,
        symbol: str,
        bar_size: str,
    ) -> Optional[Dict[str, Any]]:
        """
        取得 Bar 快取統計
        
        Args:
            symbol: 標的代碼
            bar_size: K 棒週期
            
        Returns:
            統計資訊
        """
        with self._lock:
            symbol_caches = self._bar_caches.get(symbol)
            if symbol_caches is None:
                return None
            
            cache = symbol_caches.get(bar_size)
            if cache is None:
                return None
            
            return {
                "symbol": symbol,
                "bar_size": bar_size,
                "count": len(cache.bars),
                "max_size": cache.max_size,
                "update_count": cache.update_count,
                "last_update": cache.last_update_time,
                "has_current": cache.current is not None,
            }
    
    # ========== 便捷方法 ==========
    
    def get_last_price(self, symbol: str) -> Optional[float]:
        """
        取得最新價格
        
        Args:
            symbol: 標的代碼
            
        Returns:
            最新價格或 None
        """
        tick = self.get_latest_tick(symbol)
        if tick:
            return tick.last or tick.mid
        return None
    
    def get_bid_ask(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        取得買賣報價
        
        Args:
            symbol: 標的代碼
            
        Returns:
            {"bid": float, "ask": float, "spread": float} 或 None
        """
        tick = self.get_latest_tick(symbol)
        if tick and tick.bid and tick.ask:
            return {
                "bid": tick.bid,
                "ask": tick.ask,
                "spread": tick.ask - tick.bid,
                "mid": (tick.bid + tick.ask) / 2,
            }
        return None
    
    def get_ohlc(
        self,
        symbol: str,
        bar_size: str,
        count: int = 1,
    ) -> Optional[Dict[str, List[float]]]:
        """
        取得 OHLC 數據（用於技術分析）
        
        Args:
            symbol: 標的代碼
            bar_size: K 棒週期
            count: Bar 數量
            
        Returns:
            {"open": [], "high": [], "low": [], "close": [], "volume": []}
        """
        bars = self.get_bars(symbol, bar_size, count)
        if not bars:
            return None
        
        return {
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
            "timestamp": [b.bar_start for b in bars],
        }
    
    # ========== 快取管理 ==========
    
    def clear_symbol(self, symbol: str) -> None:
        """
        清除指定標的的所有快取
        
        Args:
            symbol: 標的代碼
        """
        with self._lock:
            # 清除 Tick 快取
            if symbol in self._tick_caches:
                self._tick_caches[symbol].clear()
                del self._tick_caches[symbol]
            
            # 清除 Bar 快取
            if symbol in self._bar_caches:
                for cache in self._bar_caches[symbol].values():
                    cache.clear()
                del self._bar_caches[symbol]
            
            logger.info(f"清除標的快取: {symbol}")
    
    def clear_all(self) -> None:
        """清除所有快取"""
        with self._lock:
            # 清除 Tick
            for cache in self._tick_caches.values():
                cache.clear()
            self._tick_caches.clear()
            
            # 清除 Bar
            for symbol_caches in self._bar_caches.values():
                for cache in symbol_caches.values():
                    cache.clear()
            self._bar_caches.clear()
            
            logger.info("清除所有快取")
    
    def get_symbols(self) -> List[str]:
        """取得所有快取的標的"""
        with self._lock:
            tick_symbols = set(self._tick_caches.keys())
            bar_symbols = set(self._bar_caches.keys())
            return list(tick_symbols | bar_symbols)
    
    def get_bar_sizes(self, symbol: str) -> List[str]:
        """取得指定標的的所有 Bar 週期"""
        with self._lock:
            symbol_caches = self._bar_caches.get(symbol)
            if symbol_caches is None:
                return []
            return list(symbol_caches.keys())
    
    # ========== 統計 ==========
    
    def get_stats(self) -> Dict[str, Any]:
        """取得快取統計資訊"""
        with self._lock:
            tick_count = sum(len(c.ticks) for c in self._tick_caches.values())
            bar_count = sum(
                len(c.bars)
                for symbol_caches in self._bar_caches.values()
                for c in symbol_caches.values()
            )
            
            return {
                "running": self._running,
                "tick_symbols": len(self._tick_caches),
                "bar_symbols": len(self._bar_caches),
                "total_ticks": tick_count,
                "total_bars": bar_count,
                "tick_updates": self._total_tick_updates,
                "bar_updates": self._total_bar_updates,
            }
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """取得詳細統計資訊"""
        with self._lock:
            tick_stats = {
                symbol: {
                    "count": len(cache.ticks),
                    "updates": cache.update_count,
                    "last_update": cache.last_update_time,
                }
                for symbol, cache in self._tick_caches.items()
            }
            
            bar_stats = {}
            for symbol, symbol_caches in self._bar_caches.items():
                bar_stats[symbol] = {
                    bar_size: {
                        "count": len(cache.bars),
                        "updates": cache.update_count,
                        "last_update": cache.last_update_time,
                    }
                    for bar_size, cache in symbol_caches.items()
                }
            
            return {
                "running": self._running,
                "tick_stats": tick_stats,
                "bar_stats": bar_stats,
                "total_tick_updates": self._total_tick_updates,
                "total_bar_updates": self._total_bar_updates,
            }


# ============================================================
# 全局單例
# ============================================================

_cache: Optional[MarketDataCache] = None
_cache_lock = threading.Lock()


def get_market_data_cache(
    event_bus: Optional[EventBus] = None,
    tick_cache_size: int = 1000,
    bar_cache_size: int = 500,
    auto_subscribe: bool = False,
) -> MarketDataCache:
    """
    取得全局 MarketDataCache 實例（單例模式）
    
    Args:
        event_bus: 事件總線
        tick_cache_size: Tick 快取大小
        bar_cache_size: Bar 快取大小
        auto_subscribe: 是否自動訂閱
        
    Returns:
        MarketDataCache 實例
    """
    global _cache
    
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                _cache = MarketDataCache(
                    event_bus=event_bus,
                    tick_cache_size=tick_cache_size,
                    bar_cache_size=bar_cache_size,
                    auto_subscribe=auto_subscribe,
                )
    
    return _cache


def reset_market_data_cache() -> None:
    """重置全局快取（用於測試）"""
    global _cache
    
    with _cache_lock:
        if _cache is not None:
            _cache.stop()
            _cache.clear_all()
        _cache = None