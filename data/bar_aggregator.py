"""
BarAggregator 模組 - Tick 轉 Bar 聚合器

將 Tick 數據聚合成多個時間週期的 K 線數據
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, List, Deque, Callable
import threading

from core.events import TickEvent, BarEvent, EventType
from core.event_bus import EventBus, get_event_bus


# 設定 logger
logger = logging.getLogger(__name__)


class BarSize(Enum):
    """
    K 線時間週期枚舉
    
    值為秒數，方便計算
    """
    
    SEC_1 = 1           # 1 秒
    SEC_5 = 5           # 5 秒
    SEC_10 = 10         # 10 秒
    SEC_15 = 15         # 15 秒
    SEC_30 = 30         # 30 秒
    MIN_1 = 60          # 1 分鐘
    MIN_2 = 120         # 2 分鐘
    MIN_3 = 180         # 3 分鐘
    MIN_5 = 300         # 5 分鐘
    MIN_10 = 600        # 10 分鐘
    MIN_15 = 900        # 15 分鐘
    MIN_30 = 1800       # 30 分鐘
    HOUR_1 = 3600       # 1 小時
    HOUR_2 = 7200       # 2 小時
    HOUR_4 = 14400      # 4 小時
    DAY_1 = 86400       # 1 天
    
    @property
    def seconds(self) -> int:
        """取得秒數"""
        return self.value
    
    @property
    def label(self) -> str:
        """取得顯示標籤"""
        labels = {
            1: "1s", 5: "5s", 10: "10s", 15: "15s", 30: "30s",
            60: "1min", 120: "2min", 180: "3min", 300: "5min",
            600: "10min", 900: "15min", 1800: "30min",
            3600: "1hour", 7200: "2hour", 14400: "4hour",
            86400: "1day",
        }
        return labels.get(self.value, f"{self.value}s")
    
    @classmethod
    def from_string(cls, s: str) -> "BarSize":
        """
        從字串解析 BarSize
        
        支援格式: "1s", "5s", "1min", "5min", "1hour", "1day" 等
        """
        s = s.lower().strip()
        mapping = {
            "1s": cls.SEC_1, "5s": cls.SEC_5, "10s": cls.SEC_10,
            "15s": cls.SEC_15, "30s": cls.SEC_30,
            "1min": cls.MIN_1, "2min": cls.MIN_2, "3min": cls.MIN_3,
            "5min": cls.MIN_5, "10min": cls.MIN_10, "15min": cls.MIN_15,
            "30min": cls.MIN_30,
            "1hour": cls.HOUR_1, "2hour": cls.HOUR_2, "4hour": cls.HOUR_4,
            "1day": cls.DAY_1,
            # IB 格式
            "1 sec": cls.SEC_1, "5 secs": cls.SEC_5,
            "1 min": cls.MIN_1, "5 mins": cls.MIN_5,
            "15 mins": cls.MIN_15, "30 mins": cls.MIN_30,
            "1 hour": cls.HOUR_1, "1 day": cls.DAY_1,
        }
        if s in mapping:
            return mapping[s]
        raise ValueError(f"無法解析 BarSize: {s}")


@dataclass
class BarBuilder:
    """
    K 棒建構器
    
    用於累積 Tick 數據並建構單一 K 棒
    """
    
    symbol: str
    bar_size: BarSize
    bar_start: datetime
    
    # OHLCV
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: int = 0
    
    # 額外統計
    tick_count: int = 0
    vwap_sum: float = 0.0  # 用於計算 VWAP
    
    @property
    def bar_end(self) -> datetime:
        """K 棒結束時間"""
        return self.bar_start + timedelta(seconds=self.bar_size.seconds)
    
    @property
    def is_empty(self) -> bool:
        """是否沒有數據"""
        return self.tick_count == 0
    
    @property
    def vwap(self) -> Optional[float]:
        """成交量加權平均價"""
        if self.volume > 0:
            return self.vwap_sum / self.volume
        return self.close
    
    def update(self, price: float, size: int = 0) -> None:
        """
        更新 K 棒數據
        
        Args:
            price: 成交價
            size: 成交量
        """
        if price <= 0:
            return
        
        self.tick_count += 1
        
        # 更新 OHLC
        if self.open is None:
            self.open = price
        
        if self.high is None or price > self.high:
            self.high = price
        
        if self.low is None or price < self.low:
            self.low = price
        
        self.close = price
        
        # 更新成交量
        if size > 0:
            self.volume += size
            self.vwap_sum += price * size
    
    def to_bar_event(self) -> BarEvent:
        """轉換為 BarEvent"""
        return BarEvent(
            event_type=EventType.BAR,
            symbol=self.symbol,
            open=self.open or 0.0,
            high=self.high or 0.0,
            low=self.low or 0.0,
            close=self.close or 0.0,
            volume=self.volume,
            bar_size=self.bar_size.label,
            bar_start=self.bar_start,
            bar_end=self.bar_end,
            vwap=self.vwap,
            trade_count=self.tick_count,
        )
    
    def reset(self, new_bar_start: datetime) -> None:
        """
        重置建構器，開始新的 K 棒
        
        Args:
            new_bar_start: 新 K 棒的開始時間
        """
        self.bar_start = new_bar_start
        self.open = None
        self.high = None
        self.low = None
        self.close = None
        self.volume = 0
        self.tick_count = 0
        self.vwap_sum = 0.0


@dataclass
class SymbolAggregator:
    """
    單一標的的多週期聚合器
    
    管理一個標的的多個時間週期 K 棒建構
    """
    
    symbol: str
    bar_sizes: List[BarSize]
    history_size: int = 100
    
    # 各週期的 BarBuilder
    builders: Dict[BarSize, BarBuilder] = field(default_factory=dict)
    
    # 歷史 K 棒緩存: {BarSize: deque[BarEvent]}
    history: Dict[BarSize, Deque[BarEvent]] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化各週期的建構器和歷史緩存"""
        now = datetime.now()
        
        for bar_size in self.bar_sizes:
            # 對齊時間
            aligned_start = self._align_time(now, bar_size)
            
            self.builders[bar_size] = BarBuilder(
                symbol=self.symbol,
                bar_size=bar_size,
                bar_start=aligned_start,
            )
            
            self.history[bar_size] = deque(maxlen=self.history_size)
    
    def _align_time(self, dt: datetime, bar_size: BarSize) -> datetime:
        """
        對齊時間到 K 棒邊界
        
        例如：對於 5 分鐘 K 棒，10:03:25 會對齊到 10:00:00
        """
        seconds = bar_size.seconds
        
        # 計算當天開始的秒數
        day_start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        elapsed = (dt - day_start).total_seconds()
        
        # 對齊到週期邊界
        aligned_elapsed = (int(elapsed) // seconds) * seconds
        
        return day_start + timedelta(seconds=aligned_elapsed)
    
    def update(self, tick: TickEvent) -> List[BarEvent]:
        """
        更新所有週期的 K 棒
        
        Args:
            tick: TickEvent
            
        Returns:
            完成的 BarEvent 列表
        """
        completed_bars = []
        now = tick.timestamp
        price = tick.last or tick.mid
        size = tick.last_size or 0
        
        if price is None or price <= 0:
            return completed_bars
        
        for bar_size, builder in self.builders.items():
            # 檢查是否需要關閉當前 K 棒
            if now >= builder.bar_end:
                # 完成當前 K 棒（如果有數據）
                if not builder.is_empty:
                    bar_event = builder.to_bar_event()
                    completed_bars.append(bar_event)
                    self.history[bar_size].append(bar_event)
                
                # 重置到新的 K 棒
                new_start = self._align_time(now, bar_size)
                builder.reset(new_start)
            
            # 更新當前 K 棒
            builder.update(price, size)
        
        return completed_bars
    
    def get_history(self, bar_size: BarSize, count: Optional[int] = None) -> List[BarEvent]:
        """
        取得歷史 K 棒
        
        Args:
            bar_size: K 棒週期
            count: 數量，None 為全部
            
        Returns:
            BarEvent 列表（從舊到新）
        """
        history = self.history.get(bar_size)
        if history is None:
            return []
        
        if count is None:
            return list(history)
        
        return list(history)[-count:]
    
    def get_current_bar(self, bar_size: BarSize) -> Optional[BarEvent]:
        """
        取得當前未完成的 K 棒
        
        Args:
            bar_size: K 棒週期
            
        Returns:
            當前 BarEvent（可能不完整）
        """
        builder = self.builders.get(bar_size)
        if builder is None or builder.is_empty:
            return None
        
        return builder.to_bar_event()
    
    def force_complete(self, bar_size: Optional[BarSize] = None) -> List[BarEvent]:
        """
        強制完成當前 K 棒
        
        Args:
            bar_size: 指定週期，None 為所有週期
            
        Returns:
            完成的 BarEvent 列表
        """
        completed_bars = []
        now = datetime.now()
        
        sizes = [bar_size] if bar_size else list(self.builders.keys())
        
        for bs in sizes:
            builder = self.builders.get(bs)
            if builder and not builder.is_empty:
                bar_event = builder.to_bar_event()
                completed_bars.append(bar_event)
                self.history[bs].append(bar_event)
                
                new_start = self._align_time(now, bs)
                builder.reset(new_start)
        
        return completed_bars


class BarAggregator:
    """
    Bar 聚合器
    
    訂閱 TickEvent，將 Tick 數據聚合成多個時間週期的 K 棒
    
    使用方式:
        aggregator = BarAggregator(event_bus)
        
        # 添加標的
        aggregator.add_symbol("AAPL", [BarSize.MIN_1, BarSize.MIN_5])
        aggregator.add_symbol("MSFT", [BarSize.SEC_5, BarSize.MIN_1])
        
        # 啟動（訂閱 TickEvent）
        aggregator.start()
        
        # 取得歷史 K 棒
        bars = aggregator.get_history("AAPL", BarSize.MIN_5, count=20)
    """
    
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        default_bar_sizes: Optional[List[BarSize]] = None,
        history_size: int = 100,
        publish_bars: bool = True,
    ):
        """
        初始化 Bar 聚合器
        
        Args:
            event_bus: 事件總線
            default_bar_sizes: 預設的 K 棒週期列表
            history_size: 每個週期保留的歷史 K 棒數量
            publish_bars: 是否發布完成的 BarEvent
        """
        self._event_bus = event_bus or get_event_bus()
        self._default_bar_sizes = default_bar_sizes or [
            BarSize.SEC_5,
            BarSize.MIN_1,
            BarSize.MIN_5,
        ]
        self._history_size = history_size
        self._publish_bars = publish_bars
        
        # 各標的的聚合器: {symbol: SymbolAggregator}
        self._aggregators: Dict[str, SymbolAggregator] = {}
        
        # 運行狀態
        self._running = False
        
        # 線程安全
        self._lock = threading.RLock()
        
        # 統計
        self._tick_count = 0
        self._bar_count = 0
        
        # 回調
        self._on_bar_callbacks: List[Callable[[BarEvent], None]] = []
        
        logger.debug("BarAggregator 初始化完成")
    
    # ========== 標的管理 ==========
    
    def add_symbol(
        self,
        symbol: str,
        bar_sizes: Optional[List[BarSize]] = None,
    ) -> None:
        """
        添加要聚合的標的
        
        Args:
            symbol: 標的代碼
            bar_sizes: K 棒週期列表，None 使用預設
        """
        with self._lock:
            if symbol in self._aggregators:
                logger.warning(f"標的 {symbol} 已存在，跳過")
                return
            
            sizes = bar_sizes or self._default_bar_sizes
            
            self._aggregators[symbol] = SymbolAggregator(
                symbol=symbol,
                bar_sizes=sizes,
                history_size=self._history_size,
            )
            
            logger.info(f"添加標的 {symbol}，週期: {[s.label for s in sizes]}")
    
    def remove_symbol(self, symbol: str) -> bool:
        """
        移除標的
        
        Args:
            symbol: 標的代碼
            
        Returns:
            是否成功移除
        """
        with self._lock:
            if symbol in self._aggregators:
                del self._aggregators[symbol]
                logger.info(f"移除標的 {symbol}")
                return True
            return False
    
    def get_symbols(self) -> List[str]:
        """取得所有標的"""
        with self._lock:
            return list(self._aggregators.keys())
    
    # ========== 控制 ==========
    
    def start(self) -> None:
        """啟動聚合器，訂閱 TickEvent"""
        if self._running:
            logger.warning("BarAggregator 已在運行中")
            return
        
        self._event_bus.subscribe(EventType.TICK, self._on_tick)
        self._running = True
        logger.info("BarAggregator 已啟動")
    
    def stop(self) -> None:
        """停止聚合器"""
        if not self._running:
            return
        
        self._event_bus.unsubscribe(EventType.TICK, self._on_tick)
        self._running = False
        logger.info("BarAggregator 已停止")
    
    # ========== 事件處理 ==========
    
    def _on_tick(self, event: TickEvent) -> None:
        """處理 TickEvent"""
        symbol = event.symbol
        
        with self._lock:
            aggregator = self._aggregators.get(symbol)
            if aggregator is None:
                return
            
            self._tick_count += 1
            
            # 更新聚合器
            completed_bars = aggregator.update(event)
            
            # 處理完成的 K 棒
            for bar_event in completed_bars:
                self._bar_count += 1
                
                # 發布 BarEvent
                if self._publish_bars:
                    self._event_bus.publish(bar_event)
                
                # 執行回調
                for callback in self._on_bar_callbacks:
                    try:
                        callback(bar_event)
                    except Exception as e:
                        logger.error(f"Bar 回調錯誤: {e}")
    
    def on_bar(self, callback: Callable[[BarEvent], None]) -> Callable:
        """
        註冊 K 棒完成回調
        
        可作為裝飾器:
            @aggregator.on_bar
            def handle_bar(bar: BarEvent):
                print(bar)
        """
        self._on_bar_callbacks.append(callback)
        return callback
    
    # ========== 數據查詢 ==========
    
    def get_history(
        self,
        symbol: str,
        bar_size: BarSize,
        count: Optional[int] = None,
    ) -> List[BarEvent]:
        """
        取得歷史 K 棒
        
        Args:
            symbol: 標的代碼
            bar_size: K 棒週期
            count: 數量，None 為全部
            
        Returns:
            BarEvent 列表（從舊到新）
        """
        with self._lock:
            aggregator = self._aggregators.get(symbol)
            if aggregator is None:
                return []
            
            return aggregator.get_history(bar_size, count)
    
    def get_current_bar(
        self,
        symbol: str,
        bar_size: BarSize,
    ) -> Optional[BarEvent]:
        """
        取得當前未完成的 K 棒
        
        Args:
            symbol: 標的代碼
            bar_size: K 棒週期
            
        Returns:
            當前 BarEvent
        """
        with self._lock:
            aggregator = self._aggregators.get(symbol)
            if aggregator is None:
                return None
            
            return aggregator.get_current_bar(bar_size)
    
    def get_latest_bar(
        self,
        symbol: str,
        bar_size: BarSize,
    ) -> Optional[BarEvent]:
        """
        取得最新完成的 K 棒
        
        Args:
            symbol: 標的代碼
            bar_size: K 棒週期
            
        Returns:
            最新 BarEvent
        """
        history = self.get_history(symbol, bar_size, count=1)
        return history[-1] if history else None
    
    def force_complete_all(self) -> List[BarEvent]:
        """
        強制完成所有標的的當前 K 棒
        
        Returns:
            完成的 BarEvent 列表
        """
        completed_bars = []
        
        with self._lock:
            for aggregator in self._aggregators.values():
                bars = aggregator.force_complete()
                completed_bars.extend(bars)
                
                # 發布和回調
                for bar_event in bars:
                    self._bar_count += 1
                    
                    if self._publish_bars:
                        self._event_bus.publish(bar_event)
                    
                    for callback in self._on_bar_callbacks:
                        try:
                            callback(bar_event)
                        except Exception as e:
                            logger.error(f"Bar 回調錯誤: {e}")
        
        return completed_bars
    
    # ========== 統計 ==========
    
    def get_stats(self) -> Dict[str, any]:
        """取得統計資訊"""
        return {
            "running": self._running,
            "symbols": len(self._aggregators),
            "tick_count": self._tick_count,
            "bar_count": self._bar_count,
        }
    
    def reset_stats(self) -> None:
        """重置統計"""
        self._tick_count = 0
        self._bar_count = 0