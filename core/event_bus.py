"""
EventBus 模組 - 事件總線

實現發布/訂閱模式的事件總線，支援同步與異步處理
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Any, Union
import threading
import time

from .events import Event, EventType


# 設定 logger
logger = logging.getLogger(__name__)


# 類型別名
EventHandler = Callable[[Event], Any]
AsyncEventHandler = Callable[[Event], Any]


@dataclass
class HandlerInfo:
    """Handler 資訊"""
    
    handler: Callable
    is_async: bool
    priority: int = 0  # 數字越大優先級越高
    name: str = ""
    
    def __post_init__(self):
        if not self.name:
            self.name = getattr(self.handler, "__name__", str(self.handler))


@dataclass
class EventBusStats:
    """事件總線統計資訊"""
    
    total_published: int = 0
    total_processed: int = 0
    events_by_type: Dict[EventType, int] = field(default_factory=lambda: defaultdict(int))
    processing_errors: int = 0
    avg_processing_time_ms: float = 0.0
    
    # 追蹤最近處理時間
    _processing_times: List[float] = field(default_factory=list)
    _max_tracking: int = 1000
    
    def record_processing_time(self, time_ms: float):
        """記錄處理時間"""
        self._processing_times.append(time_ms)
        if len(self._processing_times) > self._max_tracking:
            self._processing_times.pop(0)
        if self._processing_times:
            self.avg_processing_time_ms = sum(self._processing_times) / len(self._processing_times)


class EventBus:
    """
    事件總線
    
    支援同步與異步事件處理的發布/訂閱系統
    
    使用方式:
        event_bus = EventBus()
        
        # 訂閱事件
        @event_bus.subscribe(EventType.BAR)
        def on_bar(event):
            print(f"收到 Bar: {event.symbol}")
        
        # 發布事件
        event_bus.publish(bar_event)
        
        # 異步運行
        await event_bus.start()
    """
    
    def __init__(self, use_async_queue: bool = True, queue_size: int = 10000):
        """
        初始化事件總線
        
        Args:
            use_async_queue: 是否使用異步隊列
            queue_size: 異步隊列大小
        """
        # Handler 存儲: {EventType: [HandlerInfo, ...]}
        self._handlers: Dict[EventType, List[HandlerInfo]] = defaultdict(list)
        
        # 異步隊列
        self._use_async_queue = use_async_queue
        self._queue_size = queue_size
        self._queue: Optional[asyncio.Queue] = None
        
        # 運行狀態
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
        # 統計
        self._stats = EventBusStats()
        
        # 線程安全鎖
        self._lock = threading.RLock()
        
        # 暫停控制
        self._paused = False
        
        logger.debug("EventBus 初始化完成")
    
    # ========== 訂閱管理 ==========
    
    def subscribe(
        self,
        event_type: Union[EventType, List[EventType]],
        handler: Optional[Callable] = None,
        priority: int = 0,
    ) -> Callable:
        """
        訂閱事件
        
        可作為裝飾器或直接調用:
            # 裝飾器方式
            @event_bus.subscribe(EventType.BAR)
            def on_bar(event):
                pass
            
            # 直接調用
            event_bus.subscribe(EventType.BAR, on_bar)
        
        Args:
            event_type: 事件類型（可以是單一類型或列表）
            handler: 事件處理函數
            priority: 優先級（越大越先執行）
            
        Returns:
            裝飾器或原始 handler
        """
        # 統一轉為列表
        event_types = [event_type] if isinstance(event_type, EventType) else event_type
        
        def decorator(func: Callable) -> Callable:
            is_async = asyncio.iscoroutinefunction(func)
            handler_info = HandlerInfo(
                handler=func,
                is_async=is_async,
                priority=priority,
            )
            
            with self._lock:
                for et in event_types:
                    self._handlers[et].append(handler_info)
                    # 按優先級排序（高優先級在前）
                    self._handlers[et].sort(key=lambda h: -h.priority)
                    logger.debug(
                        f"訂閱事件 {et.name}: {handler_info.name} "
                        f"(async={is_async}, priority={priority})"
                    )
            
            return func
        
        # 支援直接調用和裝飾器兩種方式
        if handler is not None:
            return decorator(handler)
        return decorator
    
    def unsubscribe(
        self,
        event_type: EventType,
        handler: Callable,
    ) -> bool:
        """
        取消訂閱
        
        Args:
            event_type: 事件類型
            handler: 要移除的處理函數
            
        Returns:
            是否成功移除
        """
        with self._lock:
            handlers = self._handlers.get(event_type, [])
            for i, info in enumerate(handlers):
                if info.handler == handler:
                    handlers.pop(i)
                    logger.debug(f"取消訂閱 {event_type.name}: {info.name}")
                    return True
        return False
    
    def unsubscribe_all(self, event_type: Optional[EventType] = None) -> int:
        """
        取消所有訂閱
        
        Args:
            event_type: 指定事件類型，None 表示所有
            
        Returns:
            移除的 handler 數量
        """
        with self._lock:
            if event_type is None:
                count = sum(len(handlers) for handlers in self._handlers.values())
                self._handlers.clear()
            else:
                count = len(self._handlers.get(event_type, []))
                self._handlers[event_type] = []
            
            logger.debug(f"取消訂閱 {count} 個 handlers")
            return count
    
    def get_handlers(self, event_type: EventType) -> List[HandlerInfo]:
        """取得指定事件類型的所有 handlers"""
        with self._lock:
            return list(self._handlers.get(event_type, []))
    
    def has_handlers(self, event_type: EventType) -> bool:
        """檢查是否有訂閱指定事件類型的 handlers"""
        with self._lock:
            return len(self._handlers.get(event_type, [])) > 0
    
    # ========== 事件發布 ==========
    
    def publish(self, event: Event) -> None:
        """
        發布事件（同步）
        
        如果事件總線正在異步運行，會加入隊列
        否則直接同步處理
        
        Args:
            event: 要發布的事件
        """
        if self._paused:
            return
        
        self._stats.total_published += 1
        self._stats.events_by_type[event.event_type] += 1
        
        if self._running and self._queue is not None:
            # 異步模式：加入隊列
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(f"事件隊列已滿，丟棄事件: {event.event_type.name}")
        else:
            # 同步模式：直接處理
            self._process_event_sync(event)
    
    async def publish_async(self, event: Event) -> None:
        """
        發布事件（異步）
        
        Args:
            event: 要發布的事件
        """
        if self._paused:
            return
        
        self._stats.total_published += 1
        self._stats.events_by_type[event.event_type] += 1
        
        if self._queue is not None:
            await self._queue.put(event)
        else:
            await self._process_event_async(event)
    
    def emit(self, event: Event) -> None:
        """publish 的別名"""
        self.publish(event)
    
    # ========== 事件處理 ==========
    
    def _process_event_sync(self, event: Event) -> None:
        """同步處理事件"""
        start_time = time.perf_counter()
        
        handlers = self.get_handlers(event.event_type)
        
        for handler_info in handlers:
            try:
                if handler_info.is_async:
                    # 在同步上下文中運行異步函數
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(handler_info.handler(event))
                    except RuntimeError:
                        # 沒有運行中的事件循環，創建新的來運行
                        asyncio.run(handler_info.handler(event))
                else:
                    handler_info.handler(event)
                    
                self._stats.total_processed += 1
                
            except Exception as e:
                self._stats.processing_errors += 1
                logger.error(
                    f"處理事件 {event.event_type.name} 時發生錯誤 "
                    f"(handler={handler_info.name}): {e}"
                )
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._stats.record_processing_time(elapsed_ms)
    
    async def _process_event_async(self, event: Event) -> None:
        """異步處理事件"""
        start_time = time.perf_counter()
        
        handlers = self.get_handlers(event.event_type)
        
        for handler_info in handlers:
            try:
                if handler_info.is_async:
                    await handler_info.handler(event)
                else:
                    handler_info.handler(event)
                    
                self._stats.total_processed += 1
                
            except Exception as e:
                self._stats.processing_errors += 1
                logger.error(
                    f"處理事件 {event.event_type.name} 時發生錯誤 "
                    f"(handler={handler_info.name}): {e}"
                )
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._stats.record_processing_time(elapsed_ms)
    
    # ========== 事件循環控制 ==========
    
    async def start(self) -> None:
        """
        啟動事件總線（異步模式）
        
        開始處理事件隊列
        """
        if self._running:
            logger.warning("EventBus 已在運行中")
            return
        
        self._running = True
        self._queue = asyncio.Queue(maxsize=self._queue_size)
        self._task = asyncio.create_task(self._event_loop())
        
        logger.info("EventBus 已啟動")
    
    async def stop(self, timeout: float = 5.0) -> None:
        """
        停止事件總線
        
        Args:
            timeout: 等待處理完剩餘事件的超時時間
        """
        if not self._running:
            return
        
        self._running = False
        
        # 等待隊列清空
        if self._queue is not None:
            try:
                await asyncio.wait_for(self._queue.join(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(f"等待隊列清空超時，剩餘 {self._queue.qsize()} 個事件")
        
        # 取消任務
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        
        self._queue = None
        logger.info("EventBus 已停止")
    
    async def _event_loop(self) -> None:
        """事件處理主循環"""
        logger.debug("事件循環開始")
        
        while self._running:
            try:
                # 等待事件，設置超時以便能夠響應停止信號
                try:
                    event = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue
                
                # 處理事件
                await self._process_event_async(event)
                self._queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"事件循環發生錯誤: {e}")
        
        logger.debug("事件循環結束")
    
    def pause(self) -> None:
        """暫停事件處理"""
        self._paused = True
        logger.info("EventBus 已暫停")
    
    def resume(self) -> None:
        """恢復事件處理"""
        self._paused = False
        logger.info("EventBus 已恢復")
    
    # ========== 狀態查詢 ==========
    
    @property
    def is_running(self) -> bool:
        """是否正在運行"""
        return self._running
    
    @property
    def is_paused(self) -> bool:
        """是否暫停中"""
        return self._paused
    
    @property
    def queue_size(self) -> int:
        """當前隊列中的事件數量"""
        if self._queue is not None:
            return self._queue.qsize()
        return 0
    
    @property
    def stats(self) -> EventBusStats:
        """取得統計資訊"""
        return self._stats
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """取得統計摘要"""
        return {
            "total_published": self._stats.total_published,
            "total_processed": self._stats.total_processed,
            "processing_errors": self._stats.processing_errors,
            "avg_processing_time_ms": round(self._stats.avg_processing_time_ms, 3),
            "queue_size": self.queue_size,
            "is_running": self._running,
            "is_paused": self._paused,
            "events_by_type": {
                et.name: count for et, count in self._stats.events_by_type.items()
            },
        }
    
    def reset_stats(self) -> None:
        """重置統計資訊"""
        self._stats = EventBusStats()
        logger.debug("統計資訊已重置")


# ============================================================
# 全局單例
# ============================================================

_event_bus: Optional[EventBus] = None
_event_bus_lock = threading.Lock()


def get_event_bus(
    use_async_queue: bool = True,
    queue_size: int = 10000,
) -> EventBus:
    """
    取得全局 EventBus 實例（單例模式）
    
    Args:
        use_async_queue: 是否使用異步隊列
        queue_size: 隊列大小
        
    Returns:
        EventBus 實例
    """
    global _event_bus
    
    if _event_bus is None:
        with _event_bus_lock:
            if _event_bus is None:
                _event_bus = EventBus(
                    use_async_queue=use_async_queue,
                    queue_size=queue_size,
                )
    
    return _event_bus


def reset_event_bus() -> None:
    """
    重置全局 EventBus
    
    主要用於測試
    """
    global _event_bus
    
    with _event_bus_lock:
        if _event_bus is not None:
            # 如果正在運行，需要先停止
            if _event_bus.is_running:
                logger.warning("重置運行中的 EventBus")
        _event_bus = None