"""
Performance 模組 - 性能監控

提供系統性能追蹤和監控功能
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Callable
import statistics

# 可選依賴
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# 設定 logger
logger = logging.getLogger(__name__)


@dataclass
class LatencyStats:
    """延遲統計"""
    
    count: int = 0
    total: float = 0.0
    min: float = float('inf')
    max: float = 0.0
    avg: float = 0.0
    p50: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            "count": self.count,
            "total_ms": f"{self.total:.2f}",
            "min_ms": f"{self.min:.3f}" if self.min != float('inf') else "N/A",
            "max_ms": f"{self.max:.3f}",
            "avg_ms": f"{self.avg:.3f}",
            "p50_ms": f"{self.p50:.3f}",
            "p90_ms": f"{self.p90:.3f}",
            "p95_ms": f"{self.p95:.3f}",
            "p99_ms": f"{self.p99:.3f}",
        }


@dataclass
class ThroughputStats:
    """吞吐量統計"""
    
    total_events: int = 0
    events_per_second: float = 0.0
    peak_events_per_second: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            "total_events": self.total_events,
            "events_per_second": f"{self.events_per_second:.2f}",
            "peak_events_per_second": f"{self.peak_events_per_second:.2f}",
        }


@dataclass
class SystemStats:
    """系統資源統計"""
    
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    thread_count: int = 0
    
    # 進程資訊
    process_cpu_percent: float = 0.0
    process_memory_mb: float = 0.0
    process_threads: int = 0
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            "cpu_percent": f"{self.cpu_percent:.1f}%",
            "memory_percent": f"{self.memory_percent:.1f}%",
            "memory_used_mb": f"{self.memory_used_mb:.1f}",
            "memory_available_mb": f"{self.memory_available_mb:.1f}",
            "thread_count": self.thread_count,
            "process_cpu_percent": f"{self.process_cpu_percent:.1f}%",
            "process_memory_mb": f"{self.process_memory_mb:.1f}",
            "process_threads": self.process_threads,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PerformanceMetrics:
    """綜合性能指標"""
    
    latency: LatencyStats = field(default_factory=LatencyStats)
    throughput: ThroughputStats = field(default_factory=ThroughputStats)
    system: SystemStats = field(default_factory=SystemStats)
    
    # 自訂指標
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    # 時間資訊
    start_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    
    @property
    def uptime(self) -> timedelta:
        """運行時間"""
        return datetime.now() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            "latency": self.latency.to_dict(),
            "throughput": self.throughput.to_dict(),
            "system": self.system.to_dict(),
            "custom_metrics": self.custom_metrics,
            "uptime_seconds": self.uptime.total_seconds(),
            "start_time": self.start_time.isoformat(),
            "last_update": self.last_update.isoformat(),
        }


class LatencyTracker:
    """
    延遲追蹤器
    
    追蹤特定操作的延遲
    """
    
    def __init__(self, name: str, max_samples: int = 10000):
        """
        初始化
        
        Args:
            name: 追蹤器名稱
            max_samples: 最大樣本數
        """
        self.name = name
        self._samples: deque = deque(maxlen=max_samples)
        self._lock = threading.Lock()
        
        # 累計統計
        self._count = 0
        self._total = 0.0
        self._min = float('inf')
        self._max = 0.0
    
    def record(self, latency_ms: float) -> None:
        """記錄延遲（毫秒）"""
        with self._lock:
            self._samples.append(latency_ms)
            self._count += 1
            self._total += latency_ms
            self._min = min(self._min, latency_ms)
            self._max = max(self._max, latency_ms)
    
    def record_duration(self, start_time: float) -> float:
        """
        記錄從開始時間到現在的延遲
        
        Args:
            start_time: time.perf_counter() 的返回值
            
        Returns:
            延遲毫秒數
        """
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.record(latency_ms)
        return latency_ms
    
    def get_stats(self) -> LatencyStats:
        """取得統計"""
        with self._lock:
            if not self._samples:
                return LatencyStats()
            
            samples = list(self._samples)
            sorted_samples = sorted(samples)
            
            return LatencyStats(
                count=self._count,
                total=self._total,
                min=self._min,
                max=self._max,
                avg=self._total / self._count if self._count > 0 else 0,
                p50=self._percentile(sorted_samples, 50),
                p90=self._percentile(sorted_samples, 90),
                p95=self._percentile(sorted_samples, 95),
                p99=self._percentile(sorted_samples, 99),
            )
    
    def _percentile(self, sorted_data: List[float], p: float) -> float:
        """計算百分位數"""
        if not sorted_data:
            return 0.0
        
        k = (len(sorted_data) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f
        
        if f == c:
            return sorted_data[f]
        
        return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)
    
    def reset(self) -> None:
        """重置統計"""
        with self._lock:
            self._samples.clear()
            self._count = 0
            self._total = 0.0
            self._min = float('inf')
            self._max = 0.0


class ThroughputTracker:
    """
    吞吐量追蹤器
    
    追蹤事件處理的吞吐量
    """
    
    def __init__(self, name: str, window_seconds: int = 60):
        """
        初始化
        
        Args:
            name: 追蹤器名稱
            window_seconds: 統計窗口（秒）
        """
        self.name = name
        self._window = window_seconds
        self._events: deque = deque()
        self._lock = threading.Lock()
        
        # 累計統計
        self._total_events = 0
        self._peak_rate = 0.0
        self._start_time = time.time()
    
    def record(self, count: int = 1) -> None:
        """記錄事件"""
        now = time.time()
        
        with self._lock:
            self._events.append((now, count))
            self._total_events += count
            
            # 清理過期數據
            cutoff = now - self._window
            while self._events and self._events[0][0] < cutoff:
                self._events.popleft()
            
            # 更新峰值
            current_rate = self._calculate_rate()
            self._peak_rate = max(self._peak_rate, current_rate)
    
    def _calculate_rate(self) -> float:
        """計算當前速率"""
        if not self._events:
            return 0.0
        
        total = sum(count for _, count in self._events)
        time_span = time.time() - self._events[0][0]
        
        if time_span <= 0:
            return 0.0
        
        return total / time_span
    
    def get_stats(self) -> ThroughputStats:
        """取得統計"""
        with self._lock:
            return ThroughputStats(
                total_events=self._total_events,
                events_per_second=self._calculate_rate(),
                peak_events_per_second=self._peak_rate,
            )
    
    def reset(self) -> None:
        """重置統計"""
        with self._lock:
            self._events.clear()
            self._total_events = 0
            self._peak_rate = 0.0
            self._start_time = time.time()


class PerformanceMonitor:
    """
    性能監控器
    
    追蹤系統整體性能
    
    使用方式:
        monitor = PerformanceMonitor()
        monitor.start()
        
        # 記錄延遲
        start = time.perf_counter()
        # ... 執行操作 ...
        monitor.record_latency("order_processing", start)
        
        # 記錄事件
        monitor.record_event("tick_received")
        
        # 取得統計
        stats = monitor.get_stats()
    """
    
    def __init__(
        self,
        report_interval: int = 60,
        enable_system_monitoring: bool = True,
        enable_auto_report: bool = True,
    ):
        """
        初始化性能監控器
        
        Args:
            report_interval: 報告間隔（秒）
            enable_system_monitoring: 是否啟用系統資源監控
            enable_auto_report: 是否自動輸出報告
        """
        self._report_interval = report_interval
        self._enable_system_monitoring = enable_system_monitoring and HAS_PSUTIL
        self._enable_auto_report = enable_auto_report
        
        # 延遲追蹤器
        self._latency_trackers: Dict[str, LatencyTracker] = {}
        
        # 吞吐量追蹤器
        self._throughput_trackers: Dict[str, ThroughputTracker] = {}
        
        # 自訂指標
        self._custom_metrics: Dict[str, float] = {}
        
        # 系統統計歷史
        self._system_stats_history: deque = deque(maxlen=60)
        
        # 運行狀態
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._start_time = datetime.now()
        
        # 線程安全
        self._lock = threading.Lock()
        
        # 進程資訊
        self._process = psutil.Process() if HAS_PSUTIL else None
        
        logger.info("PerformanceMonitor 初始化完成")
    
    # ========== 控制 ==========
    
    def start(self) -> None:
        """啟動監控"""
        if self._running:
            logger.warning("PerformanceMonitor 已在運行中")
            return
        
        self._running = True
        self._start_time = datetime.now()
        
        # 啟動監控線程
        if self._enable_system_monitoring or self._enable_auto_report:
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
            )
            self._monitor_thread.start()
        
        logger.info("PerformanceMonitor 已啟動")
    
    def stop(self) -> None:
        """停止監控"""
        if not self._running:
            return
        
        self._running = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
            self._monitor_thread = None
        
        logger.info("PerformanceMonitor 已停止")
    
    def _monitor_loop(self) -> None:
        """監控循環"""
        last_report = time.time()
        
        while self._running:
            try:
                # 收集系統統計
                if self._enable_system_monitoring:
                    stats = self._collect_system_stats()
                    self._system_stats_history.append(stats)
                
                # 定期報告
                now = time.time()
                if self._enable_auto_report and now - last_report >= self._report_interval:
                    self._output_report()
                    last_report = now
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"監控循環錯誤: {e}")
    
    # ========== 延遲追蹤 ==========
    
    def record_latency(
        self,
        name: str,
        start_time: Optional[float] = None,
        latency_ms: Optional[float] = None,
    ) -> Optional[float]:
        """
        記錄延遲
        
        Args:
            name: 操作名稱
            start_time: time.perf_counter() 的返回值
            latency_ms: 直接指定延遲毫秒數
            
        Returns:
            延遲毫秒數
        """
        with self._lock:
            if name not in self._latency_trackers:
                self._latency_trackers[name] = LatencyTracker(name)
            
            tracker = self._latency_trackers[name]
        
        if latency_ms is not None:
            tracker.record(latency_ms)
            return latency_ms
        elif start_time is not None:
            return tracker.record_duration(start_time)
        
        return None
    
    def get_latency_tracker(self, name: str) -> LatencyTracker:
        """取得或建立延遲追蹤器"""
        with self._lock:
            if name not in self._latency_trackers:
                self._latency_trackers[name] = LatencyTracker(name)
            return self._latency_trackers[name]
    
    # ========== 吞吐量追蹤 ==========
    
    def record_event(self, name: str, count: int = 1) -> None:
        """
        記錄事件（用於吞吐量統計）
        
        Args:
            name: 事件名稱
            count: 事件數量
        """
        with self._lock:
            if name not in self._throughput_trackers:
                self._throughput_trackers[name] = ThroughputTracker(name)
            
            self._throughput_trackers[name].record(count)
    
    def get_throughput_tracker(self, name: str) -> ThroughputTracker:
        """取得或建立吞吐量追蹤器"""
        with self._lock:
            if name not in self._throughput_trackers:
                self._throughput_trackers[name] = ThroughputTracker(name)
            return self._throughput_trackers[name]
    
    # ========== 自訂指標 ==========
    
    def set_metric(self, name: str, value: float) -> None:
        """設定自訂指標"""
        with self._lock:
            self._custom_metrics[name] = value
    
    def increment_metric(self, name: str, delta: float = 1) -> None:
        """增加自訂指標"""
        with self._lock:
            self._custom_metrics[name] = self._custom_metrics.get(name, 0) + delta
    
    def get_metric(self, name: str) -> Optional[float]:
        """取得自訂指標"""
        return self._custom_metrics.get(name)
    
    # ========== 系統資源監控 ==========
    
    def _collect_system_stats(self) -> SystemStats:
        """收集系統統計"""
        if not HAS_PSUTIL:
            return SystemStats()
        
        try:
            # 系統資源
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            # 進程資源
            process_cpu = 0.0
            process_memory = 0.0
            process_threads = 0
            
            if self._process:
                try:
                    process_cpu = self._process.cpu_percent()
                    process_memory = self._process.memory_info().rss / 1024 / 1024
                    process_threads = self._process.num_threads()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            return SystemStats(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / 1024 / 1024,
                memory_available_mb=memory.available / 1024 / 1024,
                thread_count=threading.active_count(),
                process_cpu_percent=process_cpu,
                process_memory_mb=process_memory,
                process_threads=process_threads,
            )
            
        except Exception as e:
            logger.error(f"收集系統統計錯誤: {e}")
            return SystemStats()
    
    def get_system_stats(self) -> SystemStats:
        """取得當前系統統計"""
        return self._collect_system_stats()
    
    # ========== 統計查詢 ==========
    
    def get_stats(self) -> PerformanceMetrics:
        """取得綜合統計"""
        metrics = PerformanceMetrics(
            start_time=self._start_time,
            last_update=datetime.now(),
        )
        
        # 合併所有延遲統計
        all_latencies = []
        for tracker in self._latency_trackers.values():
            stats = tracker.get_stats()
            if stats.count > 0:
                # 使用加權平均
                all_latencies.extend([stats.avg] * stats.count)
        
        if all_latencies:
            sorted_latencies = sorted(all_latencies)
            metrics.latency = LatencyStats(
                count=len(all_latencies),
                total=sum(all_latencies),
                min=min(all_latencies),
                max=max(all_latencies),
                avg=statistics.mean(all_latencies),
                p50=statistics.median(all_latencies),
                p90=sorted_latencies[int(len(sorted_latencies) * 0.9)],
                p95=sorted_latencies[int(len(sorted_latencies) * 0.95)],
                p99=sorted_latencies[int(len(sorted_latencies) * 0.99)],
            )
        
        # 合併所有吞吐量統計
        total_events = 0
        total_rate = 0.0
        peak_rate = 0.0
        
        for tracker in self._throughput_trackers.values():
            stats = tracker.get_stats()
            total_events += stats.total_events
            total_rate += stats.events_per_second
            peak_rate = max(peak_rate, stats.peak_events_per_second)
        
        metrics.throughput = ThroughputStats(
            total_events=total_events,
            events_per_second=total_rate,
            peak_events_per_second=peak_rate,
        )
        
        # 系統統計
        metrics.system = self._collect_system_stats()
        
        # 自訂指標
        metrics.custom_metrics = self._custom_metrics.copy()
        
        return metrics
    
    def get_latency_stats(self, name: str) -> Optional[LatencyStats]:
        """取得特定操作的延遲統計"""
        tracker = self._latency_trackers.get(name)
        if tracker:
            return tracker.get_stats()
        return None
    
    def get_throughput_stats(self, name: str) -> Optional[ThroughputStats]:
        """取得特定事件的吞吐量統計"""
        tracker = self._throughput_trackers.get(name)
        if tracker:
            return tracker.get_stats()
        return None
    
    def get_all_latency_stats(self) -> Dict[str, LatencyStats]:
        """取得所有延遲統計"""
        return {
            name: tracker.get_stats()
            for name, tracker in self._latency_trackers.items()
        }
    
    def get_all_throughput_stats(self) -> Dict[str, ThroughputStats]:
        """取得所有吞吐量統計"""
        return {
            name: tracker.get_stats()
            for name, tracker in self._throughput_trackers.items()
        }
    
    # ========== 報告 ==========
    
    def _output_report(self) -> None:
        """輸出性能報告到日誌"""
        metrics = self.get_stats()
        
        # 格式化報告
        report_lines = [
            "=" * 50,
            "Performance Report",
            "=" * 50,
            f"Uptime: {metrics.uptime}",
            "",
            "--- System ---",
            f"CPU: {metrics.system.cpu_percent:.1f}%",
            f"Memory: {metrics.system.memory_percent:.1f}% "
            f"({metrics.system.memory_used_mb:.0f} MB used)",
            f"Process: CPU {metrics.system.process_cpu_percent:.1f}%, "
            f"Memory {metrics.system.process_memory_mb:.0f} MB, "
            f"Threads {metrics.system.process_threads}",
            "",
            "--- Throughput ---",
            f"Total Events: {metrics.throughput.total_events}",
            f"Events/sec: {metrics.throughput.events_per_second:.2f}",
            f"Peak Events/sec: {metrics.throughput.peak_events_per_second:.2f}",
        ]
        
        # 延遲統計
        if self._latency_trackers:
            report_lines.append("")
            report_lines.append("--- Latency (ms) ---")
            for name, tracker in self._latency_trackers.items():
                stats = tracker.get_stats()
                if stats.count > 0:
                    report_lines.append(
                        f"  {name}: avg={stats.avg:.2f}, "
                        f"p99={stats.p99:.2f}, max={stats.max:.2f} "
                        f"(n={stats.count})"
                    )
        
        # 自訂指標
        if self._custom_metrics:
            report_lines.append("")
            report_lines.append("--- Custom Metrics ---")
            for name, value in self._custom_metrics.items():
                report_lines.append(f"  {name}: {value}")
        
        report_lines.append("=" * 50)
        
        # 輸出到日誌
        logger.info("\n".join(report_lines))
    
    def get_report(self) -> str:
        """取得報告字串"""
        metrics = self.get_stats()
        return str(metrics.to_dict())
    
    # ========== 重置 ==========
    
    def reset(self) -> None:
        """重置所有統計"""
        with self._lock:
            for tracker in self._latency_trackers.values():
                tracker.reset()
            for tracker in self._throughput_trackers.values():
                tracker.reset()
            self._custom_metrics.clear()
            self._system_stats_history.clear()
            self._start_time = datetime.now()
        
        logger.info("PerformanceMonitor 統計已重置")


# ============================================================
# 便捷裝飾器
# ============================================================

def measure_latency(
    monitor: PerformanceMonitor,
    name: str,
) -> Callable:
    """
    測量函數執行延遲的裝飾器
    
    使用方式:
        @measure_latency(monitor, "my_function")
        def my_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                monitor.record_latency(name, start_time=start)
        return wrapper
    return decorator


def count_calls(
    monitor: PerformanceMonitor,
    name: str,
) -> Callable:
    """
    計數函數調用的裝飾器
    
    使用方式:
        @count_calls(monitor, "my_function")
        def my_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            monitor.record_event(name)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================
# 全局實例
# ============================================================

_monitor: Optional[PerformanceMonitor] = None
_monitor_lock = threading.Lock()


def get_performance_monitor() -> PerformanceMonitor:
    """取得全局 PerformanceMonitor 實例"""
    global _monitor
    
    if _monitor is None:
        with _monitor_lock:
            if _monitor is None:
                _monitor = PerformanceMonitor()
    
    return _monitor


def reset_performance_monitor() -> None:
    """重置全局監控器"""
    global _monitor
    
    with _monitor_lock:
        if _monitor:
            _monitor.stop()
        _monitor = None