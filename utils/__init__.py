"""
Utils 模組 - 工具函數層

提供通用工具函數和輔助類：
- 日誌設定
- 通知服務
- 市場時間工具
- 時間工具
- 性能監控

注意：此檔案不能直接執行，需作為套件導入使用：
    from utils import setup_logger, get_logger
    
或從專案根目錄執行：
    python -c "from utils import setup_logger; print('OK')"
"""

from utils.logger import (
    setup_logger,
    get_logger,
    set_log_level,
    LogLevel,
)

from utils.notifier import (
    Notifier,
    NotificationLevel,
    NotificationChannel,
    NotificationConfig,
)

from utils.market_hours import (
    # 類
    MarketCalendar,
    MarketHours,
    MarketType,
    MarketSession,
    # 便捷函數
    is_market_open,
    is_trading_day,
    get_market_hours,
    get_next_market_open,
    get_next_market_close,
    get_next_trading_day,
    get_market_status,
    time_until_market_open,
    format_duration,
    # 時區轉換
    get_eastern_time,
    get_utc_time,
    get_taiwan_time,
    get_us_market_hours_in_taiwan_time,
    get_calendar,
)

from utils.performance import (
    PerformanceMonitor,
    measure_latency,
    count_calls,
    get_performance_monitor,
)

__all__ = [
    # 日誌
    "setup_logger",
    "get_logger",
    "set_log_level",
    "LogLevel",
    # 通知
    "Notifier",
    "NotificationLevel",
    "NotificationChannel",
    "NotificationConfig",
    # 市場時間
    "MarketCalendar",
    "MarketHours",
    "MarketType",
    "MarketSession",
    "is_market_open",
    "is_trading_day",
    "get_market_hours",
    "get_next_market_open",
    "get_next_market_close",
    "get_next_trading_day",
    "get_market_status",
    "time_until_market_open",
    "format_duration",
    "get_eastern_time",
    "get_utc_time",
    "get_taiwan_time",
    "get_us_market_hours_in_taiwan_time",
    "get_calendar",
    # 性能監控
    "PerformanceMonitor",
    "measure_latency",
    "count_calls",
    "get_performance_monitor",
]


# 如果直接執行此檔案，顯示使用說明
if __name__ == "__main__":
    print(__doc__)
    print("可用的匯出：")
    for item in __all__:
        print(f"  - {item}")