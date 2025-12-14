"""
TimeUtils 模組 - 時間工具

提供時間相關的工具函數，整合市場時間功能
"""

import time as _time
from datetime import datetime, date, time, timedelta
from typing import Optional, Union, Callable
from functools import wraps
from zoneinfo import ZoneInfo


# ============================================================
# 時區定義
# ============================================================

EASTERN_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")
TAIWAN_TZ = ZoneInfo("Asia/Taipei")
LONDON_TZ = ZoneInfo("Europe/London")
TOKYO_TZ = ZoneInfo("Asia/Tokyo")


# ============================================================
# 美股市場時間常數
# ============================================================

# 盤前交易開始
PRE_MARKET_OPEN = time(4, 0)      # 04:00 ET

# 正常交易時段
MARKET_OPEN = time(9, 30)         # 09:30 ET
MARKET_CLOSE = time(16, 0)        # 16:00 ET

# 盤後交易結束
AFTER_HOURS_CLOSE = time(20, 0)   # 20:00 ET

# 半日交易收盤
HALF_DAY_CLOSE = time(13, 0)      # 13:00 ET


# ============================================================
# 時間取得函數
# ============================================================

def get_eastern_time(dt: Optional[datetime] = None) -> datetime:
    """
    取得美東時間
    
    Args:
        dt: datetime，None 表示當前時間
        
    Returns:
        美東時間的 datetime
        
    使用方式:
        # 取得當前美東時間
        et = get_eastern_time()
        
        # 轉換指定時間
        et = get_eastern_time(some_datetime)
    """
    if dt is None:
        return datetime.now(EASTERN_TZ)
    elif dt.tzinfo is None:
        # 假設無時區的 datetime 是 UTC
        dt = dt.replace(tzinfo=UTC_TZ)
    
    return dt.astimezone(EASTERN_TZ)


def get_utc_time(dt: Optional[datetime] = None) -> datetime:
    """取得 UTC 時間"""
    if dt is None:
        return datetime.now(UTC_TZ)
    elif dt.tzinfo is None:
        return dt.replace(tzinfo=UTC_TZ)
    return dt.astimezone(UTC_TZ)


def get_taiwan_time(dt: Optional[datetime] = None) -> datetime:
    """取得台灣時間"""
    if dt is None:
        return datetime.now(TAIWAN_TZ)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC_TZ)
    return dt.astimezone(TAIWAN_TZ)


def get_local_time(dt: Optional[datetime] = None) -> datetime:
    """取得本地時間"""
    if dt is None:
        return datetime.now()
    elif dt.tzinfo is None:
        return dt
    return dt.astimezone()


def now() -> datetime:
    """取得當前 UTC 時間"""
    return datetime.now(UTC_TZ)


def now_eastern() -> datetime:
    """取得當前美東時間"""
    return datetime.now(EASTERN_TZ)


def now_taiwan() -> datetime:
    """取得當前台灣時間"""
    return datetime.now(TAIWAN_TZ)


# ============================================================
# 市場時間函數
# ============================================================

def get_market_hours(d: Optional[date] = None) -> dict:
    """
    取得市場開收盤時間
    
    Args:
        d: 日期，None 表示今天
        
    Returns:
        包含各時段時間的字典
        
    使用方式:
        hours = get_market_hours()
        print(hours['market_open'])
    """
    if d is None:
        d = get_eastern_time().date()
    
    return {
        'date': d,
        'pre_market_open': datetime.combine(d, PRE_MARKET_OPEN, tzinfo=EASTERN_TZ),
        'market_open': datetime.combine(d, MARKET_OPEN, tzinfo=EASTERN_TZ),
        'market_close': datetime.combine(d, MARKET_CLOSE, tzinfo=EASTERN_TZ),
        'after_hours_close': datetime.combine(d, AFTER_HOURS_CLOSE, tzinfo=EASTERN_TZ),
    }


def is_market_open(
    dt: Optional[datetime] = None,
    include_extended: bool = False,
) -> bool:
    """
    檢查是否在交易時段
    
    Args:
        dt: datetime，None 表示當前時間
        include_extended: 是否包含盤前盤後交易時段
        
    Returns:
        是否在交易時段
        
    使用方式:
        # 檢查是否在正常交易時段
        if is_market_open():
            print("市場開盤中")
        
        # 檢查是否在延長交易時段（含盤前盤後）
        if is_market_open(include_extended=True):
            print("延長交易時段")
    """
    et = get_eastern_time(dt)
    current_time = et.time()
    
    # 檢查是否週末
    if et.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    
    if include_extended:
        # 盤前到盤後
        return PRE_MARKET_OPEN <= current_time < AFTER_HOURS_CLOSE
    else:
        # 只有正常交易時段
        return MARKET_OPEN <= current_time < MARKET_CLOSE


def is_pre_market(dt: Optional[datetime] = None) -> bool:
    """檢查是否在盤前交易時段"""
    et = get_eastern_time(dt)
    current_time = et.time()
    
    if et.weekday() >= 5:
        return False
    
    return PRE_MARKET_OPEN <= current_time < MARKET_OPEN


def is_after_hours(dt: Optional[datetime] = None) -> bool:
    """檢查是否在盤後交易時段"""
    et = get_eastern_time(dt)
    current_time = et.time()
    
    if et.weekday() >= 5:
        return False
    
    return MARKET_CLOSE <= current_time < AFTER_HOURS_CLOSE


def is_regular_hours(dt: Optional[datetime] = None) -> bool:
    """檢查是否在正常交易時段"""
    return is_market_open(dt, include_extended=False)


def is_trading_day(d: Optional[date] = None) -> bool:
    """
    檢查是否為交易日（不含假日檢查）
    
    注意：此函數只檢查週末，不檢查假日
    完整假日檢查請使用 market_hours 模組
    """
    if d is None:
        d = get_eastern_time().date()
    return d.weekday() < 5


def time_until_market_open(dt: Optional[datetime] = None) -> timedelta:
    """
    計算距離開盤的時間差
    
    Args:
        dt: datetime，None 表示當前時間
        
    Returns:
        timedelta，已開盤時返回 timedelta(0)
        
    使用方式:
        remaining = time_until_market_open()
        print(f"距離開盤: {remaining}")
    """
    et = get_eastern_time(dt)
    current_time = et.time()
    today = et.date()
    
    # 如果今天是交易日且還沒開盤
    if et.weekday() < 5 and current_time < MARKET_OPEN:
        market_open_dt = datetime.combine(today, MARKET_OPEN, tzinfo=EASTERN_TZ)
        return market_open_dt - et
    
    # 如果今天是交易日且在交易時段
    if et.weekday() < 5 and MARKET_OPEN <= current_time < MARKET_CLOSE:
        return timedelta(0)
    
    # 找下一個交易日
    next_day = today + timedelta(days=1)
    while next_day.weekday() >= 5:  # 跳過週末
        next_day += timedelta(days=1)
    
    next_open = datetime.combine(next_day, MARKET_OPEN, tzinfo=EASTERN_TZ)
    return next_open - et


def time_until_market_close(dt: Optional[datetime] = None) -> timedelta:
    """
    計算距離收盤的時間差
    
    Args:
        dt: datetime，None 表示當前時間
        
    Returns:
        timedelta，已收盤時返回 timedelta(0)
    """
    et = get_eastern_time(dt)
    current_time = et.time()
    today = et.date()
    
    # 如果在交易時段
    if et.weekday() < 5 and current_time < MARKET_CLOSE:
        market_close_dt = datetime.combine(today, MARKET_CLOSE, tzinfo=EASTERN_TZ)
        return market_close_dt - et
    
    return timedelta(0)


def get_next_market_open(dt: Optional[datetime] = None) -> datetime:
    """取得下一次開盤時間"""
    et = get_eastern_time(dt)
    current_time = et.time()
    today = et.date()
    
    # 今天還沒開盤
    if et.weekday() < 5 and current_time < MARKET_OPEN:
        return datetime.combine(today, MARKET_OPEN, tzinfo=EASTERN_TZ)
    
    # 找下一個交易日
    next_day = today + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    
    return datetime.combine(next_day, MARKET_OPEN, tzinfo=EASTERN_TZ)


def get_next_market_close(dt: Optional[datetime] = None) -> datetime:
    """取得下一次收盤時間"""
    et = get_eastern_time(dt)
    current_time = et.time()
    today = et.date()
    
    # 今天還沒收盤
    if et.weekday() < 5 and current_time < MARKET_CLOSE:
        return datetime.combine(today, MARKET_CLOSE, tzinfo=EASTERN_TZ)
    
    # 找下一個交易日
    next_day = today + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    
    return datetime.combine(next_day, MARKET_CLOSE, tzinfo=EASTERN_TZ)


# ============================================================
# 時間格式化
# ============================================================

def format_duration(td: timedelta) -> str:
    """
    格式化時間差為易讀字串
    
    Args:
        td: timedelta
        
    Returns:
        格式化字串，如 "2h 30m 15s"
    """
    total_seconds = int(td.total_seconds())
    
    if total_seconds < 0:
        return "-" + format_duration(timedelta(seconds=-total_seconds))
    
    if total_seconds == 0:
        return "0s"
    
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 and days == 0:  # 超過一天不顯示秒
        parts.append(f"{seconds}s")
    
    return " ".join(parts)


def format_timestamp(
    dt: datetime,
    fmt: str = "%Y-%m-%d %H:%M:%S",
    include_tz: bool = True,
) -> str:
    """
    格式化時間戳
    
    Args:
        dt: datetime
        fmt: 格式字串
        include_tz: 是否包含時區
        
    Returns:
        格式化字串
    """
    result = dt.strftime(fmt)
    if include_tz and dt.tzinfo:
        result += f" {dt.tzinfo}"
    return result


def format_time_ago(dt: datetime) -> str:
    """
    格式化為「多久以前」
    
    Args:
        dt: datetime
        
    Returns:
        如 "5 minutes ago", "2 hours ago"
    """
    now_dt = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
    diff = now_dt - dt
    
    seconds = int(diff.total_seconds())
    
    if seconds < 0:
        return "in the future"
    elif seconds < 60:
        return f"{seconds} seconds ago"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    elif seconds < 86400:
        hours = seconds // 3600
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    else:
        days = seconds // 86400
        return f"{days} day{'s' if days > 1 else ''} ago"


# ============================================================
# 時間解析
# ============================================================

def parse_time(
    time_str: str,
    formats: Optional[list] = None,
) -> Optional[datetime]:
    """
    解析時間字串
    
    Args:
        time_str: 時間字串
        formats: 格式列表，None 使用預設
        
    Returns:
        datetime 或 None
    """
    if formats is None:
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d",
            "%d/%m/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
            "%H:%M:%S",
            "%H:%M",
        ]
    
    for fmt in formats:
        try:
            return datetime.strptime(time_str, fmt)
        except ValueError:
            continue
    
    return None


def parse_duration(duration_str: str) -> timedelta:
    """
    解析時間長度字串
    
    Args:
        duration_str: 如 "1h", "30m", "1h30m", "2d"
        
    Returns:
        timedelta
    """
    import re
    
    total_seconds = 0
    
    # 匹配各種時間單位
    patterns = [
        (r'(\d+)d', 86400),   # days
        (r'(\d+)h', 3600),    # hours
        (r'(\d+)m', 60),      # minutes
        (r'(\d+)s', 1),       # seconds
    ]
    
    for pattern, multiplier in patterns:
        match = re.search(pattern, duration_str)
        if match:
            total_seconds += int(match.group(1)) * multiplier
    
    return timedelta(seconds=total_seconds)


# ============================================================
# 時間範圍
# ============================================================

def get_today_range(tz: Optional[ZoneInfo] = None) -> tuple:
    """取得今天的開始和結束時間"""
    tz = tz or UTC_TZ
    today = datetime.now(tz).date()
    start = datetime.combine(today, time.min, tzinfo=tz)
    end = datetime.combine(today, time.max, tzinfo=tz)
    return start, end


def get_week_range(tz: Optional[ZoneInfo] = None) -> tuple:
    """取得本週的開始（週一）和結束（週日）時間"""
    tz = tz or UTC_TZ
    today = datetime.now(tz).date()
    start_of_week = today - timedelta(days=today.weekday())
    end_of_week = start_of_week + timedelta(days=6)
    
    start = datetime.combine(start_of_week, time.min, tzinfo=tz)
    end = datetime.combine(end_of_week, time.max, tzinfo=tz)
    return start, end


def get_month_range(tz: Optional[ZoneInfo] = None) -> tuple:
    """取得本月的開始和結束時間"""
    tz = tz or UTC_TZ
    today = datetime.now(tz).date()
    start_of_month = today.replace(day=1)
    
    # 計算月底
    if today.month == 12:
        end_of_month = today.replace(year=today.year + 1, month=1, day=1) - timedelta(days=1)
    else:
        end_of_month = today.replace(month=today.month + 1, day=1) - timedelta(days=1)
    
    start = datetime.combine(start_of_month, time.min, tzinfo=tz)
    end = datetime.combine(end_of_month, time.max, tzinfo=tz)
    return start, end


# ============================================================
# 工具裝飾器
# ============================================================

def timeit(func: Callable) -> Callable:
    """
    計時裝飾器
    
    使用方式:
        @timeit
        def slow_function():
            time.sleep(1)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = _time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = _time.perf_counter() - start
        print(f"{func.__name__} 執行時間: {elapsed:.4f}s")
        return result
    return wrapper


def retry_until_market_open(
    check_interval: int = 60,
    max_wait: int = 86400,
) -> Callable:
    """
    等待市場開盤的裝飾器
    
    Args:
        check_interval: 檢查間隔（秒）
        max_wait: 最大等待時間（秒）
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            waited = 0
            while not is_market_open() and waited < max_wait:
                remaining = time_until_market_open()
                print(f"等待市場開盤... 剩餘 {format_duration(remaining)}")
                _time.sleep(min(check_interval, remaining.total_seconds()))
                waited += check_interval
            
            if is_market_open():
                return func(*args, **kwargs)
            else:
                raise TimeoutError("等待市場開盤超時")
        return wrapper
    return decorator


# ============================================================
# 匯出
# ============================================================

__all__ = [
    # 時區
    "EASTERN_TZ",
    "UTC_TZ",
    "TAIWAN_TZ",
    "LONDON_TZ",
    "TOKYO_TZ",
    # 市場時間常數
    "PRE_MARKET_OPEN",
    "MARKET_OPEN",
    "MARKET_CLOSE",
    "AFTER_HOURS_CLOSE",
    "HALF_DAY_CLOSE",
    # 時間取得
    "get_eastern_time",
    "get_utc_time",
    "get_taiwan_time",
    "get_local_time",
    "now",
    "now_eastern",
    "now_taiwan",
    # 市場時間
    "get_market_hours",
    "is_market_open",
    "is_pre_market",
    "is_after_hours",
    "is_regular_hours",
    "is_trading_day",
    "time_until_market_open",
    "time_until_market_close",
    "get_next_market_open",
    "get_next_market_close",
    # 格式化
    "format_duration",
    "format_timestamp",
    "format_time_ago",
    # 解析
    "parse_time",
    "parse_duration",
    # 時間範圍
    "get_today_range",
    "get_week_range",
    "get_month_range",
    # 裝飾器
    "timeit",
    "retry_until_market_open",
]