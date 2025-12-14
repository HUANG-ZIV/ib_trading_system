"""
MarketHours 模組 - 市場交易時間管理

提供不同市場（美股、期貨、外匯）的交易時間管理
支援美國主要假日和半日交易日
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from enum import Enum, auto
from typing import Optional, Tuple, List, Dict, Set
from zoneinfo import ZoneInfo


# 設定 logger
logger = logging.getLogger(__name__)


# ============================================================
# 時區常數
# ============================================================

EASTERN_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")
TAIWAN_TZ = ZoneInfo("Asia/Taipei")
CHICAGO_TZ = ZoneInfo("America/Chicago")
LONDON_TZ = ZoneInfo("Europe/London")
TOKYO_TZ = ZoneInfo("Asia/Tokyo")


# ============================================================
# 市場類型
# ============================================================

class MarketType(Enum):
    """市場類型"""
    
    US_STOCK = "us_stock"           # 美國股票
    US_FUTURES = "us_futures"       # 美國期貨
    US_OPTIONS = "us_options"       # 美國選擇權
    FOREX = "forex"                 # 外匯
    CRYPTO = "crypto"               # 加密貨幣
    HK_STOCK = "hk_stock"           # 香港股票
    TW_STOCK = "tw_stock"           # 台灣股票


class MarketSession(Enum):
    """市場交易時段"""
    
    CLOSED = auto()          # 休市
    PRE_MARKET = auto()      # 盤前
    REGULAR = auto()         # 正常交易
    AFTER_HOURS = auto()     # 盤後
    OVERNIGHT = auto()       # 夜盤（期貨）


# ============================================================
# 市場時間配置
# ============================================================

@dataclass
class MarketTimeConfig:
    """市場時間配置"""
    
    market_type: MarketType
    timezone: ZoneInfo
    
    # 盤前交易
    pre_market_open: Optional[time] = None
    
    # 正常交易
    market_open: time = time(9, 30)
    market_close: time = time(16, 0)
    
    # 盤後交易
    after_hours_close: Optional[time] = None
    
    # 半日收盤時間
    half_day_close: time = time(13, 0)
    
    # 夜盤（期貨用）
    overnight_open: Optional[time] = None
    overnight_close: Optional[time] = None
    
    # 是否支援延長交易
    has_extended_hours: bool = False
    
    # 是否 24 小時交易
    is_24_hours: bool = False
    
    # 週末是否交易
    trades_on_weekend: bool = False


# 美國股票市場配置
US_STOCK_CONFIG = MarketTimeConfig(
    market_type=MarketType.US_STOCK,
    timezone=EASTERN_TZ,
    pre_market_open=time(4, 0),      # 04:00 ET
    market_open=time(9, 30),         # 09:30 ET
    market_close=time(16, 0),        # 16:00 ET
    after_hours_close=time(20, 0),   # 20:00 ET
    half_day_close=time(13, 0),      # 13:00 ET
    has_extended_hours=True,
)

# 美國期貨市場配置（CME E-mini）
US_FUTURES_CONFIG = MarketTimeConfig(
    market_type=MarketType.US_FUTURES,
    timezone=CHICAGO_TZ,
    market_open=time(9, 30),          # 09:30 CT (RTH)
    market_close=time(16, 0),         # 16:00 CT (RTH)
    overnight_open=time(18, 0),       # 18:00 CT (前一日)
    overnight_close=time(17, 0),      # 17:00 CT (當日)
    has_extended_hours=True,
)

# 美國選擇權市場配置
US_OPTIONS_CONFIG = MarketTimeConfig(
    market_type=MarketType.US_OPTIONS,
    timezone=EASTERN_TZ,
    market_open=time(9, 30),         # 09:30 ET
    market_close=time(16, 0),        # 16:00 ET
    half_day_close=time(13, 0),
    has_extended_hours=False,
)

# 外匯市場配置（24/5 交易）
FOREX_CONFIG = MarketTimeConfig(
    market_type=MarketType.FOREX,
    timezone=UTC_TZ,
    # 外匯從週日 17:00 ET 到週五 17:00 ET
    market_open=time(22, 0),         # 22:00 UTC (週日)
    market_close=time(22, 0),        # 22:00 UTC (週五)
    is_24_hours=True,
    trades_on_weekend=False,          # 週六日不交易
)

# 加密貨幣市場配置（24/7 交易）
CRYPTO_CONFIG = MarketTimeConfig(
    market_type=MarketType.CRYPTO,
    timezone=UTC_TZ,
    market_open=time(0, 0),
    market_close=time(0, 0),
    is_24_hours=True,
    trades_on_weekend=True,
)

# 香港股票市場配置
HK_STOCK_CONFIG = MarketTimeConfig(
    market_type=MarketType.HK_STOCK,
    timezone=ZoneInfo("Asia/Hong_Kong"),
    market_open=time(9, 30),         # 09:30 HKT
    market_close=time(16, 0),        # 16:00 HKT (午休 12:00-13:00)
    half_day_close=time(12, 0),
)

# 台灣股票市場配置
TW_STOCK_CONFIG = MarketTimeConfig(
    market_type=MarketType.TW_STOCK,
    timezone=TAIWAN_TZ,
    market_open=time(9, 0),          # 09:00 TST
    market_close=time(13, 30),       # 13:30 TST
)

# 市場配置映射
MARKET_CONFIGS: Dict[MarketType, MarketTimeConfig] = {
    MarketType.US_STOCK: US_STOCK_CONFIG,
    MarketType.US_FUTURES: US_FUTURES_CONFIG,
    MarketType.US_OPTIONS: US_OPTIONS_CONFIG,
    MarketType.FOREX: FOREX_CONFIG,
    MarketType.CRYPTO: CRYPTO_CONFIG,
    MarketType.HK_STOCK: HK_STOCK_CONFIG,
    MarketType.TW_STOCK: TW_STOCK_CONFIG,
}


# ============================================================
# 美國市場假日（2024-2026）
# ============================================================

# 美國股票市場假日
US_MARKET_HOLIDAYS_2024 = [
    date(2024, 1, 1),    # New Year's Day
    date(2024, 1, 15),   # Martin Luther King Jr. Day
    date(2024, 2, 19),   # Presidents' Day
    date(2024, 3, 29),   # Good Friday
    date(2024, 5, 27),   # Memorial Day
    date(2024, 6, 19),   # Juneteenth National Independence Day
    date(2024, 7, 4),    # Independence Day
    date(2024, 9, 2),    # Labor Day
    date(2024, 11, 28),  # Thanksgiving Day
    date(2024, 12, 25),  # Christmas Day
]

US_MARKET_HOLIDAYS_2025 = [
    date(2025, 1, 1),    # New Year's Day
    date(2025, 1, 20),   # Martin Luther King Jr. Day
    date(2025, 2, 17),   # Presidents' Day
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 26),   # Memorial Day
    date(2025, 6, 19),   # Juneteenth National Independence Day
    date(2025, 7, 4),    # Independence Day
    date(2025, 9, 1),    # Labor Day
    date(2025, 11, 27),  # Thanksgiving Day
    date(2025, 12, 25),  # Christmas Day
]

US_MARKET_HOLIDAYS_2026 = [
    date(2026, 1, 1),    # New Year's Day
    date(2026, 1, 19),   # Martin Luther King Jr. Day
    date(2026, 2, 16),   # Presidents' Day
    date(2026, 4, 3),    # Good Friday
    date(2026, 5, 25),   # Memorial Day
    date(2026, 6, 19),   # Juneteenth National Independence Day
    date(2026, 7, 3),    # Independence Day (observed, 7/4 is Saturday)
    date(2026, 9, 7),    # Labor Day
    date(2026, 11, 26),  # Thanksgiving Day
    date(2026, 12, 25),  # Christmas Day
]

# 半日交易日
US_MARKET_HALF_DAYS_2024 = [
    date(2024, 7, 3),    # July 3rd (Independence Day Eve)
    date(2024, 11, 29),  # Day after Thanksgiving
    date(2024, 12, 24),  # Christmas Eve
]

US_MARKET_HALF_DAYS_2025 = [
    date(2025, 7, 3),    # July 3rd
    date(2025, 11, 28),  # Day after Thanksgiving
    date(2025, 12, 24),  # Christmas Eve
]

US_MARKET_HALF_DAYS_2026 = [
    date(2026, 11, 27),  # Day after Thanksgiving
    date(2026, 12, 24),  # Christmas Eve
]

# 合併假日列表
US_MARKET_HOLIDAYS = (
    US_MARKET_HOLIDAYS_2024 + 
    US_MARKET_HOLIDAYS_2025 + 
    US_MARKET_HOLIDAYS_2026
)

US_MARKET_HALF_DAYS = (
    US_MARKET_HALF_DAYS_2024 + 
    US_MARKET_HALF_DAYS_2025 + 
    US_MARKET_HALF_DAYS_2026
)


# ============================================================
# 假日名稱映射
# ============================================================

HOLIDAY_NAMES = {
    # 2024
    date(2024, 1, 1): "New Year's Day",
    date(2024, 1, 15): "Martin Luther King Jr. Day",
    date(2024, 2, 19): "Presidents' Day",
    date(2024, 3, 29): "Good Friday",
    date(2024, 5, 27): "Memorial Day",
    date(2024, 6, 19): "Juneteenth",
    date(2024, 7, 4): "Independence Day",
    date(2024, 9, 2): "Labor Day",
    date(2024, 11, 28): "Thanksgiving Day",
    date(2024, 12, 25): "Christmas Day",
    # 2025
    date(2025, 1, 1): "New Year's Day",
    date(2025, 1, 20): "Martin Luther King Jr. Day",
    date(2025, 2, 17): "Presidents' Day",
    date(2025, 4, 18): "Good Friday",
    date(2025, 5, 26): "Memorial Day",
    date(2025, 6, 19): "Juneteenth",
    date(2025, 7, 4): "Independence Day",
    date(2025, 9, 1): "Labor Day",
    date(2025, 11, 27): "Thanksgiving Day",
    date(2025, 12, 25): "Christmas Day",
    # 2026
    date(2026, 1, 1): "New Year's Day",
    date(2026, 1, 19): "Martin Luther King Jr. Day",
    date(2026, 2, 16): "Presidents' Day",
    date(2026, 4, 3): "Good Friday",
    date(2026, 5, 25): "Memorial Day",
    date(2026, 6, 19): "Juneteenth",
    date(2026, 7, 3): "Independence Day (observed)",
    date(2026, 9, 7): "Labor Day",
    date(2026, 11, 26): "Thanksgiving Day",
    date(2026, 12, 25): "Christmas Day",
}


# ============================================================
# MarketHours dataclass
# ============================================================

@dataclass
class MarketHours:
    """市場交易時間"""
    
    date: date
    market_type: MarketType
    pre_market_open: Optional[datetime] = None
    market_open: datetime = None
    market_close: datetime = None
    after_hours_close: Optional[datetime] = None
    overnight_open: Optional[datetime] = None
    overnight_close: Optional[datetime] = None
    
    is_holiday: bool = False
    is_half_day: bool = False
    holiday_name: str = ""
    
    @property
    def is_trading_day(self) -> bool:
        """是否為交易日"""
        return not self.is_holiday
    
    @property
    def regular_hours_duration(self) -> timedelta:
        """正常交易時段長度"""
        if self.market_open and self.market_close:
            return self.market_close - self.market_open
        return timedelta(0)
    
    @property
    def extended_hours_duration(self) -> Optional[timedelta]:
        """延長交易時段長度"""
        if self.pre_market_open and self.after_hours_close:
            return self.after_hours_close - self.pre_market_open
        return None
    
    def to_dict(self) -> dict:
        """轉換為字典"""
        return {
            "date": self.date.isoformat(),
            "market_type": self.market_type.value,
            "pre_market_open": self.pre_market_open.isoformat() if self.pre_market_open else None,
            "market_open": self.market_open.isoformat() if self.market_open else None,
            "market_close": self.market_close.isoformat() if self.market_close else None,
            "after_hours_close": self.after_hours_close.isoformat() if self.after_hours_close else None,
            "is_holiday": self.is_holiday,
            "is_half_day": self.is_half_day,
            "holiday_name": self.holiday_name,
            "is_trading_day": self.is_trading_day,
        }


# ============================================================
# MarketCalendar 類
# ============================================================

class MarketCalendar:
    """
    市場日曆
    
    管理不同市場的交易日、假日和交易時間
    
    使用方式:
        calendar = MarketCalendar(MarketType.US_STOCK)
        
        # 檢查是否為交易日
        if calendar.is_trading_day(date.today()):
            print("今天是交易日")
        
        # 取得市場時間
        hours = calendar.get_market_hours(date.today())
        
        # 取得下一個交易日
        next_day = calendar.get_next_trading_day(date.today())
    """
    
    def __init__(
        self,
        market_type: MarketType = MarketType.US_STOCK,
        holidays: Optional[List[date]] = None,
        half_days: Optional[List[date]] = None,
    ):
        """
        初始化市場日曆
        
        Args:
            market_type: 市場類型
            holidays: 假日列表（覆蓋預設）
            half_days: 半日交易日列表（覆蓋預設）
        """
        self._market_type = market_type
        self._config = MARKET_CONFIGS.get(market_type, US_STOCK_CONFIG)
        
        # 設定假日
        if holidays is not None:
            self._holidays = set(holidays)
        else:
            self._holidays = set(US_MARKET_HOLIDAYS)
        
        # 設定半日交易日
        if half_days is not None:
            self._half_days = set(half_days)
        else:
            self._half_days = set(US_MARKET_HALF_DAYS)
    
    @property
    def market_type(self) -> MarketType:
        """市場類型"""
        return self._market_type
    
    @property
    def timezone(self) -> ZoneInfo:
        """市場時區"""
        return self._config.timezone
    
    # ========== 假日檢查 ==========
    
    def is_holiday(self, d: date) -> bool:
        """
        檢查是否為假日
        
        Args:
            d: 日期
            
        Returns:
            是否為假日
        """
        return d in self._holidays
    
    def is_half_day(self, d: date) -> bool:
        """
        檢查是否為半日交易日
        
        Args:
            d: 日期
            
        Returns:
            是否為半日交易日
        """
        return d in self._half_days
    
    def is_weekend(self, d: date) -> bool:
        """
        檢查是否為週末
        
        Args:
            d: 日期
            
        Returns:
            是否為週末
        """
        if self._config.trades_on_weekend:
            return False
        return d.weekday() >= 5  # 5=Saturday, 6=Sunday
    
    def is_trading_day(self, d: date) -> bool:
        """
        檢查是否為交易日
        
        Args:
            d: 日期
            
        Returns:
            是否為交易日
        """
        # 24/7 市場（如加密貨幣）每天都是交易日
        if self._config.is_24_hours and self._config.trades_on_weekend:
            return True
        
        # 外匯市場週末不交易
        if self._config.is_24_hours and not self._config.trades_on_weekend:
            return not self.is_weekend(d)
        
        # 一般市場：排除週末和假日
        return not self.is_weekend(d) and not self.is_holiday(d)
    
    def get_holiday_name(self, d: date) -> str:
        """
        取得假日名稱
        
        Args:
            d: 日期
            
        Returns:
            假日名稱，非假日返回空字串
        """
        return HOLIDAY_NAMES.get(d, "")
    
    # ========== 交易日導航 ==========
    
    def get_next_trading_day(self, d: date) -> date:
        """
        取得下一個交易日
        
        Args:
            d: 起始日期
            
        Returns:
            下一個交易日
        """
        next_day = d + timedelta(days=1)
        
        # 防止無限迴圈
        max_iterations = 30
        iterations = 0
        
        while not self.is_trading_day(next_day) and iterations < max_iterations:
            next_day += timedelta(days=1)
            iterations += 1
        
        return next_day
    
    def get_previous_trading_day(self, d: date) -> date:
        """
        取得上一個交易日
        
        Args:
            d: 起始日期
            
        Returns:
            上一個交易日
        """
        prev_day = d - timedelta(days=1)
        
        # 防止無限迴圈
        max_iterations = 30
        iterations = 0
        
        while not self.is_trading_day(prev_day) and iterations < max_iterations:
            prev_day -= timedelta(days=1)
            iterations += 1
        
        return prev_day
    
    def get_trading_days(
        self,
        start: date,
        end: date,
    ) -> List[date]:
        """
        取得期間內的所有交易日
        
        Args:
            start: 開始日期
            end: 結束日期
            
        Returns:
            交易日列表
        """
        days = []
        current = start
        
        while current <= end:
            if self.is_trading_day(current):
                days.append(current)
            current += timedelta(days=1)
        
        return days
    
    def count_trading_days(
        self,
        start: date,
        end: date,
    ) -> int:
        """
        計算期間內的交易日數
        
        Args:
            start: 開始日期
            end: 結束日期
            
        Returns:
            交易日數
        """
        return len(self.get_trading_days(start, end))
    
    def get_nth_trading_day(
        self,
        start: date,
        n: int,
    ) -> date:
        """
        取得第 N 個交易日
        
        Args:
            start: 起始日期
            n: 交易日數（正數向前，負數向後）
            
        Returns:
            第 N 個交易日
        """
        if n == 0:
            return start if self.is_trading_day(start) else self.get_next_trading_day(start)
        
        current = start
        count = 0
        direction = 1 if n > 0 else -1
        target = abs(n)
        
        while count < target:
            current += timedelta(days=direction)
            if self.is_trading_day(current):
                count += 1
        
        return current
    
    # ========== 市場時間 ==========
    
    def get_market_hours(self, d: Optional[date] = None) -> MarketHours:
        """
        取得指定日期的市場開收盤時間
        
        Args:
            d: 日期，None 表示今天
            
        Returns:
            MarketHours 物件
        """
        if d is None:
            d = self._get_current_date()
        
        is_holiday = self.is_holiday(d) or self.is_weekend(d)
        is_half_day = self.is_half_day(d)
        holiday_name = self.get_holiday_name(d) if is_holiday else ""
        
        tz = self._config.timezone
        
        # 盤前交易
        pre_market_open = None
        if self._config.pre_market_open and self._config.has_extended_hours:
            pre_market_open = datetime.combine(d, self._config.pre_market_open, tzinfo=tz)
        
        # 正常交易
        market_open = datetime.combine(d, self._config.market_open, tzinfo=tz)
        
        if is_half_day:
            market_close = datetime.combine(d, self._config.half_day_close, tzinfo=tz)
        else:
            market_close = datetime.combine(d, self._config.market_close, tzinfo=tz)
        
        # 盤後交易
        after_hours_close = None
        if self._config.after_hours_close and self._config.has_extended_hours:
            if is_half_day:
                # 半日交易日盤後較短
                after_hours_close = datetime.combine(d, time(17, 0), tzinfo=tz)
            else:
                after_hours_close = datetime.combine(d, self._config.after_hours_close, tzinfo=tz)
        
        # 夜盤（期貨）
        overnight_open = None
        overnight_close = None
        if self._config.overnight_open:
            overnight_open = datetime.combine(d, self._config.overnight_open, tzinfo=tz)
        if self._config.overnight_close:
            overnight_close = datetime.combine(d, self._config.overnight_close, tzinfo=tz)
        
        return MarketHours(
            date=d,
            market_type=self._market_type,
            pre_market_open=pre_market_open,
            market_open=market_open,
            market_close=market_close,
            after_hours_close=after_hours_close,
            overnight_open=overnight_open,
            overnight_close=overnight_close,
            is_holiday=is_holiday,
            is_half_day=is_half_day,
            holiday_name=holiday_name,
        )
    
    def get_current_session(
        self,
        dt: Optional[datetime] = None,
    ) -> MarketSession:
        """
        取得當前市場時段
        
        Args:
            dt: datetime，None 表示當前時間
            
        Returns:
            MarketSession
        """
        if dt is None:
            dt = datetime.now(self._config.timezone)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC_TZ)
        
        # 轉換到市場時區
        local_dt = dt.astimezone(self._config.timezone)
        
        # 24 小時市場
        if self._config.is_24_hours:
            if self._config.trades_on_weekend:
                return MarketSession.REGULAR
            else:
                # 外匯市場：週末不交易
                if self.is_weekend(local_dt.date()):
                    return MarketSession.CLOSED
                return MarketSession.REGULAR
        
        hours = self.get_market_hours(local_dt.date())
        
        if hours.is_holiday:
            return MarketSession.CLOSED
        
        current_time = local_dt.time()
        
        # 檢查各時段
        if self._config.pre_market_open:
            if current_time < self._config.pre_market_open:
                return MarketSession.CLOSED
            elif current_time < self._config.market_open:
                return MarketSession.PRE_MARKET
        else:
            if current_time < self._config.market_open:
                return MarketSession.CLOSED
        
        # 正常交易時段
        close_time = self._config.half_day_close if hours.is_half_day else self._config.market_close
        if current_time < close_time:
            return MarketSession.REGULAR
        
        # 盤後
        if self._config.after_hours_close:
            after_close = time(17, 0) if hours.is_half_day else self._config.after_hours_close
            if current_time < after_close:
                return MarketSession.AFTER_HOURS
        
        return MarketSession.CLOSED
    
    # ========== 假日管理 ==========
    
    def add_holiday(self, d: date, name: str = "") -> None:
        """
        添加假日
        
        Args:
            d: 日期
            name: 假日名稱
        """
        self._holidays.add(d)
        if name:
            HOLIDAY_NAMES[d] = name
    
    def remove_holiday(self, d: date) -> None:
        """移除假日"""
        self._holidays.discard(d)
    
    def add_half_day(self, d: date) -> None:
        """添加半日交易日"""
        self._half_days.add(d)
    
    def remove_half_day(self, d: date) -> None:
        """移除半日交易日"""
        self._half_days.discard(d)
    
    def get_holidays_in_range(
        self,
        start: date,
        end: date,
    ) -> List[Tuple[date, str]]:
        """
        取得期間內的假日
        
        Args:
            start: 開始日期
            end: 結束日期
            
        Returns:
            [(日期, 名稱), ...]
        """
        holidays = []
        for d in sorted(self._holidays):
            if start <= d <= end:
                name = HOLIDAY_NAMES.get(d, "Holiday")
                holidays.append((d, name))
        return holidays
    
    # ========== 工具方法 ==========
    
    def _get_current_date(self) -> date:
        """取得當前日期（市場時區）"""
        return datetime.now(self._config.timezone).date()


# ============================================================
# 全局實例和便捷函數
# ============================================================

# 全局日曆實例
_calendars: Dict[MarketType, MarketCalendar] = {}


def get_calendar(market_type: MarketType = MarketType.US_STOCK) -> MarketCalendar:
    """
    取得市場日曆實例
    
    Args:
        market_type: 市場類型
        
    Returns:
        MarketCalendar 實例
    """
    if market_type not in _calendars:
        _calendars[market_type] = MarketCalendar(market_type)
    return _calendars[market_type]


# ============================================================
# 時間工具函數
# ============================================================

def get_eastern_time(dt: Optional[datetime] = None) -> datetime:
    """
    取得美東時間
    
    Args:
        dt: datetime，None 表示當前時間
        
    Returns:
        美東時間的 datetime
    """
    if dt is None:
        return datetime.now(EASTERN_TZ)
    elif dt.tzinfo is None:
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


# ============================================================
# 便捷函數（美股市場）
# ============================================================

def is_trading_day(
    d: Optional[date] = None,
    market_type: MarketType = MarketType.US_STOCK,
) -> bool:
    """
    檢查是否為交易日
    
    Args:
        d: 日期，None 表示今天
        market_type: 市場類型
        
    Returns:
        是否為交易日
    """
    calendar = get_calendar(market_type)
    if d is None:
        d = datetime.now(calendar.timezone).date()
    return calendar.is_trading_day(d)


def get_next_trading_day(
    d: Optional[date] = None,
    market_type: MarketType = MarketType.US_STOCK,
) -> date:
    """
    取得下一個交易日
    
    Args:
        d: 起始日期，None 表示今天
        market_type: 市場類型
        
    Returns:
        下一個交易日
    """
    calendar = get_calendar(market_type)
    if d is None:
        d = datetime.now(calendar.timezone).date()
    return calendar.get_next_trading_day(d)


def get_market_hours(
    d: Optional[date] = None,
    market_type: MarketType = MarketType.US_STOCK,
) -> MarketHours:
    """
    取得市場開收盤時間
    
    Args:
        d: 日期，None 表示今天
        market_type: 市場類型
        
    Returns:
        MarketHours 物件
    """
    calendar = get_calendar(market_type)
    return calendar.get_market_hours(d)


def is_market_open(
    dt: Optional[datetime] = None,
    include_extended: bool = False,
    market_type: MarketType = MarketType.US_STOCK,
) -> bool:
    """
    檢查是否在交易時段
    
    Args:
        dt: datetime，None 表示當前時間
        include_extended: 是否包含盤前盤後
        market_type: 市場類型
        
    Returns:
        是否在交易時段
    """
    calendar = get_calendar(market_type)
    session = calendar.get_current_session(dt)
    
    if include_extended:
        return session in [
            MarketSession.PRE_MARKET,
            MarketSession.REGULAR,
            MarketSession.AFTER_HOURS,
        ]
    return session == MarketSession.REGULAR


def time_until_market_open(
    dt: Optional[datetime] = None,
    market_type: MarketType = MarketType.US_STOCK,
) -> timedelta:
    """
    計算距離開盤的時間差
    
    Args:
        dt: datetime，None 表示當前時間
        market_type: 市場類型
        
    Returns:
        timedelta
    """
    calendar = get_calendar(market_type)
    config = MARKET_CONFIGS[market_type]
    
    if dt is None:
        dt = datetime.now(config.timezone)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC_TZ)
    
    local_dt = dt.astimezone(config.timezone)
    hours = calendar.get_market_hours(local_dt.date())
    
    # 今天是交易日且還沒開盤
    if hours.is_trading_day and local_dt < hours.market_open:
        return hours.market_open - local_dt
    
    # 今天已開盤
    if hours.is_trading_day and local_dt < hours.market_close:
        return timedelta(0)
    
    # 找下一個交易日
    next_day = calendar.get_next_trading_day(local_dt.date())
    next_hours = calendar.get_market_hours(next_day)
    
    return next_hours.market_open - local_dt


def get_next_market_open(
    dt: Optional[datetime] = None,
    market_type: MarketType = MarketType.US_STOCK,
) -> datetime:
    """
    取得下一次開盤時間
    
    Args:
        dt: datetime，None 表示當前時間
        market_type: 市場類型
        
    Returns:
        下一次開盤的 datetime
    """
    calendar = get_calendar(market_type)
    config = MARKET_CONFIGS[market_type]
    
    if dt is None:
        dt = datetime.now(config.timezone)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC_TZ)
    
    local_dt = dt.astimezone(config.timezone)
    hours = calendar.get_market_hours(local_dt.date())
    
    # 今天是交易日且還沒開盤
    if hours.is_trading_day and local_dt < hours.market_open:
        return hours.market_open
    
    # 找下一個交易日
    next_day = calendar.get_next_trading_day(local_dt.date())
    next_hours = calendar.get_market_hours(next_day)
    
    return next_hours.market_open


def get_next_market_close(
    dt: Optional[datetime] = None,
    market_type: MarketType = MarketType.US_STOCK,
) -> datetime:
    """
    取得下一次收盤時間
    
    Args:
        dt: datetime，None 表示當前時間
        market_type: 市場類型
        
    Returns:
        下一次收盤的 datetime
    """
    calendar = get_calendar(market_type)
    config = MARKET_CONFIGS[market_type]
    
    if dt is None:
        dt = datetime.now(config.timezone)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC_TZ)
    
    local_dt = dt.astimezone(config.timezone)
    hours = calendar.get_market_hours(local_dt.date())
    
    # 今天是交易日且還沒收盤
    if hours.is_trading_day and local_dt < hours.market_close:
        return hours.market_close
    
    # 找下一個交易日
    next_day = calendar.get_next_trading_day(local_dt.date())
    next_hours = calendar.get_market_hours(next_day)
    
    return next_hours.market_close


def format_duration(td: timedelta) -> str:
    """
    格式化時間差
    
    Args:
        td: timedelta
        
    Returns:
        格式化的字串，如 "2h 30m 15s"
    """
    total_seconds = int(td.total_seconds())
    
    if total_seconds < 0:
        return "0"
    
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 and hours == 0:
        parts.append(f"{seconds}s")
    
    return " ".join(parts) if parts else "0s"


def get_market_status(
    market_type: MarketType = MarketType.US_STOCK,
) -> dict:
    """
    取得當前市場狀態摘要
    
    Args:
        market_type: 市場類型
        
    Returns:
        狀態字典
    """
    calendar = get_calendar(market_type)
    config = MARKET_CONFIGS[market_type]
    
    now = datetime.now(config.timezone)
    session = calendar.get_current_session(now)
    hours = calendar.get_market_hours(now.date())
    
    status = {
        "market_type": market_type.value,
        "local_time": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "session": session.name,
        "is_trading_day": hours.is_trading_day,
        "is_half_day": hours.is_half_day,
    }
    
    if hours.holiday_name:
        status["holiday_name"] = hours.holiday_name
    
    if session == MarketSession.CLOSED:
        time_to_open = time_until_market_open(now, market_type)
        next_day = calendar.get_next_trading_day(now.date())
        next_hours = calendar.get_market_hours(next_day)
        status["next_open"] = next_hours.market_open.strftime("%Y-%m-%d %H:%M %Z")
        status["time_until_open"] = format_duration(time_to_open)
    elif session == MarketSession.REGULAR:
        time_to_close = hours.market_close - now
        status["market_close"] = hours.market_close.strftime("%H:%M %Z")
        status["time_until_close"] = format_duration(time_to_close)
    
    return status


# ============================================================
# 台灣時間轉換
# ============================================================

def get_us_market_hours_in_taiwan_time(
    d: Optional[date] = None,
) -> dict:
    """
    取得美股市場時間（台灣時間）
    
    美股交易時間：
    - 夏令時間（3月-11月）：台灣時間 21:30 - 04:00+1
    - 冬令時間（11月-3月）：台灣時間 22:30 - 05:00+1
    
    Args:
        d: 日期，None 表示今天
        
    Returns:
        包含台灣時間的字典
    """
    hours = get_market_hours(d, MarketType.US_STOCK)
    
    result = {
        "date_us": hours.date.isoformat(),
        "is_trading_day": hours.is_trading_day,
        "is_half_day": hours.is_half_day,
    }
    
    if hours.pre_market_open:
        result["pre_market_open_tw"] = hours.pre_market_open.astimezone(TAIWAN_TZ).strftime("%H:%M")
    
    if hours.market_open:
        result["market_open_tw"] = hours.market_open.astimezone(TAIWAN_TZ).strftime("%H:%M")
    
    if hours.market_close:
        result["market_close_tw"] = hours.market_close.astimezone(TAIWAN_TZ).strftime("%H:%M")
    
    if hours.after_hours_close:
        result["after_hours_close_tw"] = hours.after_hours_close.astimezone(TAIWAN_TZ).strftime("%H:%M")
    
    return result