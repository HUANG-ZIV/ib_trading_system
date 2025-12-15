"""
Symbols 模組 - 交易標的定義

定義各種證券類型和交易標的配置
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict
from datetime import datetime


class SecurityType(Enum):
    """證券類型枚舉"""
    
    STOCK = "STK"       # 股票
    FUTURE = "FUT"      # 期貨
    OPTION = "OPT"      # 期權
    FOREX = "CASH"      # 外匯
    INDEX = "IND"       # 指數
    CFD = "CFD"         # 差價合約
    COMMODITY = "CMDTY" # 商品


class OptionRight(Enum):
    """期權類型"""
    
    CALL = "C"
    PUT = "P"


@dataclass
class SymbolConfig:
    """
    交易標的配置
    
    統一的標的定義，用於建立 IB 合約
    """
    
    # ========== 基本資訊 ==========
    symbol: str                              # 標的代碼
    security_type: SecurityType              # 證券類型
    exchange: str = "SMART"                  # 交易所
    currency: str = "USD"                    # 貨幣
    
    # ========== 期貨/期權專用 ==========
    expiry: Optional[str] = None             # 到期日 (YYYYMMDD)
    multiplier: Optional[str] = None         # 合約乘數
    
    # ========== 期權專用 ==========
    strike: Optional[float] = None           # 行權價
    right: Optional[OptionRight] = None      # 期權類型 (CALL/PUT)
    
    # ========== 外匯專用 ==========
    # 外匯 symbol 為貨幣對，如 "EUR" (對 USD)
    
    # ========== 自訂設定 ==========
    primary_exchange: Optional[str] = None   # 主要交易所
    local_symbol: Optional[str] = None       # 本地代碼
    trading_class: Optional[str] = None      # 交易類別
    
    # ========== 元數據 ==========
    description: str = ""                    # 描述
    tick_size: Optional[float] = None        # 最小跳動
    min_qty: int = 1                         # 最小交易數量
    
    def __post_init__(self):
        """驗證配置"""
        if self.security_type == SecurityType.FUTURE and not self.expiry:
            # 期貨可以不指定到期日（使用連續合約）
            pass
        
        if self.security_type == SecurityType.OPTION:
            if not all([self.expiry, self.strike, self.right]):
                raise ValueError("Option requires expiry, strike, and right")
    
    def to_dict(self) -> Dict:
        """轉換為字典（用於建立 IB 合約）"""
        result = {
            "symbol": self.symbol,
            "secType": self.security_type.value,
            "exchange": self.exchange,
            "currency": self.currency,
        }
        
        if self.expiry:
            result["lastTradeDateOrContractMonth"] = self.expiry
        if self.multiplier:
            result["multiplier"] = self.multiplier
        if self.strike:
            result["strike"] = self.strike
        if self.right:
            result["right"] = self.right.value
        if self.primary_exchange:
            result["primaryExchange"] = self.primary_exchange
        if self.local_symbol:
            result["localSymbol"] = self.local_symbol
        if self.trading_class:
            result["tradingClass"] = self.trading_class
            
        return result


# ============================================================
# 工廠函數
# ============================================================

def create_stock(
    symbol: str,
    exchange: str = "SMART",
    currency: str = "USD",
    primary_exchange: Optional[str] = None,
    **kwargs
) -> SymbolConfig:
    """
    建立股票標的配置
    
    Args:
        symbol: 股票代碼
        exchange: 交易所 (預設 SMART)
        currency: 貨幣 (預設 USD)
        primary_exchange: 主要交易所
        
    Returns:
        SymbolConfig 實例
        
    Example:
        >>> apple = create_stock("AAPL")
        >>> tsmc = create_stock("TSM", primary_exchange="NYSE")
    """
    return SymbolConfig(
        symbol=symbol,
        security_type=SecurityType.STOCK,
        exchange=exchange,
        currency=currency,
        primary_exchange=primary_exchange,
        **kwargs
    )


def create_future(
    symbol: str,
    expiry: Optional[str] = None,
    exchange: str = "CME",
    currency: str = "USD",
    multiplier: Optional[str] = None,
    **kwargs
) -> SymbolConfig:
    """
    建立期貨標的配置
    
    Args:
        symbol: 期貨代碼 (如 ES, NQ, CL)
        expiry: 到期月份 (YYYYMM) 或到期日 (YYYYMMDD)
        exchange: 交易所
        currency: 貨幣
        multiplier: 合約乘數
        
    Returns:
        SymbolConfig 實例
        
    Example:
        >>> es = create_future("ES", expiry="202412", multiplier="50")
        >>> cl = create_future("CL", exchange="NYMEX")
    """
    return SymbolConfig(
        symbol=symbol,
        security_type=SecurityType.FUTURE,
        exchange=exchange,
        currency=currency,
        expiry=expiry,
        multiplier=multiplier,
        **kwargs
    )


def create_option(
    symbol: str,
    expiry: str,
    strike: float,
    right: OptionRight,
    exchange: str = "SMART",
    currency: str = "USD",
    multiplier: str = "100",
    **kwargs
) -> SymbolConfig:
    """
    建立期權標的配置
    
    Args:
        symbol: 標的代碼
        expiry: 到期日 (YYYYMMDD)
        strike: 行權價
        right: CALL 或 PUT
        exchange: 交易所
        currency: 貨幣
        multiplier: 合約乘數 (預設 100)
        
    Returns:
        SymbolConfig 實例
        
    Example:
        >>> call = create_option("AAPL", "20241220", 200.0, OptionRight.CALL)
        >>> put = create_option("SPY", "20241220", 450.0, OptionRight.PUT)
    """
    return SymbolConfig(
        symbol=symbol,
        security_type=SecurityType.OPTION,
        exchange=exchange,
        currency=currency,
        expiry=expiry,
        strike=strike,
        right=right,
        multiplier=multiplier,
        **kwargs
    )


def create_forex(
    symbol: str,
    currency: str = "USD",
    exchange: str = "IDEALPRO",
    **kwargs
) -> SymbolConfig:
    """
    建立外匯標的配置
    
    Args:
        symbol: 貨幣代碼 (如 EUR, GBP, JPY)
        currency: 報價貨幣 (預設 USD)
        exchange: 交易所 (預設 IDEALPRO)
        
    Returns:
        SymbolConfig 實例
        
    Example:
        >>> eurusd = create_forex("EUR")  # EUR/USD
        >>> gbpusd = create_forex("GBP")  # GBP/USD
    """
    return SymbolConfig(
        symbol=symbol,
        security_type=SecurityType.FOREX,
        exchange=exchange,
        currency=currency,
        **kwargs
    )


def create_index(
    symbol: str,
    exchange: str = "CBOE",
    currency: str = "USD",
    **kwargs
) -> SymbolConfig:
    """
    建立指數標的配置
    
    Args:
        symbol: 指數代碼 (如 SPX, VIX)
        exchange: 交易所
        currency: 貨幣
        
    Returns:
        SymbolConfig 實例
    """
    return SymbolConfig(
        symbol=symbol,
        security_type=SecurityType.INDEX,
        exchange=exchange,
        currency=currency,
        **kwargs
    )


# ============================================================
# 預設標的列表
# ============================================================

# 美股 - 科技股
US_TECH_STOCKS: List[SymbolConfig] = [
    create_stock("AAPL", primary_exchange="NASDAQ", description="Apple Inc."),
    create_stock("MSFT", primary_exchange="NASDAQ", description="Microsoft Corp."),
    create_stock("GOOGL", primary_exchange="NASDAQ", description="Alphabet Inc."),
    create_stock("AMZN", primary_exchange="NASDAQ", description="Amazon.com Inc."),
    create_stock("META", primary_exchange="NASDAQ", description="Meta Platforms Inc."),
    create_stock("NVDA", primary_exchange="NASDAQ", description="NVIDIA Corp."),
    create_stock("TSLA", primary_exchange="NASDAQ", description="Tesla Inc."),
]

# 美股 - ETF
US_ETFS: List[SymbolConfig] = [
    create_stock("SPY", primary_exchange="ARCA", description="S&P 500 ETF"),
    create_stock("QQQ", primary_exchange="NASDAQ", description="NASDAQ 100 ETF"),
    create_stock("IWM", primary_exchange="ARCA", description="Russell 2000 ETF"),
    create_stock("DIA", primary_exchange="ARCA", description="Dow Jones ETF"),
    create_stock("TLT", primary_exchange="NASDAQ", description="20+ Year Treasury Bond ETF"),
    create_stock("GLD", primary_exchange="ARCA", description="Gold ETF"),
]

# 美股 - 綜合列表
US_STOCKS: List[SymbolConfig] = US_TECH_STOCKS + US_ETFS

# 美國期貨
US_FUTURES: List[SymbolConfig] = [
    create_future("ES", exchange="CME", multiplier="50", description="E-mini S&P 500"),
    create_future("NQ", exchange="CME", multiplier="20", description="E-mini NASDAQ 100"),
    create_future("YM", exchange="CBOT", multiplier="5", description="E-mini Dow"),
    create_future("RTY", exchange="CME", multiplier="50", description="E-mini Russell 2000"),
    create_future("CL", exchange="NYMEX", multiplier="1000", description="Crude Oil"),
    create_future("GC", exchange="COMEX", multiplier="100", description="Gold"),
    create_future("SI", exchange="COMEX", multiplier="5000", description="Silver"),
    create_future("ZB", exchange="CBOT", multiplier="1000", description="30-Year T-Bond"),
    create_future("ZN", exchange="CBOT", multiplier="1000", description="10-Year T-Note"),
]

# 外匯
# 外匯對的命名慣例：第一個貨幣是基準貨幣，第二個是報價貨幣
# EUR/USD = 1 歐元值多少美元
FOREX_PAIRS: List[SymbolConfig] = [
    create_forex("EUR", currency="USD", description="EUR/USD"),
    create_forex("GBP", currency="USD", description="GBP/USD"),
    create_forex("USD", currency="JPY", description="USD/JPY"),  # USD 是基準貨幣
    create_forex("AUD", currency="USD", description="AUD/USD"),
    create_forex("USD", currency="CHF", description="USD/CHF"),  # USD 是基準貨幣
]


# ============================================================
# 輔助函數
# ============================================================

def get_watchlist(name: str) -> List[SymbolConfig]:
    """
    取得預設的觀察清單
    
    Args:
        name: 清單名稱 (us_stocks, us_tech, us_etfs, us_futures, forex)
        
    Returns:
        SymbolConfig 列表
    """
    watchlists = {
        "us_stocks": US_STOCKS,
        "us_tech": US_TECH_STOCKS,
        "us_etfs": US_ETFS,
        "us_futures": US_FUTURES,
        "forex": FOREX_PAIRS,
    }
    
    name = name.lower()
    if name not in watchlists:
        available = ", ".join(watchlists.keys())
        raise ValueError(f"Unknown watchlist: {name}. Available: {available}")
    
    return watchlists[name]


def find_symbol(symbol: str, watchlist: Optional[List[SymbolConfig]] = None) -> Optional[SymbolConfig]:
    """
    在清單中尋找標的
    
    Args:
        symbol: 標的代碼
        watchlist: 搜尋的清單（預設搜尋所有）
        
    Returns:
        找到的 SymbolConfig 或 None
    """
    if watchlist is None:
        watchlist = US_STOCKS + US_FUTURES + FOREX_PAIRS
    
    for config in watchlist:
        if config.symbol.upper() == symbol.upper():
            return config
    
    return None


def get_next_expiry(symbol: str, months_ahead: int = 1) -> str:
    """
    取得期貨的下一個到期月份
    
    Args:
        symbol: 期貨代碼
        months_ahead: 幾個月後 (1 = 近月)
        
    Returns:
        到期月份字串 (YYYYMM)
    """
    now = datetime.now()
    year = now.year
    month = now.month + months_ahead
    
    while month > 12:
        month -= 12
        year += 1
    
    return f"{year}{month:02d}"

def create_commodity(
    symbol: str,
    exchange: str = "SMART",
    currency: str = "USD",
    **kwargs
) -> SymbolConfig:
    """
    建立商品標的配置
    
    Args:
        symbol: 商品代碼 (如 XAUUSD, XAGUSD)
        exchange: 交易所 (預設 SMART)
        currency: 貨幣 (預設 USD)
        
    Returns:
        SymbolConfig 實例
        
    Example:
        >>> gold = create_commodity("XAUUSD")  # 黃金
        >>> silver = create_commodity("XAGUSD")  # 白銀
    """
    return SymbolConfig(
        symbol=symbol,
        security_type=SecurityType.COMMODITY,
        exchange=exchange,
        currency=currency,
        **kwargs
    )


# 貴金屬商品
COMMODITIES: List[SymbolConfig] = [
    create_commodity("XAUUSD", description="London Gold"),
    create_commodity("XAGUSD", description="London Silver"),
]
