"""
Event Contracts 配置模組

定義 ForecastEx 和 CME Event Contract 相關配置
包含經濟日曆、市場定義等
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, date


# FOMC 會議日期 (Fed Funds Rate 決議日)
FOMC_DATES_2025: List[str] = [
    "20250129", "20250319", "20250507", "20250618",
    "20250730", "20250917", "20251105", "20251217",
]

# CPI 發布日期
CPI_DATES_2025: List[str] = [
    "20250115", "20250212", "20250312", "20250410",
    "20250513", "20250611", "20250710", "20250813",
    "20250910", "20251015", "20251112", "20251210",
]

# NFP/失業率發布日期
NFP_DATES_2025: List[str] = [
    "20250110", "20250207", "20250307", "20250404",
    "20250502", "20250606", "20250703", "20250801",
    "20250905", "20251003", "20251107", "20251205",
]


@dataclass
class ForecastExMarket:
    """ForecastEx 市場定義"""
    symbol: str
    name: str
    description: str
    tick_size: float = 0.01


FORECASTEX_MARKETS: Dict[str, ForecastExMarket] = {
    "FF": ForecastExMarket("FF", "Fed Funds Rate", "Federal Funds Target Rate"),
    "CPI": ForecastExMarket("CPI", "Consumer Price Index", "CPI YoY change"),
    "UNRATE": ForecastExMarket("UNRATE", "Unemployment Rate", "U.S. Unemployment Rate"),
    "GDP": ForecastExMarket("GDP", "Gross Domestic Product", "Real GDP QoQ"),
}

FED_FUNDS_STRIKES: List[float] = [
    3.875, 4.000, 4.125, 4.250, 4.375, 4.500, 4.625, 4.750, 4.875, 5.000,
]

CURRENT_FED_FUNDS_RATE: float = 4.375


def get_next_fomc_date(after: Optional[date] = None) -> Optional[str]:
    if after is None:
        after = date.today()
    for fomc_date in FOMC_DATES_2025:
        fomc = datetime.strptime(fomc_date, "%Y%m%d").date()
        if fomc > after:
            return fomc_date
    return None


def get_next_cpi_date(after: Optional[date] = None) -> Optional[str]:
    if after is None:
        after = date.today()
    for cpi_date in CPI_DATES_2025:
        cpi = datetime.strptime(cpi_date, "%Y%m%d").date()
        if cpi > after:
            return cpi_date
    return None


def days_until_fomc(fomc_date: str) -> int:
    fomc = datetime.strptime(fomc_date, "%Y%m%d").date()
    return (fomc - date.today()).days


def print_economic_calendar():
    print("\n" + "=" * 50)
    print("📅 2025 經濟日曆")
    print("=" * 50)
    print("\n🏛️ FOMC 會議:")
    for d in FOMC_DATES_2025:
        dt = datetime.strptime(d, "%Y%m%d")
        print(f"  {dt.strftime('%Y-%m-%d (%a)')}")
    next_fomc = get_next_fomc_date()
    if next_fomc:
        print(f"\n⏰ 下一個 FOMC: {next_fomc} ({days_until_fomc(next_fomc)} 天後)")
    print("=" * 50 + "\n")
