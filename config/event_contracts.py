"""Event Contracts 配置 - 2025 & 2026"""
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, date

# 2025 FOMC 會議（決議日）
FOMC_DATES_2025 = [
    "20250129", "20250319", "20250507", "20250618",
    "20250730", "20250917", "20251029", "20251210",
]

# 2026 FOMC 會議（決議日）
FOMC_DATES_2026 = [
    "20260128",  # 1月28日
    "20260318",  # 3月18日
    "20260429",  # 4月29日
    "20260617",  # 6月17日
    "20260729",  # 7月29日
    "20260916",  # 9月16日
    "20261028",  # 10月28日
    "20261209",  # 12月9日
]

# 合併所有 FOMC 日期
FOMC_DATES = FOMC_DATES_2025 + FOMC_DATES_2026

# CPI 發布日期 2025
CPI_DATES_2025 = [
    "20250115", "20250212", "20250312", "20250410",
    "20250513", "20250611", "20250710", "20250813",
    "20250910", "20251015", "20251112", "20251210",
]

# NFP 發布日期 2025
NFP_DATES_2025 = [
    "20250110", "20250207", "20250307", "20250404",
    "20250502", "20250606", "20250703", "20250801",
    "20250905", "20251003", "20251107", "20251205",
]

@dataclass
class ForecastExMarket:
    symbol: str
    name: str
    description: str
    tick_size: float = 0.01

FORECASTEX_MARKETS = {
    "FF": ForecastExMarket("FF", "Fed Funds Rate", "Federal Funds Target Rate"),
    "CPI": ForecastExMarket("CPI", "Consumer Price Index", "CPI YoY change"),
    "UNRATE": ForecastExMarket("UNRATE", "Unemployment Rate", "U.S. Unemployment Rate"),
    "GDP": ForecastExMarket("GDP", "Gross Domestic Product", "Real GDP QoQ"),
}

FED_FUNDS_STRIKES = [
    3.875, 4.000, 4.125, 4.250, 4.375, 4.500, 4.625, 4.750, 4.875, 5.000,
]

CURRENT_FED_FUNDS_RATE = 4.375

def get_next_fomc_date(after=None):
    """取得下一個 FOMC 日期"""
    if after is None:
        after = date.today()
    for d in FOMC_DATES:
        if datetime.strptime(d, "%Y%m%d").date() > after:
            return d
    return None

def get_next_cpi_date(after=None):
    if after is None:
        after = date.today()
    for d in CPI_DATES_2025:
        if datetime.strptime(d, "%Y%m%d").date() > after:
            return d
    return None

def days_until_fomc(fomc_date):
    """計算距離 FOMC 的天數"""
    return (datetime.strptime(fomc_date, "%Y%m%d").date() - date.today()).days

def print_economic_calendar():
    """打印經濟日曆"""
    print("\n" + "=" * 50)
    print("📅 FOMC 會議日曆")
    print("=" * 50)
    
    print("\n🏛️ 2025 FOMC:")
    for d in FOMC_DATES_2025:
        dt = datetime.strptime(d, "%Y%m%d")
        print(f"  {dt.strftime('%Y-%m-%d (%a)')}")
    
    print("\n🏛️ 2026 FOMC:")
    for d in FOMC_DATES_2026:
        dt = datetime.strptime(d, "%Y%m%d")
        print(f"  {dt.strftime('%Y-%m-%d (%a)')}")
    
    next_fomc = get_next_fomc_date()
    if next_fomc:
        dt = datetime.strptime(next_fomc, "%Y%m%d")
        days = days_until_fomc(next_fomc)
        print(f"\n⏰ 下一個 FOMC: {dt.strftime('%Y-%m-%d')} ({days} 天後)")
    
    print("=" * 50 + "\n")
