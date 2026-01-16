"""FOMC 歷史數據 (2022-2025)"""
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime

@dataclass
class FOMCDecision:
    date: str
    rate_before: float
    rate_after: float
    change_bps: int
    action: str
    vote: str
    notes: str = ""

FOMC_2022 = [
    FOMCDecision("20220126", 0.25, 0.25, 0, "HOLD", "12-0"),
    FOMCDecision("20220316", 0.25, 0.50, 25, "HIKE", "8-1"),
    FOMCDecision("20220504", 0.50, 1.00, 50, "HIKE", "10-0"),
    FOMCDecision("20220615", 1.00, 1.75, 75, "HIKE", "10-0"),
    FOMCDecision("20220727", 1.75, 2.50, 75, "HIKE", "12-0"),
    FOMCDecision("20220921", 2.50, 3.25, 75, "HIKE", "12-0"),
    FOMCDecision("20221102", 3.25, 4.00, 75, "HIKE", "12-0"),
    FOMCDecision("20221214", 4.00, 4.50, 50, "HIKE", "12-0"),
]

FOMC_2023 = [
    FOMCDecision("20230201", 4.50, 4.75, 25, "HIKE", "12-0"),
    FOMCDecision("20230322", 4.75, 5.00, 25, "HIKE", "11-0"),
    FOMCDecision("20230503", 5.00, 5.25, 25, "HIKE", "10-0"),
    FOMCDecision("20230614", 5.25, 5.25, 0, "HOLD", "11-0"),
    FOMCDecision("20230726", 5.25, 5.50, 25, "HIKE", "11-0"),
    FOMCDecision("20230920", 5.50, 5.50, 0, "HOLD", "12-0"),
    FOMCDecision("20231101", 5.50, 5.50, 0, "HOLD", "12-0"),
    FOMCDecision("20231213", 5.50, 5.50, 0, "HOLD", "12-0"),
]

FOMC_2024 = [
    FOMCDecision("20240131", 5.50, 5.50, 0, "HOLD", "12-0"),
    FOMCDecision("20240320", 5.50, 5.50, 0, "HOLD", "11-0"),
    FOMCDecision("20240501", 5.50, 5.50, 0, "HOLD", "11-0"),
    FOMCDecision("20240612", 5.50, 5.50, 0, "HOLD", "11-0"),
    FOMCDecision("20240731", 5.50, 5.50, 0, "HOLD", "12-0"),
    FOMCDecision("20240918", 5.50, 5.00, -50, "CUT", "11-1"),
    FOMCDecision("20241107", 5.00, 4.75, -25, "CUT", "12-0"),
    FOMCDecision("20241218", 4.75, 4.50, -25, "CUT", "11-1"),
]

FOMC_2025 = [
    FOMCDecision("20250129", 4.50, 4.50, 0, "HOLD", "12-0"),
    FOMCDecision("20250319", 4.50, 4.50, 0, "HOLD", "11-0"),
    FOMCDecision("20250507", 4.50, 4.50, 0, "HOLD", "12-0"),
    FOMCDecision("20250618", 4.50, 4.25, -25, "CUT", "10-2"),
    FOMCDecision("20250730", 4.25, 4.25, 0, "HOLD", "11-1"),
    FOMCDecision("20250917", 4.25, 4.25, 0, "HOLD", "12-0"),
    FOMCDecision("20251029", 4.25, 4.25, 0, "HOLD", "12-0"),
    FOMCDecision("20251210", 4.25, 4.00, -25, "CUT", "11-1"),
]

FOMC_HISTORY = FOMC_2022 + FOMC_2023 + FOMC_2024 + FOMC_2025

def get_fomc_decision(date: str):
    for d in FOMC_HISTORY:
        if d.date == date:
            return d
    return None

def calculate_settlement(strike: float, actual_rate: float, is_yes: bool) -> float:
    yes_wins = actual_rate >= strike
    if is_yes:
        return 1.0 if yes_wins else 0.0
    return 0.0 if yes_wins else 1.0

def get_action_statistics() -> Dict:
    stats = {"HIKE": 0, "CUT": 0, "HOLD": 0}
    for d in FOMC_HISTORY:
        stats[d.action] += 1
    total = len(FOMC_HISTORY)
    return {
        "total": total,
        "hikes": stats["HIKE"], "cuts": stats["CUT"], "holds": stats["HOLD"],
        "hike_pct": stats["HIKE"]/total*100,
        "cut_pct": stats["CUT"]/total*100,
        "hold_pct": stats["HOLD"]/total*100,
    }

def print_fomc_history(year=None):
    data = [d for d in FOMC_HISTORY if d.date.startswith(str(year))] if year else FOMC_HISTORY
    print(f"\n{'='*60}\n📊 FOMC 歷史" + (f" ({year})" if year else "") + f"\n{'='*60}")
    for d in data:
        chg = f"{d.change_bps:+d}bp" if d.change_bps else "0"
        print(f"{d.date}  {d.rate_before:.2f} → {d.rate_after:.2f}  {chg:>6}  {d.action}")
    print("="*60)
