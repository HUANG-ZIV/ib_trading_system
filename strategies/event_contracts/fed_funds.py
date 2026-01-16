"""Fed Funds Rate 策略"""
import logging
from dataclasses import dataclass
from typing import Optional, Dict, List
from ib_insync import IB
from .base import EventContractStrategy, EventSignal
from config.event_contracts import FOMC_DATES_2025, CURRENT_FED_FUNDS_RATE, get_next_fomc_date, days_until_fomc

logger = logging.getLogger(__name__)

@dataclass
class RatePrediction:
    cut_prob: float
    hold_prob: float
    hike_prob: float

class FedFundsStrategy(EventContractStrategy):
    def __init__(self, ib: IB, current_rate: float = CURRENT_FED_FUNDS_RATE,
                 min_edge: float = 0.08, max_position: int = 50):
        super().__init__(ib, "FF", max_position)
        self._current_rate = current_rate
        self._min_edge = min_edge
        self._predictions: Dict[str, RatePrediction] = {}
    
    def set_rate_prediction(self, expiry: str, cut: float, hold: float, hike: float):
        self._predictions[expiry] = RatePrediction(cut, hold, hike)
        logger.info(f"預測 {expiry}: 降息={cut:.0%}, 維持={hold:.0%}, 升息={hike:.0%}")
    
    def generate_signals(self) -> List[EventSignal]:
        signals = []
        next_fomc = get_next_fomc_date()
        if not next_fomc or next_fomc not in self._predictions:
            return signals
        days = days_until_fomc(next_fomc)
        if days <= 0 or days > 14:
            return signals
        # 這裡需要連接 IB 獲取實際報價來生成信號
        logger.info(f"分析 FOMC {next_fomc} (還有 {days} 天)")
        return signals
    
    def print_status(self):
        print(f"\n{'='*50}")
        print(f"📊 Fed Funds 策略狀態")
        print(f"當前利率: {self._current_rate}%")
        print(f"最小 Edge: {self._min_edge:.0%}")
        next_fomc = get_next_fomc_date()
        if next_fomc:
            print(f"下一個 FOMC: {next_fomc} ({days_until_fomc(next_fomc)} 天後)")
        print(f"{'='*50}\n")
