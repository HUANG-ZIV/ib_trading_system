"""Event Contract 回測引擎"""
import logging
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from enum import Enum

try:
    from .fomc_history import FOMC_HISTORY, FOMCDecision, calculate_settlement
except:
    from fomc_history import FOMC_HISTORY, FOMCDecision, calculate_settlement

logger = logging.getLogger(__name__)

@dataclass
class BacktestTrade:
    date: str; fomc_date: str; strike: float; side: str
    quantity: int; entry_price: float; settlement: float
    pnl: float; is_winner: bool; edge: float = 0.0; reason: str = ""

@dataclass
class BacktestResult:
    start_date: str; end_date: str; initial_capital: float; final_capital: float
    total_return: float; total_return_pct: float; num_trades: int
    num_winners: int; num_losers: int; win_rate: float
    avg_win: float; avg_loss: float; profit_factor: float
    max_drawdown: float; max_drawdown_pct: float; sharpe_ratio: float
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)

@dataclass
class MarketSimulation:
    base_hold_prob: float = 0.70
    surprise_factor: float = 0.10
    spread: float = 0.02
    slippage: float = 0.01

class EventContractBacktest:
    def __init__(self, initial_capital=10000, start_date="20230101", 
                 end_date="20251231", max_position_per_trade=10, market_sim=None):
        self._initial_capital = initial_capital
        self._start_date = start_date
        self._end_date = end_date
        self._max_position = max_position_per_trade
        self._market_sim = market_sim or MarketSimulation()
        self._capital = initial_capital
        self._trades = []
        self._equity_curve = [initial_capital]
        self._fomc_dates = [d for d in FOMC_HISTORY if start_date <= d.date <= end_date]

    def simulate_market_price(self, fomc, strike, days_before=7):
        sim = self._market_sim
        actual_rate = fomc.rate_after
        yes_wins = actual_rate >= strike
        if yes_wins:
            base_yes_prob = 0.55 + random.uniform(0, 0.35)
        else:
            base_yes_prob = 0.10 + random.uniform(0, 0.35)
        noise = random.uniform(-sim.surprise_factor, sim.surprise_factor)
        yes_mid = max(0.05, min(0.95, base_yes_prob + noise))
        no_mid = 1.0 - yes_mid
        half_spread = sim.spread / 2
        return {
            "yes_bid": max(0.01, yes_mid - half_spread),
            "yes_ask": min(0.99, yes_mid + half_spread),
            "no_bid": max(0.01, no_mid - half_spread),
            "no_ask": min(0.99, no_mid + half_spread),
        }

    def execute_trade(self, fomc, strike, side, quantity, entry_price, edge=0, reason=""):
        is_yes = side == "YES"
        settlement = calculate_settlement(strike, fomc.rate_after, is_yes)
        cost = quantity * entry_price
        revenue = quantity * settlement
        pnl = revenue - cost
        trade = BacktestTrade(fomc.date, fomc.date, strike, side, quantity,
                              entry_price, settlement, pnl, pnl > 0, edge, reason)
        self._capital += pnl
        self._trades.append(trade)
        self._equity_curve.append(self._capital)
        return trade

    def run(self, strategy):
        for fomc in self._fomc_dates:
            strikes = self._get_relevant_strikes(fomc)
            for strike in strikes:
                market = self.simulate_market_price(fomc, strike)
                signal = strategy(fomc, strike, market)
                if signal:
                    qty = min(signal.get("quantity", 1), self._max_position)
                    cost = qty * signal["price"]
                    if cost > self._capital:
                        qty = int(self._capital / signal["price"])
                        if qty <= 0: continue
                    self.execute_trade(fomc, strike, signal["side"], qty,
                                       signal["price"], signal.get("edge", 0), signal.get("reason", ""))
        return self._calculate_performance()

    def _get_relevant_strikes(self, fomc):
        rate = fomc.rate_before
        return [rate + d for d in [-0.25, 0, 0.25] if 0 < rate + d < 10]

    def _calculate_performance(self):
        if not self._trades:
            return BacktestResult(self._start_date, self._end_date, self._initial_capital,
                                  self._capital, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [], self._equity_curve)
        winners = [t for t in self._trades if t.is_winner]
        losers = [t for t in self._trades if not t.is_winner]
        num_winners, num_losers = len(winners), len(losers)
        num_trades = len(self._trades)
        win_rate = num_winners / num_trades if num_trades else 0
        avg_win = sum(t.pnl for t in winners) / num_winners if num_winners else 0
        avg_loss = abs(sum(t.pnl for t in losers) / num_losers) if num_losers else 0
        gross_profit = sum(t.pnl for t in winners)
        gross_loss = abs(sum(t.pnl for t in losers))
        profit_factor = gross_profit / gross_loss if gross_loss else float('inf')
        peak, max_dd = self._initial_capital, 0
        for eq in self._equity_curve:
            if eq > peak: peak = eq
            if peak - eq > max_dd: max_dd = peak - eq
        total_return = self._capital - self._initial_capital
        total_return_pct = total_return / self._initial_capital * 100
        sharpe = 0
        if len(self._trades) > 1:
            rets = [t.pnl / self._initial_capital for t in self._trades]
            avg_ret = sum(rets) / len(rets)
            std_ret = (sum((r - avg_ret)**2 for r in rets) / len(rets))**0.5
            sharpe = (avg_ret / std_ret * len(rets)**0.5) if std_ret else 0
        return BacktestResult(self._start_date, self._end_date, self._initial_capital,
                              self._capital, total_return, total_return_pct, num_trades,
                              num_winners, num_losers, win_rate, avg_win, avg_loss,
                              profit_factor, max_dd, max_dd/self._initial_capital*100, sharpe,
                              self._trades, self._equity_curve)

    def print_report(self, result=None):
        r = result or self._calculate_performance()
        print(f"\n{'='*60}\n📊 Event Contract 回測報告\n{'='*60}")
        print(f"期間: {r.start_date} ~ {r.end_date}")
        print(f"初始: ${r.initial_capital:,.0f} → 最終: ${r.final_capital:,.0f}")
        print(f"回報: ${r.total_return:+,.2f} ({r.total_return_pct:+.1f}%)")
        print(f"交易: {r.num_trades} 筆, 勝率: {r.win_rate:.1%}")
        print(f"獲利因子: {r.profit_factor:.2f}, 最大回撤: {r.max_drawdown_pct:.1f}%")
        print("="*60)

def simple_hold_strategy(fomc, strike, market):
    if abs(strike - fomc.rate_before) < 0.001:
        yes_price = market["yes_ask"]
        if yes_price < 0.80:
            return {"side": "YES", "strike": strike, "quantity": 5,
                    "price": yes_price, "edge": 0.85 - yes_price}
    return None

def edge_based_strategy(fomc, strike, market, min_edge=0.10):
    current_rate = fomc.rate_before
    hold_prob, cut_prob, hike_prob = 0.75, 0.15, 0.10
    if strike <= current_rate - 0.25: expected_yes = 0.95
    elif strike == current_rate: expected_yes = hold_prob + hike_prob
    elif strike >= current_rate + 0.25: expected_yes = hike_prob
    else: expected_yes = 0.50
    yes_price = market["yes_ask"]
    yes_edge = expected_yes - yes_price
    if yes_edge >= min_edge:
        return {"side": "YES", "strike": strike, "quantity": max(1, int(yes_edge*20)),
                "price": yes_price, "edge": yes_edge}
    no_price = market["no_ask"]
    no_edge = (1 - expected_yes) - no_price
    if no_edge >= min_edge:
        return {"side": "NO", "strike": strike, "quantity": max(1, int(no_edge*20)),
                "price": no_price, "edge": no_edge}
    return None

def run_backtest(strategy, initial_capital=10000, start_date="20230101",
                 end_date="20251231", print_report=True):
    bt = EventContractBacktest(initial_capital, start_date, end_date)
    result = bt.run(strategy)
    if print_report: bt.print_report(result)
    return result
