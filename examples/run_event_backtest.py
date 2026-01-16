#!/usr/bin/env python3
"""Event Contract 回測範例"""
import sys
sys.path.insert(0, '.')
from backtest.event_contract import (run_backtest, simple_hold_strategy, 
                                      edge_based_strategy, print_fomc_history, get_action_statistics)

print("\n📊 FOMC 統計")
stats = get_action_statistics()
print(f"總會議: {stats['total']}, 維持: {stats['holds']} ({stats['hold_pct']:.0f}%)")

print("\n🎯 回測: 維持策略")
r1 = run_backtest(simple_hold_strategy, start_date="20230101", end_date="20251231")

print("\n🎯 回測: Edge 策略")  
r2 = run_backtest(lambda f,s,m: edge_based_strategy(f,s,m,0.10))

print(f"\n📊 比較: 維持={r1.total_return_pct:+.1f}%, Edge={r2.total_return_pct:+.1f}%")
