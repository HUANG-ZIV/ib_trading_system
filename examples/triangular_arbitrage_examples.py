#!/usr/bin/env python3
"""
三角套利策略完整範例
Triangular Arbitrage Strategy - Complete Example

這個腳本展示如何：
1. 配置策略參數
2. 下載歷史數據
3. 執行回測
4. 分析結果
5. 運行即時交易
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# 設置路徑 - 指向專案根目錄
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def example_1_basic_backtest():
    """
    範例 1: 基本回測
    
    使用模擬數據進行回測
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Backtest with Synthetic Data")
    print("=" * 60 + "\n")
    
    from strategies.triangular_arbitrage import (
        TriangularArbitrageConfig,
        TriangleType,
    )
    from backtest.triangular_backtest import (
        TriangularArbitrageBacktester,
        BacktestConfig,
        generate_synthetic_data,
    )
    
    # 1. 生成測試數據
    print("Generating synthetic data...")
    data = generate_synthetic_data(
        start_date="2018-01-01",
        end_date="2024-12-31",
        freq="1h",
    )
    print(f"Data shape: {data.shape}")
    
    # 2. 配置策略
    strategy_config = TriangularArbitrageConfig(
        strategy_id="triangular_test",
        enabled_triangles=[
            TriangleType.T1_XAU_XAG_XPT,
            TriangleType.T2_XAU_XAG_XPD,
        ],
        lookback_period=120,
        entry_zscore=2.0,
        exit_zscore=0.5,
        stop_zscore=3.5,
        min_deviation_pct=0.5,
        capital_per_triangle=50000,
        max_triangles=2,
        warmup_bars=150,
    )
    
    backtest_config = BacktestConfig(
        initial_capital=500000,
        start_date="2019-01-01",
        end_date="2024-12-31",
    )
    
    # 3. 執行回測
    print("\nRunning backtest...")
    backtester = TriangularArbitrageBacktester(
        strategy_config=strategy_config,
        backtest_config=backtest_config,
    )
    
    result = backtester.run(data, verbose=True)
    
    # 4. 視覺化結果（如果有 matplotlib）
    try:
        from backtest.visualization import BacktestVisualizer
        
        print("\nGenerating charts...")
        viz = BacktestVisualizer(result)
        viz.generate_report("backtest_reports")
        
    except ImportError:
        print("matplotlib not available, skipping visualization")
    
    return result


def example_2_parameter_optimization():
    """
    範例 2: 參數優化
    
    測試不同參數組合
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Parameter Optimization")
    print("=" * 60 + "\n")
    
    from strategies.triangular_arbitrage import (
        TriangularArbitrageConfig,
        TriangleType,
    )
    from backtest.triangular_backtest import (
        TriangularArbitrageBacktester,
        BacktestConfig,
        generate_synthetic_data,
    )
    
    # 生成數據
    data = generate_synthetic_data("2018-01-01", "2024-12-31", "1H")
    
    # 參數網格
    param_grid = {
        "entry_zscore": [1.5, 2.0, 2.5],
        "exit_zscore": [0.3, 0.5, 0.7],
        "lookback_period": [60, 120, 180],
    }
    
    results = []
    total_combinations = len(param_grid["entry_zscore"]) * len(param_grid["exit_zscore"]) * len(param_grid["lookback_period"])
    
    print(f"Testing {total_combinations} parameter combinations...")
    
    for entry_z in param_grid["entry_zscore"]:
        for exit_z in param_grid["exit_zscore"]:
            for lookback in param_grid["lookback_period"]:
                
                config = TriangularArbitrageConfig(
                    enabled_triangles=[TriangleType.T1_XAU_XAG_XPT],
                    entry_zscore=entry_z,
                    exit_zscore=exit_z,
                    lookback_period=lookback,
                    capital_per_triangle=50000,
                    warmup_bars=lookback + 30,
                )
                
                backtest_config = BacktestConfig(
                    initial_capital=500000,
                    start_date="2019-01-01",
                    end_date="2024-12-31",
                )
                
                backtester = TriangularArbitrageBacktester(config, backtest_config)
                result = backtester.run(data, verbose=False)
                
                results.append({
                    "entry_z": entry_z,
                    "exit_z": exit_z,
                    "lookback": lookback,
                    "sharpe": result.sharpe_ratio,
                    "cagr": result.cagr,
                    "max_dd": result.max_drawdown_pct,
                    "win_rate": result.win_rate,
                    "trades": result.total_trades,
                })
    
    # 顯示結果
    import pandas as pd
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("sharpe", ascending=False)
    
    print("\nTop 10 Parameter Combinations by Sharpe Ratio:")
    print(results_df.head(10).to_string(index=False))
    
    return results_df


def example_3_walk_forward_analysis():
    """
    範例 3: 滾動前進分析
    
    避免過度擬合的驗證方法
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Walk-Forward Analysis")
    print("=" * 60 + "\n")
    
    from strategies.triangular_arbitrage import (
        TriangularArbitrageConfig,
        TriangleType,
    )
    from backtest.triangular_backtest import (
        TriangularArbitrageBacktester,
        BacktestConfig,
        generate_synthetic_data,
    )
    import pandas as pd
    
    # 生成數據
    full_data = generate_synthetic_data("2016-01-01", "2024-12-31", "1H")
    
    # 定義滾動窗口
    windows = [
        {"train_start": "2016-01-01", "train_end": "2018-12-31", "test_start": "2019-01-01", "test_end": "2019-12-31"},
        {"train_start": "2017-01-01", "train_end": "2019-12-31", "test_start": "2020-01-01", "test_end": "2020-12-31"},
        {"train_start": "2018-01-01", "train_end": "2020-12-31", "test_start": "2021-01-01", "test_end": "2021-12-31"},
        {"train_start": "2019-01-01", "train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-12-31"},
        {"train_start": "2020-01-01", "train_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2023-12-31"},
        {"train_start": "2021-01-01", "train_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31"},
    ]
    
    wf_results = []
    
    for i, window in enumerate(windows):
        print(f"\nWindow {i+1}: Train {window['train_start']} to {window['train_end']}, Test {window['test_start']} to {window['test_end']}")
        
        # 訓練期回測
        train_config = BacktestConfig(
            initial_capital=500000,
            start_date=window["train_start"],
            end_date=window["train_end"],
        )
        
        strategy_config = TriangularArbitrageConfig(
            enabled_triangles=[TriangleType.T1_XAU_XAG_XPT],
            entry_zscore=2.0,
            exit_zscore=0.5,
            capital_per_triangle=50000,
        )
        
        # 訓練期
        train_backtester = TriangularArbitrageBacktester(strategy_config, train_config)
        train_result = train_backtester.run(full_data, verbose=False)
        
        # 測試期
        test_config = BacktestConfig(
            initial_capital=500000,
            start_date=window["test_start"],
            end_date=window["test_end"],
        )
        
        test_backtester = TriangularArbitrageBacktester(strategy_config, test_config)
        test_result = test_backtester.run(full_data, verbose=False)
        
        wf_results.append({
            "window": i + 1,
            "test_period": f"{window['test_start'][:4]}",
            "train_sharpe": train_result.sharpe_ratio,
            "test_sharpe": test_result.sharpe_ratio,
            "train_cagr": train_result.cagr,
            "test_cagr": test_result.cagr,
            "test_trades": test_result.total_trades,
        })
    
    wf_df = pd.DataFrame(wf_results)
    print("\nWalk-Forward Results:")
    print(wf_df.to_string(index=False))
    
    # 計算整體指標
    avg_test_sharpe = wf_df["test_sharpe"].mean()
    avg_test_cagr = wf_df["test_cagr"].mean()
    
    print(f"\nOverall Out-of-Sample Performance:")
    print(f"  Average Test Sharpe: {avg_test_sharpe:.2f}")
    print(f"  Average Test CAGR: {avg_test_cagr:.1f}%")
    
    return wf_df


def example_4_strategy_monitoring():
    """
    範例 4: 策略監控
    
    展示如何監控運行中的策略
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Strategy Monitoring (Simulation)")
    print("=" * 60 + "\n")
    
    from strategies.triangular_arbitrage import (
        TriangularArbitrageStrategy,
        TriangularArbitrageConfig,
        TriangleType,
        SPOT_SYMBOLS,
    )
    from backtest.triangular_backtest import generate_synthetic_data
    from datetime import datetime, timedelta
    import time
    
    # 配置
    config = TriangularArbitrageConfig(
        enabled_triangles=[
            TriangleType.T1_XAU_XAG_XPT,
            TriangleType.T2_XAU_XAG_XPD,
        ],
        entry_zscore=2.0,
        exit_zscore=0.5,
        capital_per_triangle=50000,
        warmup_bars=50,  # 減少預熱時間用於演示
    )
    
    # 創建策略
    strategy = TriangularArbitrageStrategy(config=config)
    strategy.start()
    
    # 生成模擬數據
    data = generate_synthetic_data("2024-01-01", "2024-01-31", "1H")
    
    print("Simulating strategy monitoring...")
    print("Press Ctrl+C to stop\n")
    
    try:
        for idx, (_, row) in enumerate(data.iterrows()):
            # 模擬每個時間點的數據
            timestamp = row["datetime"]
            
            for asset in ["XAU", "XAG", "XPT", "XPD"]:
                bar_data = {
                    "symbol": SPOT_SYMBOLS[asset],
                    "timestamp": timestamp,
                    "open": row[asset],
                    "high": row[asset] * 1.001,
                    "low": row[asset] * 0.999,
                    "close": row[asset],
                    "volume": 1000,
                }
                strategy.on_bar(bar_data)
            
            # 每 24 小時顯示一次狀態
            if idx > 0 and idx % 24 == 0:
                status = strategy.get_status()
                states = strategy.get_triangle_states()
                
                print(f"\n[{timestamp}]")
                print(f"  Warmup: {status['is_warming_up']} ({status['warmup_progress']})")
                print(f"  Open positions: {status['open_positions']}")
                print(f"  Daily PnL: ${status['daily_pnl']:.2f}")
                
                for name, state in states.items():
                    print(f"  {name}: Z={state['zscore']:.2f}, Dev={state['deviation_pct']:.3f}%")
            
            # 限制演示時間
            if idx >= 500:
                break
                
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    strategy.stop()
    
    # 顯示最終統計
    perf = strategy.get_performance_summary()
    print("\nFinal Performance:")
    print(f"  Total trades: {perf['total_trades']}")
    print(f"  Win rate: {perf['win_rate']:.1f}%")
    print(f"  Total PnL: ${perf['total_pnl']:.2f}")


def main():
    """主函數"""
    print("\n" + "=" * 60)
    print("TRIANGULAR ARBITRAGE STRATEGY - EXAMPLES")
    print("=" * 60)
    
    while True:
        print("\nSelect an example to run:")
        print("1. Basic Backtest")
        print("2. Parameter Optimization")
        print("3. Walk-Forward Analysis")
        print("4. Strategy Monitoring (Simulation)")
        print("5. Run All Examples")
        print("0. Exit")
        
        choice = input("\nEnter your choice: ").strip()
        
        if choice == "1":
            example_1_basic_backtest()
        elif choice == "2":
            example_2_parameter_optimization()
        elif choice == "3":
            example_3_walk_forward_analysis()
        elif choice == "4":
            example_4_strategy_monitoring()
        elif choice == "5":
            example_1_basic_backtest()
            example_2_parameter_optimization()
            example_3_walk_forward_analysis()
            example_4_strategy_monitoring()
        elif choice == "0":
            print("Goodbye!")
            break
        else:
            print("Invalid choice, please try again.")


if __name__ == "__main__":
    main()