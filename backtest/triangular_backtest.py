"""
三角套利策略回測引擎
Triangular Arbitrage Backtest Engine
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
import json
from pathlib import Path

from strategies.triangular_arbitrage import (
    TriangularArbitrageConfig,
    BacktestConfig,
    TriangleType,
    TRIANGLE_DEFINITIONS,
    SPOT_SYMBOLS,
    TriangleCalculator,
    TrianglePositionManager,
    DEFAULT_CONFIG,
    DEFAULT_BACKTEST_CONFIG,
)

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """回測交易記錄"""
    trade_id: int
    triangle: str
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_zscore: float
    exit_zscore: float
    entry_deviation: float
    exit_deviation: float
    pnl: float
    pnl_pct: float
    holding_days: int
    exit_reason: str


@dataclass
class BacktestResult:
    """回測結果"""
    # 基本統計
    total_return: float = 0.0
    total_return_pct: float = 0.0
    cagr: float = 0.0
    
    # 風險指標
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    calmar_ratio: float = 0.0
    
    # 交易統計
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    avg_holding_days: float = 0.0
    
    # 各三角統計
    triangle_stats: Dict[str, Dict] = field(default_factory=dict)
    
    # 時間序列
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)
    dates: List[datetime] = field(default_factory=list)
    
    # 交易記錄
    trades: List[BacktestTrade] = field(default_factory=list)
    
    # 月度報酬
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    
    # 年度報酬
    yearly_returns: Dict[str, float] = field(default_factory=dict)


class TriangularArbitrageBacktester:
    """三角套利回測器"""
    
    def __init__(
        self,
        strategy_config: Optional[TriangularArbitrageConfig] = None,
        backtest_config: Optional[BacktestConfig] = None,
    ):
        self.strategy_config = strategy_config or DEFAULT_CONFIG
        self.backtest_config = backtest_config or DEFAULT_BACKTEST_CONFIG
        
        # 計算器
        self.calculator = TriangleCalculator(
            lookback_period=self.strategy_config.lookback_period,
            use_log_deviation=True,
        )
        
        # 部位管理
        self.position_manager = TrianglePositionManager()
        
        # 回測狀態
        self._equity = self.backtest_config.initial_capital
        self._equity_curve = []
        self._dates = []
        self._trades: List[BacktestTrade] = []
        self._trade_counter = 0
        
        # 最高淨值（用於計算回撤）
        self._peak_equity = self._equity
        self._max_drawdown = 0.0
    
    def run(
        self,
        price_data: pd.DataFrame,
        verbose: bool = True,
    ) -> BacktestResult:
        """
        執行回測
        
        Args:
            price_data: 價格數據，需包含列：
                - datetime (index or column)
                - XAU, XAG, XPT, XPD (價格列)
            verbose: 是否顯示詳細日誌
                
        Returns:
            BacktestResult
        """
        logger.info("=" * 50)
        logger.info("Starting backtest...")
        logger.info(f"Period: {self.backtest_config.start_date} to {self.backtest_config.end_date}")
        logger.info(f"Initial capital: ${self.backtest_config.initial_capital:,.2f}")
        logger.info("=" * 50)
        
        # 重置狀態
        self._reset()
        
        # 確保有 datetime 索引
        if "datetime" in price_data.columns:
            price_data = price_data.set_index("datetime")
        
        # 過濾日期範圍
        start = pd.to_datetime(self.backtest_config.start_date)
        end = pd.to_datetime(self.backtest_config.end_date)
        price_data = price_data[(price_data.index >= start) & (price_data.index <= end)]
        
        # 預熱期
        warmup_end_idx = self.strategy_config.warmup_bars
        
        # 主回測循環
        for idx, (timestamp, row) in enumerate(price_data.iterrows()):
            # 取得價格
            prices = {
                "XAU": row.get("XAU", row.get("XAUUSD", 0)),
                "XAG": row.get("XAG", row.get("XAGUSD", 0)),
                "XPT": row.get("XPT", row.get("XPTUSD", 0)),
                "XPD": row.get("XPD", row.get("XPDUSD", 0)),
            }
            
            # 檢查價格是否有效
            if not all(v > 0 for v in prices.values()):
                continue
            
            # 更新計算器
            self.calculator.update_prices(prices, timestamp)
            
            # 預熱期
            if idx < warmup_end_idx:
                continue
            
            # 更新持倉 PnL
            self.position_manager.update_positions(prices, timestamp)
            
            # 檢查出場
            self._check_exits(prices, timestamp)
            
            # 檢查進場
            self._check_entries(prices, timestamp)
            
            # 更新淨值曲線
            self._update_equity(prices, timestamp)
        
        # 強制平倉所有持倉
        self._close_all_positions(prices, timestamp, "end_of_backtest")
        
        # 計算結果
        result = self._calculate_results()
        
        if verbose:
            self._print_summary(result)
        
        return result
    
    def _reset(self) -> None:
        """重置回測狀態"""
        self._equity = self.backtest_config.initial_capital
        self._equity_curve = []
        self._dates = []
        self._trades = []
        self._trade_counter = 0
        self._peak_equity = self._equity
        self._max_drawdown = 0.0
        
        self.calculator = TriangleCalculator(
            lookback_period=self.strategy_config.lookback_period,
            use_log_deviation=True,
        )
        self.position_manager = TrianglePositionManager()
    
    def _check_entries(self, prices: Dict[str, float], timestamp: datetime) -> None:
        """檢查進場"""
        # 檢查持倉數量限制
        if self.position_manager.get_position_count() >= self.strategy_config.max_triangles:
            return
        
        # 生成信號
        signals = self.calculator.generate_signals(
            entry_zscore=self.strategy_config.entry_zscore,
            min_deviation_pct=self.strategy_config.min_deviation_pct,
            enabled_triangles=self.strategy_config.enabled_triangles,
        )
        
        for signal in signals:
            if self.position_manager.has_position(signal.triangle_type):
                continue
            
            # 計算部位
            positions = self.calculator.calculate_positions(
                signal=signal,
                capital=self.strategy_config.capital_per_triangle,
                prices=prices,
            )
            
            if not positions:
                continue
            
            # 計算交易成本
            cost = self._calculate_transaction_cost(positions, prices)
            
            # 開倉
            self.position_manager.open_position(
                tri_type=signal.triangle_type,
                signal=signal,
                prices=prices,
            )
            
            self._equity -= cost
            
            logger.debug(f"Entry: {signal.triangle_type.value}, Z={signal.zscore:.2f}")
            break
    
    def _check_exits(self, prices: Dict[str, float], timestamp: datetime) -> None:
        """檢查出場"""
        exits = []
        
        for tri_type, position in list(self.position_manager.open_positions.items()):
            state = self.calculator.get_triangle_state(tri_type)
            exit_reason = None
            
            # 獲利出場
            if abs(state.zscore) < self.strategy_config.exit_zscore:
                exit_reason = "profit_target"
            
            # 時間出場
            elif position.holding_days >= self.strategy_config.max_holding_days:
                exit_reason = "time_stop"
            
            # 停損
            elif abs(state.zscore) > self.strategy_config.stop_zscore:
                exit_reason = "stop_loss"
            
            if exit_reason:
                exits.append((tri_type, exit_reason, state.zscore, state.deviation_pct))
        
        for tri_type, reason, exit_z, exit_dev in exits:
            self._execute_exit(tri_type, prices, timestamp, reason, exit_z, exit_dev)
    
    def _execute_exit(
        self,
        tri_type: TriangleType,
        prices: Dict[str, float],
        timestamp: datetime,
        reason: str,
        exit_zscore: float,
        exit_deviation: float,
    ) -> None:
        """執行出場"""
        position = self.position_manager.open_positions.get(tri_type)
        if not position:
            return
        
        # 計算交易成本
        cost = self._calculate_transaction_cost(position.positions, prices)
        
        # 平倉
        pnl = self.position_manager.close_position(tri_type, prices, reason)
        pnl -= cost  # 扣除成本
        
        self._equity += pnl
        
        # 計算 PnL 百分比
        initial_value = self.strategy_config.capital_per_triangle
        pnl_pct = (pnl / initial_value) * 100 if initial_value > 0 else 0
        
        # 記錄交易
        self._trade_counter += 1
        trade = BacktestTrade(
            trade_id=self._trade_counter,
            triangle=tri_type.value,
            entry_time=position.entry_time,
            exit_time=timestamp,
            direction="short_dev" if position.entry_zscore > 0 else "long_dev",
            entry_zscore=position.entry_zscore,
            exit_zscore=exit_zscore,
            entry_deviation=position.entry_deviation,
            exit_deviation=exit_deviation,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_days=position.holding_days,
            exit_reason=reason,
        )
        self._trades.append(trade)
        
        logger.debug(f"Exit: {tri_type.value}, PnL={pnl:.2f}, Reason={reason}")
    
    def _close_all_positions(
        self,
        prices: Dict[str, float],
        timestamp: datetime,
        reason: str,
    ) -> None:
        """平倉所有持倉"""
        for tri_type in list(self.position_manager.open_positions.keys()):
            state = self.calculator.get_triangle_state(tri_type)
            self._execute_exit(
                tri_type, prices, timestamp, reason,
                state.zscore, state.deviation_pct
            )
    
    def _calculate_transaction_cost(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
    ) -> float:
        """計算交易成本"""
        cost = 0.0
        
        for symbol, units in positions.items():
            # 找對應的 asset
            for asset, spot_symbol in SPOT_SYMBOLS.items():
                if symbol == spot_symbol:
                    price = prices.get(asset, 0)
                    notional = abs(units) * price
                    
                    # 點差成本
                    cost += notional * self.backtest_config.spot_spread_pct
                    
                    # 滑價
                    cost += notional * self.backtest_config.slippage_pct
                    break
        
        return cost
    
    def _update_equity(self, prices: Dict[str, float], timestamp: datetime) -> None:
        """更新淨值曲線"""
        # 計算未實現損益
        unrealized_pnl = 0.0
        for position in self.position_manager.open_positions.values():
            unrealized_pnl += position.current_pnl
        
        current_equity = self._equity + unrealized_pnl
        
        self._equity_curve.append(current_equity)
        self._dates.append(timestamp)
        
        # 更新最大回撤
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
        
        drawdown = (self._peak_equity - current_equity) / self._peak_equity
        if drawdown > self._max_drawdown:
            self._max_drawdown = drawdown
    
    def _calculate_results(self) -> BacktestResult:
        """計算回測結果"""
        result = BacktestResult()
        
        if not self._equity_curve:
            return result
        
        # 基本統計
        initial = self.backtest_config.initial_capital
        final = self._equity_curve[-1]
        
        result.total_return = final - initial
        result.total_return_pct = (final / initial - 1) * 100
        
        # 計算年化報酬
        if self._dates:
            years = (self._dates[-1] - self._dates[0]).days / 365.25
            if years > 0:
                result.cagr = ((final / initial) ** (1 / years) - 1) * 100
        
        # 風險指標
        equity_series = pd.Series(self._equity_curve, index=self._dates)
        returns = equity_series.pct_change().dropna()
        
        if len(returns) > 1:
            # 夏普比率
            rf_daily = self.backtest_config.risk_free_rate / 252
            excess_returns = returns - rf_daily
            if returns.std() > 0:
                result.sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std()
            
            # 索提諾比率
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                result.sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_returns.std()
        
        # 最大回撤
        result.max_drawdown = self._max_drawdown * initial
        result.max_drawdown_pct = self._max_drawdown * 100
        
        # 卡瑪比率
        if result.max_drawdown_pct > 0:
            result.calmar_ratio = result.cagr / result.max_drawdown_pct
        
        # 交易統計
        result.trades = self._trades
        result.total_trades = len(self._trades)
        
        if self._trades:
            wins = [t for t in self._trades if t.pnl > 0]
            losses = [t for t in self._trades if t.pnl <= 0]
            
            result.winning_trades = len(wins)
            result.losing_trades = len(losses)
            result.win_rate = len(wins) / len(self._trades) * 100
            
            total_wins = sum(t.pnl for t in wins) if wins else 0
            total_losses = abs(sum(t.pnl for t in losses)) if losses else 0
            
            result.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            result.avg_win = total_wins / len(wins) if wins else 0
            result.avg_loss = total_losses / len(losses) if losses else 0
            result.avg_trade = sum(t.pnl for t in self._trades) / len(self._trades)
            result.avg_holding_days = np.mean([t.holding_days for t in self._trades])
            
            # 各三角統計
            for tri_type in self.strategy_config.enabled_triangles:
                tri_trades = [t for t in self._trades if t.triangle == tri_type.value]
                if tri_trades:
                    tri_wins = [t for t in tri_trades if t.pnl > 0]
                    result.triangle_stats[tri_type.value] = {
                        "trades": len(tri_trades),
                        "wins": len(tri_wins),
                        "win_rate": len(tri_wins) / len(tri_trades) * 100,
                        "total_pnl": sum(t.pnl for t in tri_trades),
                        "avg_pnl": np.mean([t.pnl for t in tri_trades]),
                    }
        
        # 時間序列
        result.equity_curve = self._equity_curve
        result.dates = self._dates
        
        # 計算回撤曲線
        peak = self._equity_curve[0]
        result.drawdown_curve = []
        for eq in self._equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            result.drawdown_curve.append(dd)
        
        # 月度報酬
        if self._dates:
            equity_series = pd.Series(self._equity_curve, index=pd.to_datetime(self._dates))
            monthly = equity_series.resample('ME').last()
            monthly_returns = monthly.pct_change().dropna()
            
            for date, ret in monthly_returns.items():
                result.monthly_returns[date.strftime("%Y-%m")] = ret * 100
        
        # 年度報酬
        if self._dates:
            yearly = equity_series.resample('YE').last()
            yearly_returns = yearly.pct_change().dropna()
            
            for date, ret in yearly_returns.items():
                result.yearly_returns[str(date.year)] = ret * 100
        
        return result
    
    def _print_summary(self, result: BacktestResult) -> None:
        """打印回測摘要"""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS SUMMARY")
        print("=" * 60)
        
        print(f"\n{'Performance Metrics':=^50}")
        print(f"Total Return:        ${result.total_return:>12,.2f} ({result.total_return_pct:>6.2f}%)")
        print(f"CAGR:                {result.cagr:>12.2f}%")
        print(f"Sharpe Ratio:        {result.sharpe_ratio:>12.2f}")
        print(f"Sortino Ratio:       {result.sortino_ratio:>12.2f}")
        print(f"Max Drawdown:        ${result.max_drawdown:>12,.2f} ({result.max_drawdown_pct:>6.2f}%)")
        print(f"Calmar Ratio:        {result.calmar_ratio:>12.2f}")
        
        print(f"\n{'Trade Statistics':=^50}")
        print(f"Total Trades:        {result.total_trades:>12}")
        print(f"Winning Trades:      {result.winning_trades:>12}")
        print(f"Losing Trades:       {result.losing_trades:>12}")
        print(f"Win Rate:            {result.win_rate:>12.1f}%")
        print(f"Profit Factor:       {result.profit_factor:>12.2f}")
        print(f"Avg Win:             ${result.avg_win:>12,.2f}")
        print(f"Avg Loss:            ${result.avg_loss:>12,.2f}")
        print(f"Avg Trade:           ${result.avg_trade:>12,.2f}")
        print(f"Avg Holding Days:    {result.avg_holding_days:>12.1f}")
        
        if result.triangle_stats:
            print(f"\n{'Performance by Triangle':=^50}")
            for tri, stats in result.triangle_stats.items():
                print(f"\n{tri}:")
                print(f"  Trades: {stats['trades']}, Win Rate: {stats['win_rate']:.1f}%")
                print(f"  Total PnL: ${stats['total_pnl']:,.2f}, Avg PnL: ${stats['avg_pnl']:,.2f}")
        
        if result.yearly_returns:
            print(f"\n{'Yearly Returns':=^50}")
            for year, ret in sorted(result.yearly_returns.items()):
                print(f"  {year}: {ret:>8.2f}%")
        
        print("\n" + "=" * 60)


def load_sample_data(filepath: str) -> pd.DataFrame:
    """
    載入範例數據
    
    數據格式應為 CSV，包含：
    datetime, XAU, XAG, XPT, XPD
    """
    df = pd.read_csv(filepath, parse_dates=["datetime"])
    return df


def generate_synthetic_data(
    start_date: str = "2015-01-01",
    end_date: str = "2024-12-31",
    freq: str = "1h",
) -> pd.DataFrame:
    """
    生成模擬數據用於測試
    
    注意：這只是用於測試程式碼，不代表真實市場
    產生的數據會包含週期性的三角偏離，以便測試策略
    """
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    n = len(dates)
    
    np.random.seed(42)
    
    # 模擬價格走勢（使用幾何布朗運動）
    def gbm(n, s0, mu, sigma):
        dt = 1/252/24  # hourly
        returns = np.random.normal(mu*dt, sigma*np.sqrt(dt), n)
        return s0 * np.exp(np.cumsum(returns))
    
    # 基礎價格
    xau_base = gbm(n, 1800, 0.03, 0.12)  # 黃金
    xag_base = gbm(n, 23, 0.02, 0.18)     # 白銀
    xpt_base = gbm(n, 950, 0.01, 0.15)    # 鉑金
    xpd_base = gbm(n, 1400, 0.04, 0.25)   # 鈀金
    
    # 加入共同市場因子（高相關性）
    common_factor = np.cumsum(np.random.normal(0, 0.0008, n))
    xau = xau_base * np.exp(common_factor * 0.8)
    xag = xag_base * np.exp(common_factor * 0.85)
    xpt = xpt_base * np.exp(common_factor * 0.6)
    xpd = xpd_base * np.exp(common_factor * 0.4)
    
    # 加入週期性的三角偏離（這是觸發交易的關鍵）
    # 模擬市場中偶爾出現的定價偏離
    hours = np.arange(n)
    
    # 偏離週期：大約每 200-500 小時出現一次顯著偏離
    deviation_cycle_1 = np.sin(hours / 350 * 2 * np.pi) * 0.015
    deviation_cycle_2 = np.sin(hours / 500 * 2 * np.pi + 1.5) * 0.012
    deviation_cycle_3 = np.sin(hours / 280 * 2 * np.pi + 3.0) * 0.010
    
    # 加入隨機突發偏離
    random_spikes = np.zeros(n)
    spike_indices = np.random.choice(n, size=n//200, replace=False)
    random_spikes[spike_indices] = np.random.normal(0, 0.025, len(spike_indices))
    # 讓突發偏離持續一段時間後衰減
    for i in range(1, min(50, n)):
        random_spikes[i:] += random_spikes[:-i] * 0.85 ** i
    
    # 將偏離應用到不同商品（產生三角不一致性）
    xag = xag * (1 + deviation_cycle_1 + random_spikes * 0.8)
    xpt = xpt * (1 + deviation_cycle_2 - random_spikes * 0.5)
    xpd = xpd * (1 + deviation_cycle_3 + random_spikes * 0.3)
    
    # 確保價格為正
    xau = np.maximum(xau, 100)
    xag = np.maximum(xag, 5)
    xpt = np.maximum(xpt, 100)
    xpd = np.maximum(xpd, 100)
    
    df = pd.DataFrame({
        "datetime": dates,
        "XAU": xau,
        "XAG": xag,
        "XPT": xpt,
        "XPD": xpd,
    })
    
    return df


# ==================== 主程式 ====================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 生成測試數據
    print("Generating synthetic data...")
    data = generate_synthetic_data()
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")
    
    # 配置
    strategy_config = TriangularArbitrageConfig(
        enabled_triangles=[
            TriangleType.T1_XAU_XAG_XPT,
            TriangleType.T2_XAU_XAG_XPD,
        ],
        lookback_period=120,
        entry_zscore=2.0,
        exit_zscore=0.5,
        stop_zscore=3.5,
        capital_per_triangle=50000,
        max_triangles=2,
        warmup_bars=150,
    )
    
    backtest_config = BacktestConfig(
        initial_capital=500000,
        start_date="2016-01-01",
        end_date="2024-12-31",
    )
    
    # 執行回測
    backtester = TriangularArbitrageBacktester(
        strategy_config=strategy_config,
        backtest_config=backtest_config,
    )
    
    result = backtester.run(data, verbose=True)
    
    # 保存結果
    print("\nSaving results...")
    
    # 保存交易記錄
    if result.trades:
        trades_df = pd.DataFrame([
            {
                "trade_id": t.trade_id,
                "triangle": t.triangle,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "direction": t.direction,
                "entry_zscore": t.entry_zscore,
                "exit_zscore": t.exit_zscore,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "holding_days": t.holding_days,
                "exit_reason": t.exit_reason,
            }
            for t in result.trades
        ])
        trades_df.to_csv("backtest_trades.csv", index=False)
        print("Trades saved to backtest_trades.csv")
    
    # 保存淨值曲線
    if result.equity_curve:
        equity_df = pd.DataFrame({
            "datetime": result.dates,
            "equity": result.equity_curve,
            "drawdown_pct": result.drawdown_curve,
        })
        equity_df.to_csv("backtest_equity.csv", index=False)
        print("Equity curve saved to backtest_equity.csv")