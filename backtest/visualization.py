"""
三角套利回測結果視覺化
Triangular Arbitrage Backtest Visualization
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not installed for visualization")

from backtest.triangular_backtest import BacktestResult


class BacktestVisualizer:
    """回測結果視覺化"""
    
    def __init__(self, result: BacktestResult):
        self.result = result
        
        if HAS_MPL:
            plt.style.use('seaborn-v0_8-whitegrid')
    
    def plot_equity_curve(
        self,
        figsize: tuple = (14, 6),
        save_path: Optional[str] = None,
    ) -> None:
        """繪製淨值曲線"""
        if not HAS_MPL:
            print("matplotlib required for plotting")
            return
        
        if not self.result.equity_curve:
            print("No equity curve data")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        dates = pd.to_datetime(self.result.dates)
        equity = self.result.equity_curve
        
        ax.plot(dates, equity, 'b-', linewidth=1.5, label='Equity')
        ax.fill_between(dates, equity, alpha=0.3)
        
        # 標記最高點和最低點
        max_idx = np.argmax(equity)
        min_idx = np.argmin(equity)
        
        ax.scatter(dates[max_idx], equity[max_idx], color='green', s=100, zorder=5, label=f'Peak: ${equity[max_idx]:,.0f}')
        ax.scatter(dates[min_idx], equity[min_idx], color='red', s=100, zorder=5, label=f'Trough: ${equity[min_idx]:,.0f}')
        
        ax.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity ($)')
        ax.legend(loc='upper left')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45)
        
        # 添加統計信息
        stats_text = f"Total Return: {self.result.total_return_pct:.1f}%\n"
        stats_text += f"CAGR: {self.result.cagr:.1f}%\n"
        stats_text += f"Sharpe: {self.result.sharpe_ratio:.2f}\n"
        stats_text += f"Max DD: {self.result.max_drawdown_pct:.1f}%"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
    
    def plot_drawdown(
        self,
        figsize: tuple = (14, 4),
        save_path: Optional[str] = None,
    ) -> None:
        """繪製回撤曲線"""
        if not HAS_MPL:
            return
        
        if not self.result.drawdown_curve:
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        dates = pd.to_datetime(self.result.dates)
        dd = [-x for x in self.result.drawdown_curve]  # 負值顯示
        
        ax.fill_between(dates, dd, 0, color='red', alpha=0.3)
        ax.plot(dates, dd, 'r-', linewidth=1)
        
        ax.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45)
        
        # 標記最大回撤
        min_dd_idx = np.argmin(dd)
        ax.scatter(dates[min_dd_idx], dd[min_dd_idx], color='darkred', s=100, zorder=5)
        ax.annotate(f'Max DD: {-dd[min_dd_idx]:.1f}%', 
                   xy=(dates[min_dd_idx], dd[min_dd_idx]),
                   xytext=(10, -20), textcoords='offset points',
                   fontsize=10, color='darkred')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_monthly_returns(
        self,
        figsize: tuple = (14, 6),
        save_path: Optional[str] = None,
    ) -> None:
        """繪製月度報酬熱力圖"""
        if not HAS_MPL:
            return
        
        if not self.result.monthly_returns:
            return
        
        # 整理數據
        monthly_data = {}
        for date_str, ret in self.result.monthly_returns.items():
            year, month = date_str.split('-')
            if year not in monthly_data:
                monthly_data[year] = {}
            monthly_data[year][int(month)] = ret
        
        # 創建矩陣
        years = sorted(monthly_data.keys())
        months = range(1, 13)
        
        data = np.full((len(years), 12), np.nan)
        for i, year in enumerate(years):
            for j, month in enumerate(months):
                if month in monthly_data.get(year, {}):
                    data[i, j] = monthly_data[year][month]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 熱力圖
        cmap = plt.cm.RdYlGn
        im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=-10, vmax=10)
        
        # 設置標籤
        ax.set_xticks(np.arange(12))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.set_yticks(np.arange(len(years)))
        ax.set_yticklabels(years)
        
        # 添加數值標籤
        for i in range(len(years)):
            for j in range(12):
                if not np.isnan(data[i, j]):
                    text = ax.text(j, i, f'{data[i, j]:.1f}',
                                  ha='center', va='center', fontsize=8,
                                  color='white' if abs(data[i, j]) > 5 else 'black')
        
        ax.set_title('Monthly Returns (%)', fontsize=14, fontweight='bold')
        
        # 添加顏色條
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Return (%)', rotation=-90, va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_trade_distribution(
        self,
        figsize: tuple = (14, 5),
        save_path: Optional[str] = None,
    ) -> None:
        """繪製交易分布"""
        if not HAS_MPL:
            return
        
        if not self.result.trades:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        pnls = [t.pnl for t in self.result.trades]
        holding_days = [t.holding_days for t in self.result.trades]
        
        # PnL 分布
        axes[0].hist(pnls, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[0].axvline(np.mean(pnls), color='green', linestyle='-', linewidth=2, label=f'Mean: ${np.mean(pnls):.0f}')
        axes[0].set_title('PnL Distribution', fontweight='bold')
        axes[0].set_xlabel('PnL ($)')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        
        # 持倉天數分布
        axes[1].hist(holding_days, bins=20, color='coral', edgecolor='black', alpha=0.7)
        axes[1].axvline(np.mean(holding_days), color='green', linestyle='-', linewidth=2, 
                       label=f'Mean: {np.mean(holding_days):.1f} days')
        axes[1].set_title('Holding Days Distribution', fontweight='bold')
        axes[1].set_xlabel('Days')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        
        # 按三角分類的 PnL
        triangle_pnls = {}
        for trade in self.result.trades:
            if trade.triangle not in triangle_pnls:
                triangle_pnls[trade.triangle] = []
            triangle_pnls[trade.triangle].append(trade.pnl)
        
        labels = list(triangle_pnls.keys())
        values = [sum(pnls) for pnls in triangle_pnls.values()]
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        bars = axes[2].bar(labels, values, color=colors, edgecolor='black')
        axes[2].axhline(0, color='black', linestyle='-', linewidth=0.5)
        axes[2].set_title('PnL by Triangle', fontweight='bold')
        axes[2].set_xlabel('Triangle')
        axes[2].set_ylabel('Total PnL ($)')
        axes[2].tick_params(axis='x', rotation=45)
        
        # 添加數值標籤
        for bar, val in zip(bars, values):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                        f'${val:,.0f}', ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_zscore_analysis(
        self,
        figsize: tuple = (14, 5),
        save_path: Optional[str] = None,
    ) -> None:
        """繪製 Z-Score 分析"""
        if not HAS_MPL:
            return
        
        if not self.result.trades:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        entry_z = [t.entry_zscore for t in self.result.trades]
        exit_z = [t.exit_zscore for t in self.result.trades]
        pnls = [t.pnl for t in self.result.trades]
        
        # Entry Z-Score vs PnL
        colors = ['green' if p > 0 else 'red' for p in pnls]
        axes[0].scatter(entry_z, pnls, c=colors, alpha=0.6, s=50)
        axes[0].axhline(0, color='black', linestyle='-', linewidth=0.5)
        axes[0].axvline(0, color='black', linestyle='-', linewidth=0.5)
        axes[0].set_title('Entry Z-Score vs PnL', fontweight='bold')
        axes[0].set_xlabel('Entry Z-Score')
        axes[0].set_ylabel('PnL ($)')
        
        # Exit Z-Score 分布（按出場原因）
        exit_reasons = {}
        for trade in self.result.trades:
            reason = trade.exit_reason
            if reason not in exit_reasons:
                exit_reasons[reason] = []
            exit_reasons[reason].append(trade.exit_zscore)
        
        data = [exit_reasons[r] for r in exit_reasons]
        labels = list(exit_reasons.keys())
        
        bp = axes[1].boxplot(data, labels=labels, patch_artist=True)
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        axes[1].set_title('Exit Z-Score by Reason', fontweight='bold')
        axes[1].set_xlabel('Exit Reason')
        axes[1].set_ylabel('Z-Score')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(
        self,
        output_dir: str = "backtest_reports",
    ) -> None:
        """生成完整報告"""
        Path(output_dir).mkdir(exist_ok=True)
        
        print("Generating backtest report...")
        
        # 繪製所有圖表
        self.plot_equity_curve(save_path=f"{output_dir}/equity_curve.png")
        self.plot_drawdown(save_path=f"{output_dir}/drawdown.png")
        self.plot_monthly_returns(save_path=f"{output_dir}/monthly_returns.png")
        self.plot_trade_distribution(save_path=f"{output_dir}/trade_distribution.png")
        self.plot_zscore_analysis(save_path=f"{output_dir}/zscore_analysis.png")
        
        # 生成文字報告
        self._generate_text_report(f"{output_dir}/report.txt")
        
        # 生成交易明細 CSV
        if self.result.trades:
            trades_df = pd.DataFrame([
                {
                    "trade_id": t.trade_id,
                    "triangle": t.triangle,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "direction": t.direction,
                    "entry_zscore": round(t.entry_zscore, 2),
                    "exit_zscore": round(t.exit_zscore, 2),
                    "pnl": round(t.pnl, 2),
                    "pnl_pct": round(t.pnl_pct, 2),
                    "holding_days": t.holding_days,
                    "exit_reason": t.exit_reason,
                }
                for t in self.result.trades
            ])
            trades_df.to_csv(f"{output_dir}/trades.csv", index=False)
        
        print(f"Report generated in {output_dir}/")
    
    def _generate_text_report(self, filepath: str) -> None:
        """生成文字報告"""
        r = self.result
        
        lines = [
            "=" * 60,
            "TRIANGULAR ARBITRAGE BACKTEST REPORT",
            "=" * 60,
            "",
            "PERFORMANCE SUMMARY",
            "-" * 40,
            f"Total Return:        ${r.total_return:>12,.2f} ({r.total_return_pct:>6.2f}%)",
            f"CAGR:                {r.cagr:>12.2f}%",
            f"Sharpe Ratio:        {r.sharpe_ratio:>12.2f}",
            f"Sortino Ratio:       {r.sortino_ratio:>12.2f}",
            f"Max Drawdown:        ${r.max_drawdown:>12,.2f} ({r.max_drawdown_pct:>6.2f}%)",
            f"Calmar Ratio:        {r.calmar_ratio:>12.2f}",
            "",
            "TRADE STATISTICS",
            "-" * 40,
            f"Total Trades:        {r.total_trades:>12}",
            f"Winning Trades:      {r.winning_trades:>12}",
            f"Losing Trades:       {r.losing_trades:>12}",
            f"Win Rate:            {r.win_rate:>12.1f}%",
            f"Profit Factor:       {r.profit_factor:>12.2f}",
            f"Average Win:         ${r.avg_win:>12,.2f}",
            f"Average Loss:        ${r.avg_loss:>12,.2f}",
            f"Average Trade:       ${r.avg_trade:>12,.2f}",
            f"Avg Holding Days:    {r.avg_holding_days:>12.1f}",
            "",
        ]
        
        if r.triangle_stats:
            lines.append("PERFORMANCE BY TRIANGLE")
            lines.append("-" * 40)
            for tri, stats in r.triangle_stats.items():
                lines.append(f"\n{tri}:")
                lines.append(f"  Trades: {stats['trades']}")
                lines.append(f"  Win Rate: {stats['win_rate']:.1f}%")
                lines.append(f"  Total PnL: ${stats['total_pnl']:,.2f}")
                lines.append(f"  Avg PnL: ${stats['avg_pnl']:,.2f}")
            lines.append("")
        
        if r.yearly_returns:
            lines.append("YEARLY RETURNS")
            lines.append("-" * 40)
            for year, ret in sorted(r.yearly_returns.items()):
                lines.append(f"  {year}: {ret:>8.2f}%")
            lines.append("")
        
        lines.append("=" * 60)
        lines.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))


# ==================== 測試 ====================

if __name__ == "__main__":
    # 這個腳本需要先運行回測生成結果
    print("This module provides visualization for backtest results.")
    print("Run triangular_backtest.py first to generate results.")
