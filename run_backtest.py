#!/usr/bin/env python3
"""
run_backtest.py - 回測啟動腳本

用於執行策略回測並產生績效報告
"""

import argparse
import sys
import os
import csv
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any

# 確保專案路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 載入配置
from config.settings import settings
from config.symbols import DEFAULT_WATCHLIST

# 回測引擎
from backtest.engine import BacktestEngine, BacktestConfig, BacktestResult, Trade

# 策略
from strategies.base import StrategyConfig
from strategies.registry import StrategyRegistry
from strategies.examples.sma_cross import SMACrossStrategy
from strategies.examples.tick_scalper import TickScalperStrategy

# 核心組件
from core.events import BarEvent, EventType

# 工具
from utils.logger import setup_logger, get_logger


# 設定 logger
logger = get_logger(__name__)


# ============================================================
# 策略註冊
# ============================================================

def get_strategy_registry() -> StrategyRegistry:
    """取得策略註冊表"""
    registry = StrategyRegistry()
    
    # 註冊內建策略
    registry.register(
        "sma_cross",
        SMACrossStrategy,
        description="SMA 均線交叉策略",
        category="trend",
    )
    
    registry.register(
        "tick_scalper",
        TickScalperStrategy,
        description="Tick 剝頭皮策略",
        category="scalping",
    )
    
    return registry


# ============================================================
# 數據載入
# ============================================================

def generate_sample_data(
    symbol: str,
    start_date: date,
    end_date: date,
    initial_price: float = 100.0,
) -> List[BarEvent]:
    """
    生成模擬歷史數據
    
    注意：這是用於測試的模擬數據
    實際使用時應該從數據源載入真實數據
    
    Args:
        symbol: 標的代碼
        start_date: 開始日期
        end_date: 結束日期
        initial_price: 初始價格
        
    Returns:
        BarEvent 列表
    """
    import random
    
    bars = []
    price = initial_price
    current_date = start_date
    
    while current_date <= end_date:
        # 跳過週末
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue
        
        # 模擬價格變動
        change = random.gauss(0.0002, 0.02)  # 微小正向漂移
        price = price * (1 + change)
        
        # 生成 OHLC
        volatility = price * 0.01
        high = price + abs(random.gauss(0, volatility))
        low = price - abs(random.gauss(0, volatility))
        open_price = random.uniform(low, high)
        close_price = random.uniform(low, high)
        volume = random.randint(100000, 1000000)
        
        # 建立 BarEvent
        bar = BarEvent(
            event_type=EventType.BAR,
            symbol=symbol,
            timestamp=datetime.combine(current_date, datetime.min.time().replace(hour=16)),
            open=round(open_price, 2),
            high=round(high, 2),
            low=round(low, 2),
            close=round(close_price, 2),
            volume=volume,
            timeframe="1d",
        )
        bars.append(bar)
        
        price = close_price
        current_date += timedelta(days=1)
    
    return bars


def load_data_from_csv(
    filepath: str,
    symbol: str,
) -> List[BarEvent]:
    """
    從 CSV 檔案載入數據
    
    CSV 格式預期：date,open,high,low,close,volume
    
    Args:
        filepath: CSV 檔案路徑
        symbol: 標的代碼
        
    Returns:
        BarEvent 列表
    """
    bars = []
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            try:
                # 解析日期
                date_str = row.get('date') or row.get('Date') or row.get('timestamp')
                timestamp = datetime.strptime(date_str, "%Y-%m-%d")
                
                bar = BarEvent(
                    event_type=EventType.BAR,
                    symbol=symbol,
                    timestamp=timestamp,
                    open=float(row.get('open') or row.get('Open')),
                    high=float(row.get('high') or row.get('High')),
                    low=float(row.get('low') or row.get('Low')),
                    close=float(row.get('close') or row.get('Close')),
                    volume=int(float(row.get('volume') or row.get('Volume') or 0)),
                    timeframe="1d",
                )
                bars.append(bar)
                
            except Exception as e:
                logger.warning(f"解析 CSV 行失敗: {e}")
                continue
    
    # 按時間排序
    bars.sort(key=lambda x: x.timestamp)
    
    return bars


def load_historical_data(
    symbols: List[str],
    start_date: date,
    end_date: date,
    data_dir: Optional[str] = None,
) -> Dict[str, List[BarEvent]]:
    """
    載入歷史數據
    
    Args:
        symbols: 標的列表
        start_date: 開始日期
        end_date: 結束日期
        data_dir: 數據目錄（如果提供則從 CSV 載入）
        
    Returns:
        {symbol: [BarEvent, ...]}
    """
    data = {}
    
    for symbol in symbols:
        if data_dir:
            # 嘗試從 CSV 載入
            csv_path = os.path.join(data_dir, f"{symbol}.csv")
            if os.path.exists(csv_path):
                logger.info(f"從 CSV 載入 {symbol}: {csv_path}")
                bars = load_data_from_csv(csv_path, symbol)
                # 過濾日期範圍
                bars = [b for b in bars if start_date <= b.timestamp.date() <= end_date]
                data[symbol] = bars
                continue
        
        # 使用模擬數據
        logger.info(f"生成模擬數據: {symbol}")
        
        # 根據標的設定不同初始價格
        initial_prices = {
            "AAPL": 150.0,
            "MSFT": 350.0,
            "GOOGL": 140.0,
            "AMZN": 170.0,
            "NVDA": 450.0,
            "TSLA": 250.0,
            "META": 350.0,
        }
        initial_price = initial_prices.get(symbol, 100.0)
        
        data[symbol] = generate_sample_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_price=initial_price,
        )
    
    return data


# ============================================================
# 結果輸出
# ============================================================

def print_result_summary(result: BacktestResult) -> None:
    """輸出結果摘要"""
    print("\n" + "=" * 70)
    print("回測結果摘要")
    print("=" * 70)
    
    print(f"\n策略: {result.strategy_name}")
    
    if result.config:
        print(f"初始資金: ${result.config.initial_capital:,.2f}")
        print(f"回測期間: {result.config.start_date} ~ {result.config.end_date}")
    
    print("\n【績效指標】")
    print(f"  總報酬:      ${result.total_return:,.2f} ({result.total_return_pct:.2%})")
    print(f"  年化報酬:    {result.annualized_return:.2%}")
    print(f"  最大回撤:    {result.max_drawdown_pct:.2%} (${result.max_drawdown:,.2f})")
    print(f"  波動率:      {result.volatility:.2%}")
    
    print("\n【風險調整報酬】")
    print(f"  夏普比率:    {result.sharpe_ratio:.2f}")
    print(f"  索提諾比率:  {result.sortino_ratio:.2f}")
    print(f"  卡爾瑪比率:  {result.calmar_ratio:.2f}")
    
    print("\n【交易統計】")
    print(f"  總交易次數:  {result.total_trades}")
    print(f"  勝率:        {result.win_rate:.2%} ({result.winning_trades}勝 / {result.losing_trades}敗)")
    print(f"  獲利因子:    {result.profit_factor:.2f}")
    
    print("\n【盈虧分析】")
    print(f"  總獲利:      ${result.gross_profit:,.2f}")
    print(f"  總虧損:      ${result.gross_loss:,.2f}")
    print(f"  淨利潤:      ${result.net_profit:,.2f}")
    print(f"  平均獲利:    ${result.avg_win:,.2f}")
    print(f"  平均虧損:    ${result.avg_loss:,.2f}")
    print(f"  平均每筆:    ${result.avg_trade:,.2f}")
    print(f"  最大單筆獲利: ${result.largest_win:,.2f}")
    print(f"  最大單筆虧損: ${result.largest_loss:,.2f}")
    
    print("\n【連續統計】")
    print(f"  最大連勝:    {result.max_consecutive_wins}")
    print(f"  最大連敗:    {result.max_consecutive_losses}")
    
    print("\n【交易成本】")
    print(f"  總手續費:    ${result.total_commission:,.2f}")
    print(f"  總滑價:      ${result.total_slippage:,.2f}")
    
    print("\n" + "=" * 70)


def print_equity_curve(result: BacktestResult, sample_points: int = 20) -> None:
    """輸出權益曲線（簡化版）"""
    if not result.equity_curve:
        return
    
    print("\n【權益曲線】")
    
    # 取樣顯示
    total_points = len(result.equity_curve)
    step = max(1, total_points // sample_points)
    
    max_equity = max(p.equity for p in result.equity_curve)
    scale = 50 / max_equity if max_equity > 0 else 1
    
    for i in range(0, total_points, step):
        point = result.equity_curve[i]
        bar_length = int(point.equity * scale)
        bar = "█" * bar_length
        dd = f" (DD: {point.drawdown_pct:.1%})" if point.drawdown_pct > 0.05 else ""
        print(f"  {point.timestamp.strftime('%Y-%m-%d')} | ${point.equity:>12,.2f} | {bar}{dd}")
    
    # 顯示最後一點
    if total_points % step != 0:
        point = result.equity_curve[-1]
        bar_length = int(point.equity * scale)
        bar = "█" * bar_length
        print(f"  {point.timestamp.strftime('%Y-%m-%d')} | ${point.equity:>12,.2f} | {bar}")


def print_trade_list(result: BacktestResult, max_trades: int = 20) -> None:
    """輸出交易清單"""
    if not result.trades:
        return
    
    print("\n【交易清單】")
    print(f"  {'#':>3} | {'標的':<6} | {'方向':<4} | {'數量':>6} | {'進場價':>10} | {'出場價':>10} | {'盈虧':>12} | {'盈虧%':>8}")
    print("  " + "-" * 85)
    
    trades_to_show = result.trades[:max_trades]
    
    for trade in trades_to_show:
        direction = "買入" if trade.is_long else "賣出"
        exit_price = f"${trade.exit_price:.2f}" if trade.exit_price else "N/A"
        pnl_str = f"${trade.pnl:,.2f}"
        pnl_pct_str = f"{trade.pnl_pct:.2%}"
        
        # 盈虧顏色標記
        pnl_marker = "+" if trade.pnl > 0 else ""
        
        print(
            f"  {trade.trade_id:>3} | {trade.symbol:<6} | {direction:<4} | "
            f"{trade.quantity:>6} | ${trade.entry_price:>9.2f} | {exit_price:>10} | "
            f"{pnl_marker}{pnl_str:>11} | {pnl_pct_str:>8}"
        )
    
    if len(result.trades) > max_trades:
        print(f"  ... 還有 {len(result.trades) - max_trades} 筆交易")


def export_results_to_csv(
    result: BacktestResult,
    output_dir: str,
    prefix: str = "backtest",
) -> None:
    """
    匯出結果到 CSV
    
    Args:
        result: 回測結果
        output_dir: 輸出目錄
        prefix: 檔案前綴
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 匯出績效摘要
    summary_path = os.path.join(output_dir, f"{prefix}_summary_{timestamp}.csv")
    
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["指標", "數值"])
        writer.writerow(["策略名稱", result.strategy_name])
        writer.writerow(["總報酬", f"{result.total_return:.2f}"])
        writer.writerow(["總報酬率", f"{result.total_return_pct:.4f}"])
        writer.writerow(["年化報酬", f"{result.annualized_return:.4f}"])
        writer.writerow(["最大回撤", f"{result.max_drawdown_pct:.4f}"])
        writer.writerow(["波動率", f"{result.volatility:.4f}"])
        writer.writerow(["夏普比率", f"{result.sharpe_ratio:.4f}"])
        writer.writerow(["索提諾比率", f"{result.sortino_ratio:.4f}"])
        writer.writerow(["卡爾瑪比率", f"{result.calmar_ratio:.4f}"])
        writer.writerow(["總交易次數", result.total_trades])
        writer.writerow(["勝率", f"{result.win_rate:.4f}"])
        writer.writerow(["獲利因子", f"{result.profit_factor:.4f}"])
        writer.writerow(["總獲利", f"{result.gross_profit:.2f}"])
        writer.writerow(["總虧損", f"{result.gross_loss:.2f}"])
        writer.writerow(["淨利潤", f"{result.net_profit:.2f}"])
        writer.writerow(["總手續費", f"{result.total_commission:.2f}"])
    
    logger.info(f"績效摘要已匯出: {summary_path}")
    
    # 2. 匯出交易清單
    if result.trades:
        trades_path = os.path.join(output_dir, f"{prefix}_trades_{timestamp}.csv")
        
        with open(trades_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "trade_id", "symbol", "action", "quantity",
                "entry_price", "entry_time", "exit_price", "exit_time",
                "pnl", "pnl_pct", "commission", "slippage"
            ])
            
            for trade in result.trades:
                writer.writerow([
                    trade.trade_id,
                    trade.symbol,
                    trade.action.value,
                    trade.quantity,
                    trade.entry_price,
                    trade.entry_time.isoformat(),
                    trade.exit_price or "",
                    trade.exit_time.isoformat() if trade.exit_time else "",
                    f"{trade.pnl:.2f}",
                    f"{trade.pnl_pct:.4f}",
                    f"{trade.commission:.2f}",
                    f"{trade.slippage:.2f}",
                ])
        
        logger.info(f"交易清單已匯出: {trades_path}")
    
    # 3. 匯出權益曲線
    if result.equity_curve:
        equity_path = os.path.join(output_dir, f"{prefix}_equity_{timestamp}.csv")
        
        with open(equity_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "equity", "cash", "positions_value",
                "drawdown", "drawdown_pct"
            ])
            
            for point in result.equity_curve:
                writer.writerow([
                    point.timestamp.isoformat(),
                    f"{point.equity:.2f}",
                    f"{point.cash:.2f}",
                    f"{point.positions_value:.2f}",
                    f"{point.drawdown:.2f}",
                    f"{point.drawdown_pct:.4f}",
                ])
        
        logger.info(f"權益曲線已匯出: {equity_path}")


# ============================================================
# 主程式
# ============================================================

def parse_args() -> argparse.Namespace:
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description="策略回測啟動腳本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 使用預設設定
  python run_backtest.py
  
  # 指定策略和標的
  python run_backtest.py --strategy sma_cross --symbols AAPL,MSFT,GOOGL
  
  # 指定日期範圍
  python run_backtest.py --start 2023-01-01 --end 2023-12-31
  
  # 匯出結果
  python run_backtest.py --export --output results/
        """,
    )
    
    # 策略
    parser.add_argument(
        "-s", "--strategy",
        type=str,
        default="sma_cross",
        help="策略名稱 (預設: sma_cross)",
    )
    
    # 標的
    parser.add_argument(
        "--symbols",
        type=str,
        default="AAPL,MSFT,GOOGL",
        help="交易標的，用逗號分隔 (預設: AAPL,MSFT,GOOGL)",
    )
    
    # 日期範圍
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="開始日期 (YYYY-MM-DD)，預設一年前",
    )
    
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="結束日期 (YYYY-MM-DD)，預設今天",
    )
    
    # 資金
    parser.add_argument(
        "-c", "--capital",
        type=float,
        default=100000.0,
        help="初始資金 (預設: 100000)",
    )
    
    # 手續費
    parser.add_argument(
        "--commission",
        type=float,
        default=0.001,
        help="手續費率 (預設: 0.001 = 0.1%%)",
    )
    
    # 滑價
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.0005,
        help="滑價率 (預設: 0.0005 = 0.05%%)",
    )
    
    # 數據目錄
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="歷史數據目錄（CSV 檔案）",
    )
    
    # 輸出
    parser.add_argument(
        "--export",
        action="store_true",
        help="匯出結果到 CSV",
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="results",
        help="輸出目錄 (預設: results)",
    )
    
    # 顯示選項
    parser.add_argument(
        "--show-trades",
        action="store_true",
        help="顯示交易清單",
    )
    
    parser.add_argument(
        "--show-equity",
        action="store_true",
        help="顯示權益曲線",
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="詳細輸出",
    )
    
    # 策略參數
    parser.add_argument(
        "--fast-period",
        type=int,
        default=10,
        help="快線週期 (SMA 策略用，預設: 10)",
    )
    
    parser.add_argument(
        "--slow-period",
        type=int,
        default=20,
        help="慢線週期 (SMA 策略用，預設: 20)",
    )
    
    return parser.parse_args()


def main() -> int:
    """主程式"""
    # 解析參數
    args = parse_args()
    
    # 設定日誌
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logger(
        log_dir="logs",
        log_level=log_level,
        console_output=True,
        file_output=False,
    )
    
    print("\n" + "=" * 70)
    print("IB Trading System - 策略回測")
    print("=" * 70)
    
    # 解析參數
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    
    # 解析日期
    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    else:
        start_date = date.today() - timedelta(days=365)
    
    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    else:
        end_date = date.today()
    
    print(f"\n策略:      {args.strategy}")
    print(f"標的:      {', '.join(symbols)}")
    print(f"期間:      {start_date} ~ {end_date}")
    print(f"初始資金:  ${args.capital:,.2f}")
    print(f"手續費率:  {args.commission:.2%}")
    print(f"滑價率:    {args.slippage:.2%}")
    
    # 取得策略註冊表
    registry = get_strategy_registry()
    
    # 檢查策略是否存在
    available_strategies = registry.list_strategies()
    if args.strategy not in available_strategies:
        print(f"\n錯誤: 找不到策略 '{args.strategy}'")
        print(f"可用策略: {', '.join(available_strategies)}")
        return 1
    
    # 載入歷史數據
    print("\n載入歷史數據...")
    data = load_historical_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        data_dir=args.data_dir,
    )
    
    total_bars = sum(len(bars) for bars in data.values())
    print(f"已載入 {len(data)} 個標的，共 {total_bars} 根 K 線")
    
    # 建立回測配置
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital,
        commission_rate=args.commission,
        slippage_rate=args.slippage,
        allow_short=True,
        max_position_size=1000,
        max_positions=10,
    )
    
    # 建立回測引擎
    print("\n初始化回測引擎...")
    engine = BacktestEngine(config)
    
    # 設定策略
    strategy_class = registry.get_strategy(args.strategy)
    
    strategy_config = StrategyConfig(
        name=f"{args.strategy}_backtest",
        symbols=symbols,
        enabled=True,
        params={
            "fast_period": args.fast_period,
            "slow_period": args.slow_period,
        },
    )
    
    engine.set_strategy(strategy_class, strategy_config)
    
    # 載入數據
    engine.load_data(data)
    
    # 執行回測
    print("\n執行回測...")
    print("-" * 70)
    
    result = engine.run()
    
    # 輸出結果
    print_result_summary(result)
    
    if args.show_equity:
        print_equity_curve(result)
    
    if args.show_trades:
        print_trade_list(result)
    
    # 匯出結果
    if args.export:
        print("\n匯出結果...")
        export_results_to_csv(result, args.output, prefix=args.strategy)
        print(f"結果已匯出到: {args.output}/")
    
    # 產生報告
    report_path = os.path.join(args.output, f"{args.strategy}_report.txt")
    os.makedirs(args.output, exist_ok=True)
    report = engine.generate_report(result, report_path)
    print(f"\n報告已儲存: {report_path}")
    
    print("\n回測完成！")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)