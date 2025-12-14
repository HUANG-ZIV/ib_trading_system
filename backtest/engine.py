"""
BacktestEngine 模組 - 回測引擎

提供策略回測功能
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum, auto
from typing import Optional, Dict, List, Any, Callable, Type
import math
import statistics

from core.events import (
    Event,
    EventType,
    BarEvent,
    SignalEvent,
    FillEvent,
    OrderAction,
    OrderType,
)
from core.event_bus import EventBus
from strategies.base import BaseStrategy, StrategyConfig


# 設定 logger
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """回測配置"""
    
    # 時間範圍
    start_date: date = field(default_factory=lambda: date(2023, 1, 1))
    end_date: date = field(default_factory=date.today)
    
    # 資金設定
    initial_capital: float = 100000.0
    commission_rate: float = 0.001      # 手續費率 0.1%
    slippage_rate: float = 0.0005       # 滑價率 0.05%
    
    # 交易設定
    allow_short: bool = True            # 允許做空
    max_position_size: int = 1000       # 最大倉位
    max_positions: int = 10             # 最大持倉數
    
    # 風控設定
    max_drawdown_pct: float = 0.2       # 最大回撤限制 20%
    stop_on_max_drawdown: bool = False  # 達到最大回撤時停止
    
    # 其他
    data_frequency: str = "1d"          # 數據頻率
    benchmark: str = "SPY"              # 基準
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": self.initial_capital,
            "commission_rate": self.commission_rate,
            "slippage_rate": self.slippage_rate,
            "allow_short": self.allow_short,
            "max_position_size": self.max_position_size,
            "max_positions": self.max_positions,
            "max_drawdown_pct": self.max_drawdown_pct,
            "data_frequency": self.data_frequency,
            "benchmark": self.benchmark,
        }


@dataclass
class Trade:
    """交易記錄"""
    
    trade_id: int
    symbol: str
    action: OrderAction
    quantity: int
    entry_price: float
    entry_time: datetime
    
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    
    commission: float = 0.0
    slippage: float = 0.0
    
    pnl: float = 0.0
    pnl_pct: float = 0.0
    
    # 持倉時間
    holding_period: Optional[timedelta] = None
    
    @property
    def is_closed(self) -> bool:
        """是否已平倉"""
        return self.exit_price is not None
    
    @property
    def is_long(self) -> bool:
        """是否做多"""
        return self.action == OrderAction.BUY
    
    @property
    def is_short(self) -> bool:
        """是否做空"""
        return self.action == OrderAction.SELL
    
    @property
    def is_winner(self) -> bool:
        """是否獲利"""
        return self.pnl > 0
    
    def close(self, exit_price: float, exit_time: datetime) -> float:
        """平倉並計算盈虧"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.holding_period = exit_time - self.entry_time
        
        # 計算盈虧
        if self.is_long:
            self.pnl = (exit_price - self.entry_price) * self.quantity
        else:
            self.pnl = (self.entry_price - exit_price) * self.quantity
        
        # 扣除成本
        self.pnl -= (self.commission + self.slippage)
        
        # 計算百分比
        entry_value = self.entry_price * self.quantity
        if entry_value > 0:
            self.pnl_pct = self.pnl / entry_value
        
        return self.pnl
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "action": self.action.value,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "exit_price": self.exit_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "commission": self.commission,
            "slippage": self.slippage,
            "pnl": self.pnl,
            "pnl_pct": f"{self.pnl_pct:.2%}",
            "holding_period": str(self.holding_period) if self.holding_period else None,
            "is_winner": self.is_winner,
        }


@dataclass
class EquityPoint:
    """權益曲線點"""
    
    timestamp: datetime
    equity: float
    cash: float
    positions_value: float
    drawdown: float = 0.0
    drawdown_pct: float = 0.0


@dataclass
class BacktestResult:
    """回測結果"""
    
    # 基本資訊
    strategy_name: str = ""
    config: Optional[BacktestConfig] = None
    
    # 績效指標
    total_return: float = 0.0           # 總報酬
    total_return_pct: float = 0.0       # 總報酬率
    annualized_return: float = 0.0      # 年化報酬
    
    # 風險指標
    max_drawdown: float = 0.0           # 最大回撤金額
    max_drawdown_pct: float = 0.0       # 最大回撤百分比
    volatility: float = 0.0             # 波動率
    
    # 風險調整報酬
    sharpe_ratio: float = 0.0           # 夏普比率
    sortino_ratio: float = 0.0          # 索提諾比率
    calmar_ratio: float = 0.0           # 卡爾瑪比率
    
    # 交易統計
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # 盈虧統計
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    profit_factor: float = 0.0          # 獲利因子
    
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # 連續統計
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    # 持倉統計
    avg_holding_period: Optional[timedelta] = None
    
    # 成本
    total_commission: float = 0.0
    total_slippage: float = 0.0
    
    # 交易清單
    trades: List[Trade] = field(default_factory=list)
    
    # 權益曲線
    equity_curve: List[EquityPoint] = field(default_factory=list)
    
    # 時間資訊
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[timedelta] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            "strategy_name": self.strategy_name,
            "performance": {
                "total_return": self.total_return,
                "total_return_pct": f"{self.total_return_pct:.2%}",
                "annualized_return": f"{self.annualized_return:.2%}",
            },
            "risk": {
                "max_drawdown": self.max_drawdown,
                "max_drawdown_pct": f"{self.max_drawdown_pct:.2%}",
                "volatility": f"{self.volatility:.2%}",
            },
            "risk_adjusted": {
                "sharpe_ratio": f"{self.sharpe_ratio:.2f}",
                "sortino_ratio": f"{self.sortino_ratio:.2f}",
                "calmar_ratio": f"{self.calmar_ratio:.2f}",
            },
            "trades": {
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "win_rate": f"{self.win_rate:.2%}",
            },
            "profit_loss": {
                "gross_profit": self.gross_profit,
                "gross_loss": self.gross_loss,
                "net_profit": self.net_profit,
                "profit_factor": f"{self.profit_factor:.2f}",
                "avg_win": self.avg_win,
                "avg_loss": self.avg_loss,
                "avg_trade": self.avg_trade,
                "largest_win": self.largest_win,
                "largest_loss": self.largest_loss,
            },
            "streaks": {
                "max_consecutive_wins": self.max_consecutive_wins,
                "max_consecutive_losses": self.max_consecutive_losses,
            },
            "costs": {
                "total_commission": self.total_commission,
                "total_slippage": self.total_slippage,
            },
            "duration": str(self.duration) if self.duration else None,
        }
    
    def summary(self) -> str:
        """產生摘要字串"""
        return (
            f"Strategy: {self.strategy_name}\n"
            f"Total Return: {self.total_return_pct:.2%} (${self.total_return:,.2f})\n"
            f"Annualized Return: {self.annualized_return:.2%}\n"
            f"Max Drawdown: {self.max_drawdown_pct:.2%}\n"
            f"Sharpe Ratio: {self.sharpe_ratio:.2f}\n"
            f"Win Rate: {self.win_rate:.2%} ({self.winning_trades}/{self.total_trades})\n"
            f"Profit Factor: {self.profit_factor:.2f}\n"
        )


class BacktestEngine:
    """
    回測引擎
    
    模擬交易環境進行策略回測
    
    使用方式:
        engine = BacktestEngine(config)
        engine.set_strategy(MyStrategy)
        engine.load_data(data)
        result = engine.run()
        
        print(result.summary())
        engine.generate_report("report.html")
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        初始化回測引擎
        
        Args:
            config: 回測配置
        """
        self._config = config or BacktestConfig()
        
        # 事件總線（用於策略通訊）
        self._event_bus = EventBus()
        
        # 策略
        self._strategy: Optional[BaseStrategy] = None
        self._strategy_class: Optional[Type[BaseStrategy]] = None
        self._strategy_config: Optional[StrategyConfig] = None
        
        # 數據
        self._data: Dict[str, List[BarEvent]] = {}  # symbol -> bars
        self._current_bar: Dict[str, BarEvent] = {}
        self._current_time: Optional[datetime] = None
        
        # 帳戶狀態
        self._cash = self._config.initial_capital
        self._positions: Dict[str, int] = {}  # symbol -> quantity
        self._position_costs: Dict[str, float] = {}  # symbol -> avg cost
        
        # 交易記錄
        self._trades: List[Trade] = []
        self._open_trades: Dict[str, Trade] = {}  # symbol -> open trade
        self._trade_counter = 0
        
        # 權益曲線
        self._equity_curve: List[EquityPoint] = []
        self._peak_equity = self._config.initial_capital
        
        # 每日報酬（用於計算夏普比率等）
        self._daily_returns: List[float] = []
        self._last_equity = self._config.initial_capital
        
        # 訂閱信號事件
        self._event_bus.subscribe(EventType.SIGNAL, self._on_signal)
        
        logger.info(f"BacktestEngine 初始化: capital={self._config.initial_capital}")
    
    # ========== 配置 ==========
    
    def set_strategy(
        self,
        strategy_class: Type[BaseStrategy],
        config: Optional[StrategyConfig] = None,
        **kwargs,
    ) -> None:
        """
        設定回測策略
        
        Args:
            strategy_class: 策略類
            config: 策略配置
            **kwargs: 額外參數
        """
        self._strategy_class = strategy_class
        self._strategy_config = config
        self._strategy_kwargs = kwargs
        
        logger.info(f"設定策略: {strategy_class.__name__}")
    
    def load_data(
        self,
        data: Dict[str, List[BarEvent]],
    ) -> None:
        """
        載入歷史數據
        
        Args:
            data: 歷史數據 {symbol: [BarEvent, ...]}
        """
        self._data = data
        
        # 統計
        total_bars = sum(len(bars) for bars in data.values())
        symbols = list(data.keys())
        
        logger.info(f"載入數據: {len(symbols)} 個標的, {total_bars} 根 K 線")
    
    def load_data_from_list(
        self,
        symbol: str,
        bars: List[BarEvent],
    ) -> None:
        """載入單一標的數據"""
        self._data[symbol] = bars
        logger.info(f"載入 {symbol}: {len(bars)} 根 K 線")
    
    # ========== 執行回測 ==========
    
    def run(self) -> BacktestResult:
        """
        執行回測
        
        Returns:
            BacktestResult 回測結果
        """
        logger.info("開始回測...")
        start_time = datetime.now()
        
        # 重置狀態
        self._reset()
        
        # 建立策略實例
        self._create_strategy()
        
        # 初始化策略
        if self._strategy:
            self._strategy.initialize()
            self._strategy.start()
        
        # 取得所有時間點並排序
        all_times = self._get_sorted_timestamps()
        
        if not all_times:
            logger.warning("沒有數據可回測")
            return BacktestResult()
        
        logger.info(f"回測時間範圍: {all_times[0]} ~ {all_times[-1]}")
        logger.info(f"總共 {len(all_times)} 個時間點")
        
        # 遍歷每個時間點
        for i, timestamp in enumerate(all_times):
            self._current_time = timestamp
            
            # 更新當前 bar
            self._update_current_bars(timestamp)
            
            # 發送 bar 事件到策略
            for symbol, bar in self._current_bar.items():
                if bar.timestamp == timestamp:
                    if self._strategy:
                        self._strategy.on_bar(bar)
            
            # 記錄每日權益
            self._record_equity()
            
            # 檢查最大回撤
            if self._config.stop_on_max_drawdown:
                if self._check_max_drawdown():
                    logger.warning("達到最大回撤限制，停止回測")
                    break
            
            # 進度報告
            if (i + 1) % 1000 == 0:
                logger.debug(f"回測進度: {i + 1}/{len(all_times)}")
        
        # 強制平倉所有持倉
        self._close_all_positions()
        
        # 停止策略
        if self._strategy:
            self._strategy.stop()
        
        end_time = datetime.now()
        
        # 計算績效指標
        result = self._calculate_result(start_time, end_time)
        
        logger.info(f"回測完成，耗時 {end_time - start_time}")
        logger.info(f"總報酬: {result.total_return_pct:.2%}")
        
        return result
    
    def _reset(self) -> None:
        """重置回測狀態"""
        self._cash = self._config.initial_capital
        self._positions.clear()
        self._position_costs.clear()
        self._trades.clear()
        self._open_trades.clear()
        self._trade_counter = 0
        self._equity_curve.clear()
        self._daily_returns.clear()
        self._peak_equity = self._config.initial_capital
        self._last_equity = self._config.initial_capital
        self._current_bar.clear()
        self._current_time = None
    
    def _create_strategy(self) -> None:
        """建立策略實例"""
        if self._strategy_class is None:
            logger.warning("未設定策略")
            return
        
        symbols = list(self._data.keys())
        
        self._strategy = self._strategy_class(
            symbols=symbols,
            config=self._strategy_config,
            event_bus=self._event_bus,
            **self._strategy_kwargs,
        )
    
    def _get_sorted_timestamps(self) -> List[datetime]:
        """取得所有時間點並排序"""
        timestamps = set()
        
        for bars in self._data.values():
            for bar in bars:
                timestamps.add(bar.timestamp)
        
        return sorted(timestamps)
    
    def _update_current_bars(self, timestamp: datetime) -> None:
        """更新當前 bar"""
        for symbol, bars in self._data.items():
            for bar in bars:
                if bar.timestamp == timestamp:
                    self._current_bar[symbol] = bar
                    break
    
    # ========== 信號處理 ==========
    
    def _on_signal(self, signal: SignalEvent) -> None:
        """處理策略信號"""
        symbol = signal.symbol
        action = signal.action
        quantity = signal.quantity
        
        # 取得當前價格
        current_bar = self._current_bar.get(symbol)
        if current_bar is None:
            logger.warning(f"無法取得 {symbol} 的當前價格")
            return
        
        # 使用收盤價或建議價格
        price = signal.suggested_price or current_bar.close
        
        # 模擬訂單執行
        self._execute_order(symbol, action, quantity, price)
    
    def _execute_order(
        self,
        symbol: str,
        action: OrderAction,
        quantity: int,
        price: float,
    ) -> Optional[FillEvent]:
        """
        模擬訂單執行
        
        Args:
            symbol: 標的代碼
            action: 交易動作
            quantity: 數量
            price: 價格
            
        Returns:
            FillEvent 或 None
        """
        if quantity <= 0:
            return None
        
        # 計算滑價
        slippage = price * self._config.slippage_rate
        if action == OrderAction.BUY:
            fill_price = price + slippage
        else:
            fill_price = price - slippage
        
        # 計算手續費
        commission = fill_price * quantity * self._config.commission_rate
        
        # 計算所需資金
        required_cash = fill_price * quantity + commission
        
        # 檢查資金是否足夠（買入時）
        if action == OrderAction.BUY:
            if required_cash > self._cash:
                # 調整數量
                affordable = int((self._cash - commission) / fill_price)
                if affordable <= 0:
                    logger.debug(f"資金不足，無法買入 {symbol}")
                    return None
                quantity = affordable
                required_cash = fill_price * quantity + commission
        
        # 檢查做空限制
        if action == OrderAction.SELL:
            current_position = self._positions.get(symbol, 0)
            if current_position <= 0 and not self._config.allow_short:
                logger.debug(f"不允許做空 {symbol}")
                return None
        
        # 執行交易
        if action == OrderAction.BUY:
            self._cash -= required_cash
            old_position = self._positions.get(symbol, 0)
            old_cost = self._position_costs.get(symbol, 0)
            
            # 更新持倉
            new_position = old_position + quantity
            if new_position != 0:
                # 計算新的平均成本
                total_cost = old_cost * old_position + fill_price * quantity
                self._position_costs[symbol] = total_cost / new_position
            
            self._positions[symbol] = new_position
            
            # 建立交易記錄
            if old_position <= 0:
                self._open_trade(symbol, action, quantity, fill_price, commission, slippage)
            
        else:  # SELL
            self._cash += fill_price * quantity - commission
            old_position = self._positions.get(symbol, 0)
            new_position = old_position - quantity
            self._positions[symbol] = new_position
            
            # 平倉計算
            if old_position > 0:
                close_quantity = min(quantity, old_position)
                self._close_trade(symbol, fill_price, close_quantity)
                
                # 如果反向開倉
                if new_position < 0 and self._config.allow_short:
                    self._open_trade(symbol, action, -new_position, fill_price, commission, slippage)
            else:
                # 做空開倉
                self._open_trade(symbol, action, quantity, fill_price, commission, slippage)
        
        # 清理零持倉
        if self._positions.get(symbol, 0) == 0:
            self._positions.pop(symbol, None)
            self._position_costs.pop(symbol, None)
        
        logger.debug(
            f"執行 {action.value} {symbol} x{quantity} @ {fill_price:.2f}, "
            f"commission={commission:.2f}"
        )
        
        return FillEvent(
            event_type=EventType.FILL,
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=fill_price,
            commission=commission,
            order_id=self._trade_counter,
        )
    
    def _open_trade(
        self,
        symbol: str,
        action: OrderAction,
        quantity: int,
        price: float,
        commission: float,
        slippage: float,
    ) -> None:
        """開倉"""
        self._trade_counter += 1
        
        trade = Trade(
            trade_id=self._trade_counter,
            symbol=symbol,
            action=action,
            quantity=quantity,
            entry_price=price,
            entry_time=self._current_time,
            commission=commission,
            slippage=slippage,
        )
        
        self._open_trades[symbol] = trade
    
    def _close_trade(
        self,
        symbol: str,
        exit_price: float,
        quantity: int,
    ) -> None:
        """平倉"""
        trade = self._open_trades.get(symbol)
        if trade is None:
            return
        
        # 計算盈虧
        trade.close(exit_price, self._current_time)
        
        # 移動到已完成交易
        self._trades.append(trade)
        del self._open_trades[symbol]
    
    def _close_all_positions(self) -> None:
        """平倉所有持倉"""
        for symbol in list(self._positions.keys()):
            position = self._positions[symbol]
            if position == 0:
                continue
            
            current_bar = self._current_bar.get(symbol)
            if current_bar is None:
                continue
            
            if position > 0:
                self._execute_order(symbol, OrderAction.SELL, position, current_bar.close)
            else:
                self._execute_order(symbol, OrderAction.BUY, -position, current_bar.close)
    
    # ========== 權益追蹤 ==========
    
    def _record_equity(self) -> None:
        """記錄權益"""
        equity = self._calculate_equity()
        
        # 計算回撤
        if equity > self._peak_equity:
            self._peak_equity = equity
        
        drawdown = self._peak_equity - equity
        drawdown_pct = drawdown / self._peak_equity if self._peak_equity > 0 else 0
        
        # 計算持倉價值
        positions_value = equity - self._cash
        
        point = EquityPoint(
            timestamp=self._current_time,
            equity=equity,
            cash=self._cash,
            positions_value=positions_value,
            drawdown=drawdown,
            drawdown_pct=drawdown_pct,
        )
        
        self._equity_curve.append(point)
        
        # 記錄每日報酬
        if self._last_equity > 0:
            daily_return = (equity - self._last_equity) / self._last_equity
            self._daily_returns.append(daily_return)
        
        self._last_equity = equity
    
    def _calculate_equity(self) -> float:
        """計算當前權益"""
        equity = self._cash
        
        for symbol, quantity in self._positions.items():
            current_bar = self._current_bar.get(symbol)
            if current_bar:
                equity += current_bar.close * quantity
        
        return equity
    
    def _check_max_drawdown(self) -> bool:
        """檢查是否達到最大回撤"""
        if not self._equity_curve:
            return False
        
        latest = self._equity_curve[-1]
        return latest.drawdown_pct >= self._config.max_drawdown_pct
    
    # ========== 計算結果 ==========
    
    def _calculate_result(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> BacktestResult:
        """計算回測結果"""
        result = BacktestResult(
            strategy_name=self._strategy_class.__name__ if self._strategy_class else "",
            config=self._config,
            trades=self._trades.copy(),
            equity_curve=self._equity_curve.copy(),
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
        )
        
        # 基本績效
        final_equity = self._calculate_equity()
        result.total_return = final_equity - self._config.initial_capital
        result.total_return_pct = result.total_return / self._config.initial_capital
        
        # 年化報酬
        if self._equity_curve:
            days = (self._equity_curve[-1].timestamp - self._equity_curve[0].timestamp).days
            if days > 0:
                result.annualized_return = (1 + result.total_return_pct) ** (365 / days) - 1
        
        # 最大回撤
        if self._equity_curve:
            result.max_drawdown = max(p.drawdown for p in self._equity_curve)
            result.max_drawdown_pct = max(p.drawdown_pct for p in self._equity_curve)
        
        # 波動率和風險調整報酬
        if len(self._daily_returns) > 1:
            result.volatility = statistics.stdev(self._daily_returns) * math.sqrt(252)
            
            # 夏普比率（假設無風險利率為 0）
            avg_return = statistics.mean(self._daily_returns) * 252
            if result.volatility > 0:
                result.sharpe_ratio = avg_return / result.volatility
            
            # 索提諾比率
            negative_returns = [r for r in self._daily_returns if r < 0]
            if negative_returns:
                downside_deviation = statistics.stdev(negative_returns) * math.sqrt(252)
                if downside_deviation > 0:
                    result.sortino_ratio = avg_return / downside_deviation
            
            # 卡爾瑪比率
            if result.max_drawdown_pct > 0:
                result.calmar_ratio = result.annualized_return / result.max_drawdown_pct
        
        # 交易統計
        result.total_trades = len(self._trades)
        result.winning_trades = sum(1 for t in self._trades if t.is_winner)
        result.losing_trades = result.total_trades - result.winning_trades
        
        if result.total_trades > 0:
            result.win_rate = result.winning_trades / result.total_trades
        
        # 盈虧統計
        profits = [t.pnl for t in self._trades if t.pnl > 0]
        losses = [t.pnl for t in self._trades if t.pnl < 0]
        
        result.gross_profit = sum(profits)
        result.gross_loss = sum(losses)
        result.net_profit = result.gross_profit + result.gross_loss
        
        if result.gross_loss != 0:
            result.profit_factor = abs(result.gross_profit / result.gross_loss)
        
        if profits:
            result.avg_win = statistics.mean(profits)
            result.largest_win = max(profits)
        
        if losses:
            result.avg_loss = statistics.mean(losses)
            result.largest_loss = min(losses)
        
        if self._trades:
            result.avg_trade = result.net_profit / len(self._trades)
        
        # 連續統計
        result.max_consecutive_wins = self._calculate_max_consecutive(True)
        result.max_consecutive_losses = self._calculate_max_consecutive(False)
        
        # 持倉時間
        holding_periods = [t.holding_period for t in self._trades if t.holding_period]
        if holding_periods:
            avg_seconds = statistics.mean(p.total_seconds() for p in holding_periods)
            result.avg_holding_period = timedelta(seconds=avg_seconds)
        
        # 成本
        result.total_commission = sum(t.commission for t in self._trades)
        result.total_slippage = sum(t.slippage for t in self._trades)
        
        return result
    
    def _calculate_max_consecutive(self, wins: bool) -> int:
        """計算最大連續勝/敗"""
        max_count = 0
        current_count = 0
        
        for trade in self._trades:
            if trade.is_winner == wins:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count
    
    # ========== 報告 ==========
    
    def generate_report(
        self,
        result: BacktestResult,
        output_path: Optional[str] = None,
    ) -> str:
        """
        產生回測報告
        
        Args:
            result: 回測結果
            output_path: 輸出路徑，None 則只返回字串
            
        Returns:
            報告字串
        """
        report = []
        report.append("=" * 60)
        report.append("BACKTEST REPORT")
        report.append("=" * 60)
        report.append("")
        
        # 策略資訊
        report.append("【策略資訊】")
        report.append(f"策略名稱: {result.strategy_name}")
        if result.config:
            report.append(f"初始資金: ${result.config.initial_capital:,.2f}")
            report.append(f"回測期間: {result.config.start_date} ~ {result.config.end_date}")
        report.append("")
        
        # 績效摘要
        report.append("【績效摘要】")
        report.append(f"總報酬: ${result.total_return:,.2f} ({result.total_return_pct:.2%})")
        report.append(f"年化報酬: {result.annualized_return:.2%}")
        report.append(f"最大回撤: {result.max_drawdown_pct:.2%}")
        report.append(f"波動率: {result.volatility:.2%}")
        report.append("")
        
        # 風險調整報酬
        report.append("【風險調整報酬】")
        report.append(f"夏普比率: {result.sharpe_ratio:.2f}")
        report.append(f"索提諾比率: {result.sortino_ratio:.2f}")
        report.append(f"卡爾瑪比率: {result.calmar_ratio:.2f}")
        report.append("")
        
        # 交易統計
        report.append("【交易統計】")
        report.append(f"總交易次數: {result.total_trades}")
        report.append(f"勝率: {result.win_rate:.2%} ({result.winning_trades}勝 / {result.losing_trades}敗)")
        report.append(f"獲利因子: {result.profit_factor:.2f}")
        report.append(f"平均獲利: ${result.avg_win:,.2f}")
        report.append(f"平均虧損: ${result.avg_loss:,.2f}")
        report.append(f"平均每筆: ${result.avg_trade:,.2f}")
        report.append(f"最大單筆獲利: ${result.largest_win:,.2f}")
        report.append(f"最大單筆虧損: ${result.largest_loss:,.2f}")
        report.append(f"最大連勝: {result.max_consecutive_wins}")
        report.append(f"最大連敗: {result.max_consecutive_losses}")
        report.append("")
        
        # 成本
        report.append("【交易成本】")
        report.append(f"總手續費: ${result.total_commission:,.2f}")
        report.append(f"總滑價: ${result.total_slippage:,.2f}")
        report.append("")
        
        report.append("=" * 60)
        
        report_str = "\n".join(report)
        
        # 輸出到檔案
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_str)
            logger.info(f"報告已輸出到: {output_path}")
        
        return report_str