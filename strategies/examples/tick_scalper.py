"""
Tick Scalper Strategy - 高頻 Tick 剝頭皮策略

基於 Tick 級別數據的快速進出場策略範例
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Deque

from strategies.base import BaseStrategy, StrategyConfig, Position
from core.events import TickEvent, BarEvent, SignalEvent, OrderAction, OrderType


logger = logging.getLogger(__name__)


@dataclass
class TickScalperConfig(StrategyConfig):
    """Tick 剝頭皮策略配置"""
    
    # 價差觸發條件
    spread_threshold: float = 0.01      # 價差閾值（美元）
    tight_spread_trigger: float = 0.005  # 緊密價差觸發
    
    # 成交量條件
    tick_count_trigger: int = 10         # 連續 tick 數量觸發
    volume_spike_ratio: float = 2.0      # 成交量突增比率
    
    # 價格動量
    price_momentum_ticks: int = 5        # 價格動量計算的 tick 數
    min_momentum: float = 0.005          # 最小動量百分比
    
    # 交易設定
    quantity: int = 100                  # 每次交易數量
    profit_target: float = 0.02          # 目標利潤（美元/股）
    stop_loss: float = 0.01              # 止損（美元/股）
    
    # 持倉時間限制
    max_hold_seconds: int = 60           # 最大持倉時間（秒）
    
    # 風控
    max_trades_per_minute: int = 5       # 每分鐘最大交易次數
    cooldown_seconds: int = 5            # 交易冷卻時間（秒）


@dataclass
class TickData:
    """單一 Tick 數據"""
    
    timestamp: datetime
    bid: float
    ask: float
    last: float
    bid_size: int
    ask_size: int
    last_size: int
    
    @property
    def spread(self) -> float:
        """買賣價差"""
        return self.ask - self.bid
    
    @property
    def mid(self) -> float:
        """中間價"""
        return (self.bid + self.ask) / 2
    
    @property
    def imbalance(self) -> float:
        """訂單不平衡度 (-1 到 1)"""
        total = self.bid_size + self.ask_size
        if total == 0:
            return 0.0
        return (self.bid_size - self.ask_size) / total


@dataclass
class SymbolState:
    """單一標的的狀態"""
    
    symbol: str
    
    # Tick 歷史
    ticks: Deque[TickData] = field(default_factory=lambda: deque(maxlen=100))
    
    # 價格追蹤
    prices: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    volumes: Deque[int] = field(default_factory=lambda: deque(maxlen=50))
    
    # 當前狀態
    current_bid: float = 0.0
    current_ask: float = 0.0
    current_spread: float = 0.0
    avg_spread: float = 0.0
    avg_volume: float = 0.0
    
    # 動量指標
    price_momentum: float = 0.0
    volume_momentum: float = 0.0
    order_imbalance: float = 0.0
    
    # 交易狀態
    entry_price: Optional[float] = None
    entry_time: Optional[datetime] = None
    position_side: Optional[str] = None  # "LONG" or "SHORT"
    
    # 統計
    tick_count: int = 0
    trade_count_this_minute: int = 0
    last_trade_time: Optional[datetime] = None
    last_minute_reset: datetime = field(default_factory=datetime.now)
    
    def add_tick(self, tick: TickData) -> None:
        """添加 Tick 數據"""
        self.ticks.append(tick)
        self.prices.append(tick.last)
        self.volumes.append(tick.last_size)
        self.tick_count += 1
        
        # 更新當前狀態
        self.current_bid = tick.bid
        self.current_ask = tick.ask
        self.current_spread = tick.spread
        
        # 更新平均值
        if len(self.ticks) > 5:
            recent_spreads = [t.spread for t in list(self.ticks)[-20:]]
            self.avg_spread = sum(recent_spreads) / len(recent_spreads)
            
            recent_volumes = list(self.volumes)[-20:]
            self.avg_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 0
        
        # 更新訂單不平衡度
        self.order_imbalance = tick.imbalance
    
    def calculate_momentum(self, lookback: int = 5) -> float:
        """計算價格動量"""
        if len(self.prices) < lookback:
            return 0.0
        
        prices = list(self.prices)[-lookback:]
        if prices[0] == 0:
            return 0.0
        
        self.price_momentum = (prices[-1] - prices[0]) / prices[0]
        return self.price_momentum
    
    def calculate_volume_momentum(self) -> float:
        """計算成交量動量"""
        if self.avg_volume == 0 or len(self.volumes) < 5:
            return 1.0
        
        recent_avg = sum(list(self.volumes)[-5:]) / 5
        self.volume_momentum = recent_avg / self.avg_volume if self.avg_volume > 0 else 1.0
        return self.volume_momentum
    
    def reset_minute_counter(self) -> None:
        """重置每分鐘計數器"""
        now = datetime.now()
        if (now - self.last_minute_reset).seconds >= 60:
            self.trade_count_this_minute = 0
            self.last_minute_reset = now


class TickScalperStrategy(BaseStrategy):
    """
    Tick 級別剝頭皮策略
    
    策略邏輯：
    1. 監控 bid-ask spread 的變化
    2. 分析成交量突增信號
    3. 利用訂單不平衡度判斷方向
    4. 快速進出場，設定嚴格止損
    5. 強制最大持倉時間限制
    
    使用方式:
        config = TickScalperConfig(
            spread_threshold=0.01,
            tick_count_trigger=10,
            symbols=["AAPL"],
        )
        strategy = TickScalperStrategy(config=config)
    """
    
    def __init__(
        self,
        strategy_id: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        config: Optional[TickScalperConfig] = None,
        spread_threshold: float = 0.01,
        tick_count_trigger: int = 10,
        quantity: int = 100,
        **kwargs,
    ):
        """
        初始化策略
        
        Args:
            strategy_id: 策略 ID
            symbols: 標的列表
            config: 策略配置
            spread_threshold: 價差閾值
            tick_count_trigger: Tick 數量觸發
            quantity: 交易數量
        """
        # 建立配置
        if config is None:
            config = TickScalperConfig(
                spread_threshold=spread_threshold,
                tick_count_trigger=tick_count_trigger,
                quantity=quantity,
                symbols=symbols or [],
            )
        
        super().__init__(
            strategy_id=strategy_id or "tick_scalper",
            symbols=symbols,
            config=config,
            **kwargs,
        )
        
        # 策略參數
        self._spread_threshold = config.spread_threshold
        self._tight_spread_trigger = config.tight_spread_trigger
        self._tick_count_trigger = config.tick_count_trigger
        self._volume_spike_ratio = config.volume_spike_ratio
        self._price_momentum_ticks = config.price_momentum_ticks
        self._min_momentum = config.min_momentum
        self._quantity = config.quantity
        self._profit_target = config.profit_target
        self._stop_loss = config.stop_loss
        self._max_hold_seconds = config.max_hold_seconds
        self._max_trades_per_minute = config.max_trades_per_minute
        self._cooldown_seconds = config.cooldown_seconds
        
        # 每個標的的狀態
        self._symbol_states: Dict[str, SymbolState] = {}
        
        self._logger.info(
            f"TickScalperStrategy 初始化: spread_threshold={self._spread_threshold}, "
            f"tick_trigger={self._tick_count_trigger}, qty={self._quantity}"
        )
    
    # ========== 生命週期 ==========
    
    def initialize(self) -> None:
        """初始化策略"""
        super().initialize()
        
        for symbol in self._symbols:
            self._symbol_states[symbol] = SymbolState(symbol=symbol)
    
    def on_start(self) -> None:
        """策略啟動"""
        super().on_start()
        self._logger.info(
            f"TickScalperStrategy 啟動，監控標的: {list(self._symbols)}"
        )
    
    # ========== 事件處理 ==========
    
    def on_bar(self, event: BarEvent) -> Optional[SignalEvent]:
        """
        處理 Bar 事件（此策略主要使用 Tick）
        
        Bar 事件用於檢查持倉時間限制
        """
        self._stats.bar_count += 1
        
        # 檢查所有持倉的時間限制
        return self._check_position_timeout(event.symbol)
    
    def on_tick(self, event: TickEvent) -> Optional[SignalEvent]:
        """
        處理 Tick 事件（核心邏輯）
        
        Args:
            event: TickEvent
            
        Returns:
            SignalEvent 或 None
        """
        symbol = event.symbol
        self._stats.tick_count += 1
        
        # 驗證數據
        if not self._validate_tick(event):
            return None
        
        # 取得或建立標的狀態
        if symbol not in self._symbol_states:
            self._symbol_states[symbol] = SymbolState(symbol=symbol)
        
        state = self._symbol_states[symbol]
        
        # 建立 TickData
        tick = TickData(
            timestamp=event.timestamp,
            bid=event.bid or 0,
            ask=event.ask or 0,
            last=event.last or 0,
            bid_size=event.bid_size or 0,
            ask_size=event.ask_size or 0,
            last_size=event.last_size or 0,
        )
        
        # 更新狀態
        state.add_tick(tick)
        state.calculate_momentum(self._price_momentum_ticks)
        state.calculate_volume_momentum()
        state.reset_minute_counter()
        
        # 更新持倉價格
        if symbol in self._positions:
            self._positions[symbol].update_price(tick.last)
        
        # 檢查是否需要平倉
        exit_signal = self._check_exit_conditions(state, tick)
        if exit_signal:
            return exit_signal
        
        # 檢查是否可以開倉
        if self._can_trade(state):
            entry_signal = self._check_entry_conditions(state, tick)
            if entry_signal:
                return entry_signal
        
        return None
    
    def _validate_tick(self, event: TickEvent) -> bool:
        """驗證 Tick 數據"""
        if event.bid is None or event.ask is None:
            return False
        if event.bid <= 0 or event.ask <= 0:
            return False
        if event.ask < event.bid:
            return False
        return True
    
    # ========== 交易條件檢查 ==========
    
    def _can_trade(self, state: SymbolState) -> bool:
        """檢查是否可以交易"""
        # 檢查是否已有持倉
        if self.has_position(state.symbol):
            return False
        
        # 檢查每分鐘交易次數
        if state.trade_count_this_minute >= self._max_trades_per_minute:
            return False
        
        # 檢查冷卻時間
        if state.last_trade_time:
            elapsed = (datetime.now() - state.last_trade_time).total_seconds()
            if elapsed < self._cooldown_seconds:
                return False
        
        # 檢查是否有足夠數據
        if len(state.ticks) < self._tick_count_trigger:
            return False
        
        return True
    
    def _check_entry_conditions(
        self,
        state: SymbolState,
        tick: TickData,
    ) -> Optional[SignalEvent]:
        """
        檢查進場條件
        
        進場信號條件：
        1. 價差收窄到閾值以下
        2. 成交量突增
        3. 訂單不平衡度傾斜
        4. 價格動量方向一致
        """
        symbol = state.symbol
        
        # 條件 1: 價差收窄
        spread_tight = state.current_spread <= self._tight_spread_trigger
        spread_narrowing = state.current_spread < state.avg_spread * 0.8
        
        # 條件 2: 成交量突增
        volume_spike = state.volume_momentum >= self._volume_spike_ratio
        
        # 條件 3: 訂單不平衡度
        bullish_imbalance = state.order_imbalance > 0.3
        bearish_imbalance = state.order_imbalance < -0.3
        
        # 條件 4: 價格動量
        bullish_momentum = state.price_momentum > self._min_momentum
        bearish_momentum = state.price_momentum < -self._min_momentum
        
        # 綜合判斷 - 做多信號
        if (spread_tight or spread_narrowing) and volume_spike:
            if bullish_imbalance and bullish_momentum:
                self._logger.info(
                    f"{symbol} 做多信號: spread={tick.spread:.4f}, "
                    f"imbalance={state.order_imbalance:.2f}, "
                    f"momentum={state.price_momentum:.4f}"
                )
                
                # 更新狀態
                state.entry_price = tick.ask  # 以 ask 價買入
                state.entry_time = datetime.now()
                state.position_side = "LONG"
                state.trade_count_this_minute += 1
                state.last_trade_time = datetime.now()
                
                return self.emit_signal(
                    symbol=symbol,
                    action=OrderAction.BUY,
                    quantity=self._quantity,
                    price=tick.ask,
                    order_type=OrderType.LIMIT,
                    stop_loss=tick.ask - self._stop_loss,
                    take_profit=tick.ask + self._profit_target,
                    strength=min(abs(state.order_imbalance) + abs(state.price_momentum) * 10, 1.0),
                    reason=f"Scalp Long: spread={tick.spread:.4f}, imb={state.order_imbalance:.2f}",
                    metadata={
                        "spread": tick.spread,
                        "imbalance": state.order_imbalance,
                        "momentum": state.price_momentum,
                        "volume_momentum": state.volume_momentum,
                    },
                )
        
        # 綜合判斷 - 做空信號
        if (spread_tight or spread_narrowing) and volume_spike:
            if bearish_imbalance and bearish_momentum:
                self._logger.info(
                    f"{symbol} 做空信號: spread={tick.spread:.4f}, "
                    f"imbalance={state.order_imbalance:.2f}, "
                    f"momentum={state.price_momentum:.4f}"
                )
                
                # 更新狀態
                state.entry_price = tick.bid  # 以 bid 價賣出
                state.entry_time = datetime.now()
                state.position_side = "SHORT"
                state.trade_count_this_minute += 1
                state.last_trade_time = datetime.now()
                
                return self.emit_signal(
                    symbol=symbol,
                    action=OrderAction.SELL,
                    quantity=self._quantity,
                    price=tick.bid,
                    order_type=OrderType.LIMIT,
                    stop_loss=tick.bid + self._stop_loss,
                    take_profit=tick.bid - self._profit_target,
                    strength=min(abs(state.order_imbalance) + abs(state.price_momentum) * 10, 1.0),
                    reason=f"Scalp Short: spread={tick.spread:.4f}, imb={state.order_imbalance:.2f}",
                    metadata={
                        "spread": tick.spread,
                        "imbalance": state.order_imbalance,
                        "momentum": state.price_momentum,
                        "volume_momentum": state.volume_momentum,
                    },
                )
        
        return None
    
    def _check_exit_conditions(
        self,
        state: SymbolState,
        tick: TickData,
    ) -> Optional[SignalEvent]:
        """
        檢查出場條件
        
        出場條件：
        1. 達到止盈目標
        2. 觸發止損
        3. 持倉時間超過限制
        4. 動量反轉
        """
        symbol = state.symbol
        position = self.get_position(symbol)
        
        if position is None or position.is_flat:
            return None
        
        current_price = tick.last
        entry_price = state.entry_price or position.avg_cost
        is_long = position.is_long
        
        # 計算盈虧
        if is_long:
            pnl_per_share = current_price - entry_price
            exit_price = tick.bid  # 以 bid 價賣出
        else:
            pnl_per_share = entry_price - current_price
            exit_price = tick.ask  # 以 ask 價買入平倉
        
        # 條件 1: 止盈
        if pnl_per_share >= self._profit_target:
            self._logger.info(
                f"{symbol} 止盈平倉: pnl={pnl_per_share:.4f}"
            )
            return self._create_exit_signal(
                symbol, position, exit_price, f"止盈 pnl={pnl_per_share:.4f}"
            )
        
        # 條件 2: 止損
        if pnl_per_share <= -self._stop_loss:
            self._logger.info(
                f"{symbol} 止損平倉: pnl={pnl_per_share:.4f}"
            )
            return self._create_exit_signal(
                symbol, position, exit_price, f"止損 pnl={pnl_per_share:.4f}"
            )
        
        # 條件 3: 持倉時間超過限制
        if state.entry_time:
            hold_time = (datetime.now() - state.entry_time).total_seconds()
            if hold_time >= self._max_hold_seconds:
                self._logger.info(
                    f"{symbol} 超時平倉: hold_time={hold_time:.1f}s"
                )
                return self._create_exit_signal(
                    symbol, position, exit_price, f"超時 {hold_time:.1f}s"
                )
        
        # 條件 4: 動量反轉
        if is_long and state.price_momentum < -self._min_momentum * 2:
            self._logger.info(
                f"{symbol} 動量反轉平倉: momentum={state.price_momentum:.4f}"
            )
            return self._create_exit_signal(
                symbol, position, exit_price, f"動量反轉 {state.price_momentum:.4f}"
            )
        elif not is_long and state.price_momentum > self._min_momentum * 2:
            self._logger.info(
                f"{symbol} 動量反轉平倉: momentum={state.price_momentum:.4f}"
            )
            return self._create_exit_signal(
                symbol, position, exit_price, f"動量反轉 {state.price_momentum:.4f}"
            )
        
        return None
    
    def _check_position_timeout(self, symbol: str) -> Optional[SignalEvent]:
        """檢查持倉超時（由 on_bar 調用）"""
        state = self._symbol_states.get(symbol)
        position = self.get_position(symbol)
        
        if state is None or position is None or position.is_flat:
            return None
        
        if state.entry_time:
            hold_time = (datetime.now() - state.entry_time).total_seconds()
            if hold_time >= self._max_hold_seconds:
                exit_price = position.market_price
                return self._create_exit_signal(
                    symbol, position, exit_price, f"Bar 檢查超時 {hold_time:.1f}s"
                )
        
        return None
    
    def _create_exit_signal(
        self,
        symbol: str,
        position: Position,
        price: float,
        reason: str,
    ) -> SignalEvent:
        """建立平倉信號"""
        state = self._symbol_states.get(symbol)
        
        # 重置狀態
        if state:
            state.entry_price = None
            state.entry_time = None
            state.position_side = None
        
        # 平倉方向
        action = OrderAction.SELL if position.is_long else OrderAction.BUY
        
        return self.emit_signal(
            symbol=symbol,
            action=action,
            quantity=abs(position.quantity),
            price=price,
            order_type=OrderType.MARKET,  # 平倉使用市價單
            reason=f"Exit: {reason}",
        )
    
    # ========== 工具方法 ==========
    
    def get_symbol_state(self, symbol: str) -> Optional[SymbolState]:
        """取得標的狀態"""
        return self._symbol_states.get(symbol)
    
    def get_strategy_state(self) -> Dict[str, any]:
        """取得策略狀態摘要"""
        states = {}
        for symbol, state in self._symbol_states.items():
            states[symbol] = {
                "tick_count": state.tick_count,
                "current_spread": state.current_spread,
                "avg_spread": state.avg_spread,
                "price_momentum": state.price_momentum,
                "volume_momentum": state.volume_momentum,
                "order_imbalance": state.order_imbalance,
                "position_side": state.position_side,
                "entry_price": state.entry_price,
                "trades_this_minute": state.trade_count_this_minute,
            }
        
        return {
            "strategy_id": self._strategy_id,
            "spread_threshold": self._spread_threshold,
            "tick_count_trigger": self._tick_count_trigger,
            "quantity": self._quantity,
            "profit_target": self._profit_target,
            "stop_loss": self._stop_loss,
            "max_hold_seconds": self._max_hold_seconds,
            "symbols": states,
            "stats": {
                "tick_count": self._stats.tick_count,
                "bar_count": self._stats.bar_count,
                "signal_count": self._stats.signal_count,
                "total_trades": self._stats.total_trades,
            },
        }