"""
SMA Cross Strategy - 簡單移動平均線交叉策略

經典的趨勢跟蹤策略範例
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Deque

from strategies.base import BaseStrategy, StrategyConfig
from core.events import BarEvent, SignalEvent, OrderAction, OrderType


logger = logging.getLogger(__name__)


@dataclass
class SMACrossConfig(StrategyConfig):
    """SMA 交叉策略配置"""
    
    # 均線週期
    fast_period: int = 10      # 快線週期
    slow_period: int = 20      # 慢線週期
    
    # 交易設定
    quantity: int = 100        # 每次交易數量
    
    # 過濾條件
    min_cross_strength: float = 0.001  # 最小交叉幅度（百分比）
    
    # 止損止盈（可選）
    use_stop_loss: bool = False
    stop_loss_pct: float = 0.02       # 2% 止損
    use_take_profit: bool = False
    take_profit_pct: float = 0.04     # 4% 止盈


@dataclass
class SymbolData:
    """單一標的的數據"""
    
    symbol: str
    
    # 價格歷史
    prices: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    
    # 均線值
    fast_sma: Optional[float] = None
    slow_sma: Optional[float] = None
    
    # 前一個均線值（用於判斷交叉）
    prev_fast_sma: Optional[float] = None
    prev_slow_sma: Optional[float] = None
    
    # 狀態
    last_signal: Optional[str] = None  # "BUY" or "SELL"
    last_signal_time: Optional[datetime] = None
    
    def update_prices(self, price: float) -> None:
        """更新價格歷史"""
        self.prices.append(price)
    
    def calculate_sma(self, period: int) -> Optional[float]:
        """計算 SMA"""
        if len(self.prices) < period:
            return None
        
        recent_prices = list(self.prices)[-period:]
        return sum(recent_prices) / period
    
    def update_sma(self, fast_period: int, slow_period: int) -> None:
        """更新均線值"""
        # 保存前一個值
        self.prev_fast_sma = self.fast_sma
        self.prev_slow_sma = self.slow_sma
        
        # 計算新的均線
        self.fast_sma = self.calculate_sma(fast_period)
        self.slow_sma = self.calculate_sma(slow_period)
    
    @property
    def has_enough_data(self) -> bool:
        """是否有足夠數據計算均線"""
        return self.fast_sma is not None and self.slow_sma is not None
    
    @property
    def has_prev_data(self) -> bool:
        """是否有前一個均線數據"""
        return self.prev_fast_sma is not None and self.prev_slow_sma is not None


class SMACrossStrategy(BaseStrategy):
    """
    簡單移動平均線交叉策略
    
    策略邏輯：
    - 當快線從下方穿越慢線（黃金交叉）時，發出買入信號
    - 當快線從上方穿越慢線（死亡交叉）時，發出賣出信號
    
    使用方式:
        config = SMACrossConfig(
            fast_period=5,
            slow_period=20,
            symbols=["AAPL", "MSFT"],
        )
        strategy = SMACrossStrategy(config=config)
    """
    
    def __init__(
        self,
        strategy_id: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        config: Optional[SMACrossConfig] = None,
        fast_period: int = 10,
        slow_period: int = 20,
        quantity: int = 100,
        **kwargs,
    ):
        """
        初始化策略
        
        Args:
            strategy_id: 策略 ID
            symbols: 標的列表
            config: 策略配置
            fast_period: 快線週期（當 config 為 None 時使用）
            slow_period: 慢線週期（當 config 為 None 時使用）
            quantity: 交易數量（當 config 為 None 時使用）
        """
        # 建立配置
        if config is None:
            config = SMACrossConfig(
                fast_period=fast_period,
                slow_period=slow_period,
                quantity=quantity,
                symbols=symbols or [],
            )
        
        super().__init__(
            strategy_id=strategy_id or "sma_cross",
            symbols=symbols,
            config=config,
            **kwargs,
        )
        
        # 策略參數
        self._fast_period = config.fast_period
        self._slow_period = config.slow_period
        self._quantity = config.quantity
        self._min_cross_strength = config.min_cross_strength
        
        # 確保週期正確
        if self._fast_period >= self._slow_period:
            raise ValueError("fast_period 必須小於 slow_period")
        
        # 每個標的的數據
        self._symbol_data: Dict[str, SymbolData] = {}
        
        self._logger.info(
            f"SMACrossStrategy 初始化: fast={self._fast_period}, "
            f"slow={self._slow_period}, qty={self._quantity}"
        )
    
    # ========== 屬性 ==========
    
    @property
    def fast_period(self) -> int:
        """快線週期"""
        return self._fast_period
    
    @property
    def slow_period(self) -> int:
        """慢線週期"""
        return self._slow_period
    
    # ========== 生命週期 ==========
    
    def initialize(self) -> None:
        """初始化策略"""
        super().initialize()
        
        # 為每個標的建立數據結構
        for symbol in self._symbols:
            self._symbol_data[symbol] = SymbolData(
                symbol=symbol,
                prices=deque(maxlen=self._slow_period + 10),
            )
    
    def on_start(self) -> None:
        """策略啟動"""
        super().on_start()
        self._logger.info(
            f"SMACrossStrategy 啟動，監控標的: {list(self._symbols)}"
        )
    
    def on_stop(self) -> None:
        """策略停止"""
        super().on_stop()
        
        # 輸出最終統計
        for symbol, data in self._symbol_data.items():
            self._logger.info(
                f"{symbol}: prices={len(data.prices)}, "
                f"fast_sma={data.fast_sma:.2f if data.fast_sma else 'N/A'}, "
                f"slow_sma={data.slow_sma:.2f if data.slow_sma else 'N/A'}"
            )
    
    # ========== 事件處理 ==========
    
    def on_bar(self, event: BarEvent) -> Optional[SignalEvent]:
        """
        處理 Bar 事件
        
        Args:
            event: BarEvent
            
        Returns:
            SignalEvent 或 None
        """
        symbol = event.symbol
        
        # 更新統計
        self._stats.bar_count += 1
        
        # 取得或建立標的數據
        if symbol not in self._symbol_data:
            self._symbol_data[symbol] = SymbolData(
                symbol=symbol,
                prices=deque(maxlen=self._slow_period + 10),
            )
        
        data = self._symbol_data[symbol]
        
        # 使用收盤價
        price = event.close
        
        # 更新價格歷史
        data.update_prices(price)
        
        # 更新均線
        data.update_sma(self._fast_period, self._slow_period)
        
        # 檢查是否有足夠數據
        if not data.has_enough_data:
            self._logger.debug(
                f"{symbol}: 數據不足，需要 {self._slow_period} 個 bar，"
                f"目前 {len(data.prices)} 個"
            )
            return None
        
        # 檢查是否有前一個數據（用於判斷交叉）
        if not data.has_prev_data:
            return None
        
        # 檢測交叉
        signal = self._check_crossover(data, price)
        
        return signal
    
    def _check_crossover(
        self,
        data: SymbolData,
        current_price: float,
    ) -> Optional[SignalEvent]:
        """
        檢測均線交叉
        
        Args:
            data: 標的數據
            current_price: 當前價格
            
        Returns:
            SignalEvent 或 None
        """
        symbol = data.symbol
        
        # 計算交叉強度
        cross_strength = abs(data.fast_sma - data.slow_sma) / data.slow_sma
        
        # 前一個狀態：快線在慢線下方
        was_below = data.prev_fast_sma < data.prev_slow_sma
        # 當前狀態：快線在慢線上方
        is_above = data.fast_sma > data.slow_sma
        
        # 黃金交叉（快線向上穿越慢線）
        if was_below and is_above:
            if cross_strength >= self._min_cross_strength:
                self._logger.info(
                    f"{symbol} 黃金交叉: fast={data.fast_sma:.2f}, "
                    f"slow={data.slow_sma:.2f}, strength={cross_strength:.4f}"
                )
                
                # 計算止損止盈
                stop_loss = None
                take_profit = None
                
                if isinstance(self._config, SMACrossConfig):
                    if self._config.use_stop_loss:
                        stop_loss = current_price * (1 - self._config.stop_loss_pct)
                    if self._config.use_take_profit:
                        take_profit = current_price * (1 + self._config.take_profit_pct)
                
                # 更新狀態
                data.last_signal = "BUY"
                data.last_signal_time = datetime.now()
                
                return self.emit_signal(
                    symbol=symbol,
                    action=OrderAction.BUY,
                    quantity=self._quantity,
                    price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    strength=min(cross_strength * 100, 1.0),
                    reason=f"黃金交叉 (fast={data.fast_sma:.2f}, slow={data.slow_sma:.2f})",
                    metadata={
                        "fast_sma": data.fast_sma,
                        "slow_sma": data.slow_sma,
                        "cross_strength": cross_strength,
                    },
                )
        
        # 死亡交叉（快線向下穿越慢線）
        elif not was_below and not is_above:
            if cross_strength >= self._min_cross_strength:
                self._logger.info(
                    f"{symbol} 死亡交叉: fast={data.fast_sma:.2f}, "
                    f"slow={data.slow_sma:.2f}, strength={cross_strength:.4f}"
                )
                
                # 計算止損止盈
                stop_loss = None
                take_profit = None
                
                if isinstance(self._config, SMACrossConfig):
                    if self._config.use_stop_loss:
                        stop_loss = current_price * (1 + self._config.stop_loss_pct)
                    if self._config.use_take_profit:
                        take_profit = current_price * (1 - self._config.take_profit_pct)
                
                # 更新狀態
                data.last_signal = "SELL"
                data.last_signal_time = datetime.now()
                
                return self.emit_signal(
                    symbol=symbol,
                    action=OrderAction.SELL,
                    quantity=self._quantity,
                    price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    strength=min(cross_strength * 100, 1.0),
                    reason=f"死亡交叉 (fast={data.fast_sma:.2f}, slow={data.slow_sma:.2f})",
                    metadata={
                        "fast_sma": data.fast_sma,
                        "slow_sma": data.slow_sma,
                        "cross_strength": cross_strength,
                    },
                )
        
        return None
    
    # ========== 工具方法 ==========
    
    def get_sma_values(self, symbol: str) -> Dict[str, Optional[float]]:
        """
        取得指定標的的均線值
        
        Args:
            symbol: 標的代碼
            
        Returns:
            {"fast_sma": float, "slow_sma": float}
        """
        data = self._symbol_data.get(symbol)
        if data is None:
            return {"fast_sma": None, "slow_sma": None}
        
        return {
            "fast_sma": data.fast_sma,
            "slow_sma": data.slow_sma,
        }
    
    def get_symbol_data(self, symbol: str) -> Optional[SymbolData]:
        """取得標的數據"""
        return self._symbol_data.get(symbol)
    
    def get_all_sma_values(self) -> Dict[str, Dict[str, Optional[float]]]:
        """取得所有標的的均線值"""
        return {
            symbol: self.get_sma_values(symbol)
            for symbol in self._symbol_data
        }
    
    def get_strategy_state(self) -> Dict[str, any]:
        """取得策略狀態摘要"""
        return {
            "strategy_id": self._strategy_id,
            "fast_period": self._fast_period,
            "slow_period": self._slow_period,
            "quantity": self._quantity,
            "symbols": list(self._symbols),
            "sma_values": self.get_all_sma_values(),
            "stats": {
                "bar_count": self._stats.bar_count,
                "signal_count": self._stats.signal_count,
                "total_trades": self._stats.total_trades,
                "win_rate": f"{self._stats.win_rate:.2%}",
            },
        }