"""
Test Strategy - 測試用策略
用於驗證系統連接、下單、平倉功能
每 N 根 K 線交替執行買入/賣出
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict

from strategies.base import BaseStrategy, StrategyConfig
from core.events import BarEvent, SignalEvent, OrderAction, OrderType


logger = logging.getLogger(__name__)


@dataclass
class TestStrategyConfig(StrategyConfig):
    """測試策略配置"""
    
    # 觸發間隔
    trigger_bars: int = 3        # 每 N 根 K 線觸發一次交易
    
    # 交易設定
    quantity: float = 1          # XAUUSD 最小單位是 1 盎司
    
    # 自動平倉
    auto_close_bars: int = 2     # 建倉後 N 根 K 線自動平倉


class TestStrategy(BaseStrategy):
    """
    測試策略
    
    邏輯：
    - 每 trigger_bars 根 K 線觸發一次交易
    - 交替執行買入和賣出
    - 建倉後 auto_close_bars 根 K 線自動平倉
    
    用途：
    - 測試數據訂閱
    - 測試下單功能
    - 測試平倉功能
    """
    
    def __init__(
        self,
        strategy_id: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        config: Optional[TestStrategyConfig] = None,
        trigger_bars: int = 3,
        quantity: float = 1,
        auto_close_bars: int = 2,
        **kwargs,
    ):
        # 建立配置
        if config is None:
            config = TestStrategyConfig(
                strategy_id=strategy_id or "test_strategy",
                symbols=symbols or [],
                trigger_bars=trigger_bars,
                quantity=quantity,
                auto_close_bars=auto_close_bars,
            )
        
        super().__init__(config=config, **kwargs)
        
        # 策略參數
        self._trigger_bars = config.trigger_bars
        self._quantity = config.quantity
        self._auto_close_bars = config.auto_close_bars
        
        # 狀態追蹤
        self._bar_count: Dict[str, int] = {}           # 每個標的的 K 線計數
        self._position: Dict[str, int] = {}            # 持倉方向 (1=多, -1=空, 0=無)
        self._bars_since_entry: Dict[str, int] = {}    # 建倉後的 K 線數
        self._next_action: Dict[str, str] = {}         # 下一次動作 (BUY/SELL)
        
        logger.info(
            f"TestStrategy 初始化: "
            f"trigger_bars={self._trigger_bars}, "
            f"quantity={self._quantity}, "
            f"auto_close_bars={self._auto_close_bars}"
        )
    
    def on_start(self) -> None:
        """策略啟動"""
        super().on_start()
        
        # 初始化每個標的的狀態
        for symbol in self.symbols:
            self._bar_count[symbol] = 0
            self._position[symbol] = 0
            self._bars_since_entry[symbol] = 0
            self._next_action[symbol] = "BUY"  # 首次動作是買入
        
        logger.info(f"TestStrategy 啟動，監控標的: {self.symbols}")
    
    def on_bar(self, event: BarEvent) -> None:
        """處理 K 線數據"""
        symbol = event.symbol
        
        if symbol not in self.symbols:
            return
        
        # 更新計數
        self._bar_count[symbol] = self._bar_count.get(symbol, 0) + 1
        bar_num = self._bar_count[symbol]
        
        logger.info(
            f"[{symbol}] Bar #{bar_num} | "
            f"Close: {event.close:.2f} | "
            f"Position: {self._position.get(symbol, 0)} | "
            f"Time: {event.timestamp}"
        )
        
        # 如果有持倉，檢查是否需要平倉
        if self._position.get(symbol, 0) != 0:
            self._bars_since_entry[symbol] = self._bars_since_entry.get(symbol, 0) + 1
            
            if self._bars_since_entry[symbol] >= self._auto_close_bars:
                self._close_position(symbol, event)
                return
        
        # 檢查是否觸發新交易
        if bar_num % self._trigger_bars == 0 and self._position.get(symbol, 0) == 0:
            self._open_position(symbol, event)
    
    def _open_position(self, symbol: str, event: BarEvent) -> None:
        """開倉"""
        action = self._next_action.get(symbol, "BUY")
        
        if action == "BUY":
            order_action = OrderAction.BUY
            self._position[symbol] = 1
            self._next_action[symbol] = "SELL"
        else:
            order_action = OrderAction.SELL
            self._position[symbol] = -1
            self._next_action[symbol] = "BUY"
        
        self._bars_since_entry[symbol] = 0
        
        logger.info(f"🔵 [{symbol}] 開倉 {action} | Price: {event.close:.2f}")
        
        # 發送訊號
        signal = SignalEvent(
            strategy_id=self.strategy_id,
            symbol=symbol,
            action=order_action,
            quantity=self._quantity,
            order_type=OrderType.MARKET,
            timestamp=datetime.now(),
            price=event.close,
            reason=f"Test trigger at bar #{self._bar_count[symbol]}",
        )
        
        self.emit_signal(signal)
    
    def _close_position(self, symbol: str, event: BarEvent) -> None:
        """平倉"""
        position = self._position.get(symbol, 0)
        
        if position == 0:
            return
        
        # 反向操作平倉
        if position == 1:
            order_action = OrderAction.SELL
            action_name = "SELL (平多)"
        else:
            order_action = OrderAction.BUY
            action_name = "BUY (平空)"
        
        logger.info(f"🔴 [{symbol}] 平倉 {action_name} | Price: {event.close:.2f}")
        
        # 發送訊號
        signal = SignalEvent(
            strategy_id=self.strategy_id,
            symbol=symbol,
            action=order_action,
            quantity=self._quantity,
            order_type=OrderType.MARKET,
            timestamp=datetime.now(),
            price=event.close,
            reason=f"Auto close after {self._auto_close_bars} bars",
        )
        
        self.emit_signal(signal)
        
        # 重置狀態
        self._position[symbol] = 0
        self._bars_since_entry[symbol] = 0
    
    def on_stop(self) -> None:
        """策略停止"""
        # 平掉所有持倉
        for symbol in self.symbols:
            if self._position.get(symbol, 0) != 0:
                logger.info(f"策略停止，平倉 {symbol}")
        
        super().on_stop()
