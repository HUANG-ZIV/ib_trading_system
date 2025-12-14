"""
test_strategies.py - 策略模組測試

測試策略基類和範例策略
"""

import asyncio
import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
from collections import deque

# 先導入 tests 模組（設定 mock）
import tests

# ============================================================
# Python 3.14+ 相容性修復
# ============================================================
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# 測試目標
from strategies.base import (
    BaseStrategy,
    StrategyState,
    StrategyConfig,
    StrategyStats,
    Position,
)
from strategies.examples.sma_cross import (
    SMACrossStrategy,
    SMACrossConfig,
    SymbolData,
)
from core.events import (
    EventType,
    BarEvent,
    SignalEvent,
    FillEvent,
    OrderAction,
)
from core.event_bus import EventBus


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def event_bus():
    """建立 EventBus 實例"""
    return EventBus()


@pytest.fixture
def mock_event_bus():
    """建立 Mock EventBus"""
    mock = MagicMock()
    mock.published_events = []
    
    def publish_side_effect(event):
        mock.published_events.append(event)
    
    mock.publish.side_effect = publish_side_effect
    return mock


@pytest.fixture
def sample_config():
    """建立範例策略配置"""
    return StrategyConfig(
        name="TestStrategy",
        symbols=["AAPL", "GOOGL"],
        max_position_size=100,
        default_quantity=10,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
    )


def create_bar_event(
    symbol: str = "AAPL",
    open_price: float = 150.0,
    high: float = 152.0,
    low: float = 149.0,
    close: float = 151.0,
    volume: int = 10000,
    timestamp: datetime = None,
) -> BarEvent:
    """建立 BarEvent"""
    return BarEvent(
        event_type=EventType.BAR,
        symbol=symbol,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
        timestamp=timestamp or datetime.now(),
    )


def create_fill_event(
    symbol: str = "AAPL",
    action: OrderAction = OrderAction.BUY,
    quantity: int = 100,
    price: float = 150.0,
) -> FillEvent:
    """建立 FillEvent"""
    return FillEvent(
        event_type=EventType.FILL,
        symbol=symbol,
        action=action,
        quantity=quantity,
        price=price,
        commission=1.0,
        order_id=1,
    )


# ============================================================
# 測試用具體策略（因為 BaseStrategy 是抽象類）
# ============================================================

class ConcreteStrategy(BaseStrategy):
    """具體策略實作（用於測試 BaseStrategy）"""
    
    def __init__(self, symbols, config=None, event_bus=None, **kwargs):
        super().__init__(symbols, config, event_bus, **kwargs)
        self.bars_received = []
        self.signals_to_emit = []
    
    def on_bar(self, event: BarEvent) -> None:
        """處理 Bar 事件"""
        self.bars_received.append(event)
        self._stats.bar_count += 1
        
        # 發送預設的信號
        for signal_params in self.signals_to_emit:
            self.emit_signal(**signal_params)
    
    def add_signal_to_emit(self, **kwargs):
        """添加要發送的信號"""
        self.signals_to_emit.append(kwargs)


# ============================================================
# BaseStrategy 測試
# ============================================================

class TestBaseStrategy:
    """BaseStrategy 測試類"""
    
    def test_strategy_init(self, mock_event_bus, sample_config):
        """測試策略初始化"""
        strategy = ConcreteStrategy(
            symbols=["AAPL", "GOOGL"],
            config=sample_config,
            event_bus=mock_event_bus,
        )
        
        assert strategy.symbols == ["AAPL", "GOOGL"]
        assert strategy.state == StrategyState.IDLE
        assert strategy.config == sample_config
        assert strategy.strategy_id is not None
    
    def test_strategy_lifecycle(self, mock_event_bus):
        """測試策略生命週期"""
        strategy = ConcreteStrategy(
            symbols=["AAPL"],
            event_bus=mock_event_bus,
        )
        
        # 初始狀態
        assert strategy.state == StrategyState.IDLE
        
        # 初始化
        strategy.initialize()
        assert strategy.state == StrategyState.IDLE  # 初始化不改變狀態
        
        # 啟動
        strategy.start()
        assert strategy.state == StrategyState.RUNNING
        
        # 暫停
        strategy.pause()
        assert strategy.state == StrategyState.PAUSED
        
        # 恢復
        strategy.resume()
        assert strategy.state == StrategyState.RUNNING
        
        # 停止
        strategy.stop()
        assert strategy.state == StrategyState.STOPPED
    
    def test_strategy_start_stop(self, mock_event_bus):
        """測試策略啟動和停止"""
        strategy = ConcreteStrategy(
            symbols=["AAPL"],
            event_bus=mock_event_bus,
        )
        
        # 啟動
        strategy.start()
        assert strategy.state == StrategyState.RUNNING
        assert strategy.stats.started_at is not None
        
        # 停止
        strategy.stop()
        assert strategy.state == StrategyState.STOPPED
        assert strategy.stats.stopped_at is not None
    
    def test_on_bar_processing(self, mock_event_bus):
        """測試 Bar 事件處理"""
        strategy = ConcreteStrategy(
            symbols=["AAPL"],
            event_bus=mock_event_bus,
        )
        strategy.start()
        
        # 發送 Bar 事件
        bar = create_bar_event("AAPL", close=150.0)
        strategy.on_bar(bar)
        
        assert len(strategy.bars_received) == 1
        assert strategy.bars_received[0].close == 150.0
        assert strategy.stats.bar_count == 1
    
    def test_signal_emission(self, mock_event_bus):
        """測試信號發送"""
        strategy = ConcreteStrategy(
            symbols=["AAPL"],
            event_bus=mock_event_bus,
        )
        strategy.start()
        
        # 設定要發送的信號
        strategy.add_signal_to_emit(
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=100,
            price=150.0,
            reason="Test signal",
        )
        
        # 觸發 on_bar 來發送信號
        bar = create_bar_event("AAPL")
        strategy.on_bar(bar)
        
        # 驗證信號已發布
        assert len(mock_event_bus.published_events) == 1
        signal = mock_event_bus.published_events[0]
        assert isinstance(signal, SignalEvent)
        assert signal.symbol == "AAPL"
        assert signal.action == OrderAction.BUY
        assert signal.quantity == 100
    
    def test_buy_convenience_method(self, mock_event_bus):
        """測試 buy 便捷方法"""
        strategy = ConcreteStrategy(
            symbols=["AAPL"],
            event_bus=mock_event_bus,
        )
        strategy.start()
        
        # 使用 buy 方法
        strategy.buy("AAPL", quantity=50, price=150.0, reason="Buy test")
        
        # 驗證信號
        assert len(mock_event_bus.published_events) == 1
        signal = mock_event_bus.published_events[0]
        assert signal.action == OrderAction.BUY
        assert signal.quantity == 50
    
    def test_sell_convenience_method(self, mock_event_bus):
        """測試 sell 便捷方法"""
        strategy = ConcreteStrategy(
            symbols=["AAPL"],
            event_bus=mock_event_bus,
        )
        strategy.start()
        
        # 使用 sell 方法
        strategy.sell("AAPL", quantity=50, price=155.0, reason="Sell test")
        
        # 驗證信號
        assert len(mock_event_bus.published_events) == 1
        signal = mock_event_bus.published_events[0]
        assert signal.action == OrderAction.SELL
        assert signal.quantity == 50
    
    def test_position_tracking(self, mock_event_bus):
        """測試持倉追蹤"""
        strategy = ConcreteStrategy(
            symbols=["AAPL"],
            event_bus=mock_event_bus,
        )
        strategy.start()
        
        # 初始無持倉
        assert strategy.is_flat("AAPL")
        assert strategy.get_position_quantity("AAPL") == 0
        
        # 模擬成交
        fill = create_fill_event("AAPL", OrderAction.BUY, 100, 150.0)
        strategy.on_fill(fill)
        
        # 驗證持倉
        assert strategy.has_position("AAPL")
        assert strategy.is_long("AAPL")
        assert strategy.get_position_quantity("AAPL") == 100
    
    def test_position_update_on_sell(self, mock_event_bus):
        """測試賣出後持倉更新"""
        strategy = ConcreteStrategy(
            symbols=["AAPL"],
            event_bus=mock_event_bus,
        )
        strategy.start()
        
        # 買入
        fill_buy = create_fill_event("AAPL", OrderAction.BUY, 100, 150.0)
        strategy.on_fill(fill_buy)
        assert strategy.get_position_quantity("AAPL") == 100
        
        # 部分賣出
        fill_sell = create_fill_event("AAPL", OrderAction.SELL, 30, 155.0)
        strategy.on_fill(fill_sell)
        assert strategy.get_position_quantity("AAPL") == 70
    
    def test_get_param(self, mock_event_bus, sample_config):
        """測試參數獲取"""
        sample_config.params = {"fast_period": 10, "slow_period": 20}
        
        strategy = ConcreteStrategy(
            symbols=["AAPL"],
            config=sample_config,
            event_bus=mock_event_bus,
        )
        
        assert strategy.get_param("fast_period") == 10
        assert strategy.get_param("slow_period") == 20
        assert strategy.get_param("nonexistent", default=999) == 999
    
    def test_cache_data(self, mock_event_bus):
        """測試數據快取"""
        strategy = ConcreteStrategy(
            symbols=["AAPL"],
            event_bus=mock_event_bus,
        )
        
        # 快取數據
        strategy.cache_data("my_key", [1, 2, 3])
        
        # 讀取快取
        assert strategy.get_cached_data("my_key") == [1, 2, 3]
        assert strategy.get_cached_data("nonexistent", default=[]) == []
    
    def test_stats_summary(self, mock_event_bus):
        """測試統計摘要"""
        strategy = ConcreteStrategy(
            symbols=["AAPL"],
            event_bus=mock_event_bus,
        )
        strategy.start()
        
        # 處理一些 bar
        for i in range(5):
            bar = create_bar_event("AAPL", close=150.0 + i)
            strategy.on_bar(bar)
        
        summary = strategy.get_stats_summary()
        
        assert summary["bar_count"] == 5
        assert summary["state"] == "RUNNING"


# ============================================================
# Position 測試
# ============================================================

class TestPosition:
    """Position 測試類"""
    
    def test_position_init(self):
        """測試 Position 初始化"""
        pos = Position(symbol="AAPL", quantity=100, avg_cost=150.0)
        
        assert pos.symbol == "AAPL"
        assert pos.quantity == 100
        assert pos.avg_cost == 150.0
        assert pos.market_price == 0.0
    
    def test_position_is_long(self):
        """測試多頭判斷"""
        pos = Position(symbol="AAPL", quantity=100, avg_cost=150.0)
        
        assert pos.is_long is True
        assert pos.is_short is False
        assert pos.is_flat is False
    
    def test_position_is_short(self):
        """測試空頭判斷"""
        pos = Position(symbol="AAPL", quantity=-100, avg_cost=150.0)
        
        assert pos.is_long is False
        assert pos.is_short is True
        assert pos.is_flat is False
    
    def test_position_is_flat(self):
        """測試空倉判斷"""
        pos = Position(symbol="AAPL", quantity=0, avg_cost=0.0)
        
        assert pos.is_long is False
        assert pos.is_short is False
        assert pos.is_flat is True
    
    def test_position_unrealized_pnl(self):
        """測試未實現盈虧計算"""
        pos = Position(symbol="AAPL", quantity=100, avg_cost=150.0, market_price=160.0)
        
        # (160 - 150) * 100 = 1000
        assert pos.unrealized_pnl == 1000.0
        
        # 百分比
        assert pos.unrealized_pnl_pct == pytest.approx(0.0667, rel=0.01)
    
    def test_position_update_price(self):
        """測試價格更新"""
        pos = Position(symbol="AAPL", quantity=100, avg_cost=150.0)
        
        pos.update_price(155.0)
        
        assert pos.market_price == 155.0
        assert pos.unrealized_pnl == 500.0
    
    def test_position_update_from_fill(self):
        """測試從成交更新持倉"""
        pos = Position(symbol="AAPL", quantity=100, avg_cost=150.0)
        
        # 加倉
        pos.update_from_fill(50, 160.0)
        
        assert pos.quantity == 150
        # 平均成本: (100*150 + 50*160) / 150 = 153.33
        assert pos.avg_cost == pytest.approx(153.33, rel=0.01)


# ============================================================
# SMACrossStrategy 測試
# ============================================================

class TestSMACrossStrategy:
    """SMACrossStrategy 測試類"""
    
    @pytest.fixture
    def sma_strategy(self, mock_event_bus):
        """建立 SMA 策略實例"""
        strategy = SMACrossStrategy(
            symbols=["AAPL"],
            event_bus=mock_event_bus,
            fast_period=5,
            slow_period=10,
            quantity=100,
        )
        strategy.initialize()
        strategy.start()
        return strategy
    
    def test_sma_strategy_init(self, sma_strategy):
        """測試 SMA 策略初始化"""
        assert sma_strategy.state == StrategyState.RUNNING
        assert "AAPL" in sma_strategy._symbol_data
    
    def test_sma_strategy_needs_enough_data(self, sma_strategy, mock_event_bus):
        """測試 SMA 策略需要足夠數據"""
        # 發送少於 slow_period 的 bar
        for i in range(8):
            bar = create_bar_event("AAPL", close=150.0 + i)
            sma_strategy.on_bar(bar)
        
        # 不應該有信號（數據不足）
        assert len(mock_event_bus.published_events) == 0
    
    def test_sma_cross_golden_cross(self, mock_event_bus):
        """測試黃金交叉信號"""
        strategy = SMACrossStrategy(
            symbols=["AAPL"],
            event_bus=mock_event_bus,
            fast_period=3,
            slow_period=5,
            quantity=100,
        )
        strategy.initialize()
        strategy.start()
        
        # 製造下降趨勢（快線在慢線下方）
        prices = [160, 158, 156, 154, 152, 150]
        for price in prices:
            bar = create_bar_event("AAPL", close=price)
            strategy.on_bar(bar)
        
        mock_event_bus.published_events.clear()
        
        # 製造上升趨勢（快線穿越慢線上方 = 黃金交叉）
        rising_prices = [152, 155, 158, 162, 168]
        for price in rising_prices:
            bar = create_bar_event("AAPL", close=price)
            strategy.on_bar(bar)
        
        # 檢查是否有 BUY 信號
        buy_signals = [
            e for e in mock_event_bus.published_events
            if isinstance(e, SignalEvent) and e.action == OrderAction.BUY
        ]
        
        # 應該有黃金交叉信號
        assert len(buy_signals) >= 1
    
    def test_sma_cross_death_cross(self, mock_event_bus):
        """測試死亡交叉信號"""
        strategy = SMACrossStrategy(
            symbols=["AAPL"],
            event_bus=mock_event_bus,
            fast_period=3,
            slow_period=5,
            quantity=100,
        )
        strategy.initialize()
        strategy.start()
        
        # 製造上升趨勢（快線在慢線上方）
        prices = [140, 142, 145, 148, 152, 156]
        for price in prices:
            bar = create_bar_event("AAPL", close=price)
            strategy.on_bar(bar)
        
        mock_event_bus.published_events.clear()
        
        # 製造下降趨勢（快線穿越慢線下方 = 死亡交叉）
        falling_prices = [154, 150, 145, 140, 135]
        for price in falling_prices:
            bar = create_bar_event("AAPL", close=price)
            strategy.on_bar(bar)
        
        # 檢查是否有 SELL 信號
        sell_signals = [
            e for e in mock_event_bus.published_events
            if isinstance(e, SignalEvent) and e.action == OrderAction.SELL
        ]
        
        # 應該有死亡交叉信號
        assert len(sell_signals) >= 1
    
    def test_sma_values_calculation(self, sma_strategy):
        """測試 SMA 值計算"""
        # 發送固定價格的 bar
        for i in range(15):
            bar = create_bar_event("AAPL", close=100.0)
            sma_strategy.on_bar(bar)
        
        # 取得 SMA 值
        sma_values = sma_strategy.get_sma_values("AAPL")
        
        if sma_values:
            # 固定價格時，SMA 應該等於價格
            assert sma_values["fast_sma"] == pytest.approx(100.0, rel=0.01)
            assert sma_values["slow_sma"] == pytest.approx(100.0, rel=0.01)
    
    def test_sma_strategy_multiple_symbols(self, mock_event_bus):
        """測試多標的 SMA 策略"""
        strategy = SMACrossStrategy(
            symbols=["AAPL", "GOOGL"],
            event_bus=mock_event_bus,
            fast_period=3,
            slow_period=5,
        )
        strategy.initialize()
        strategy.start()
        
        # 兩個標的都有數據結構
        assert "AAPL" in strategy._symbol_data
        assert "GOOGL" in strategy._symbol_data
        
        # 發送兩個標的的 bar
        for i in range(10):
            bar_aapl = create_bar_event("AAPL", close=150.0 + i)
            bar_googl = create_bar_event("GOOGL", close=2800.0 + i * 10)
            
            strategy.on_bar(bar_aapl)
            strategy.on_bar(bar_googl)
        
        # 兩個標的都有數據
        assert len(strategy._symbol_data["AAPL"].prices) > 0
        assert len(strategy._symbol_data["GOOGL"].prices) > 0
    
    def test_sma_strategy_no_signal_when_flat(self, mock_event_bus):
        """測試橫盤時不發信號"""
        strategy = SMACrossStrategy(
            symbols=["AAPL"],
            event_bus=mock_event_bus,
            fast_period=3,
            slow_period=5,
            min_cross_strength=0.01,  # 1% 最小交叉強度
        )
        strategy.initialize()
        strategy.start()
        
        # 發送幾乎不變的價格
        for i in range(20):
            bar = create_bar_event("AAPL", close=150.0 + (i % 2) * 0.1)
            strategy.on_bar(bar)
        
        # 微小波動不應產生信號
        signals = [
            e for e in mock_event_bus.published_events
            if isinstance(e, SignalEvent)
        ]
        
        # 橫盤時應該很少或沒有信號
        assert len(signals) <= 2


# ============================================================
# SymbolData 測試
# ============================================================

class TestSymbolData:
    """SymbolData 測試類"""
    
    def test_symbol_data_init(self):
        """測試 SymbolData 初始化"""
        data = SymbolData(symbol="AAPL", max_len=100)
        
        assert data.symbol == "AAPL"
        assert len(data.prices) == 0
    
    def test_symbol_data_update_prices(self):
        """測試價格更新"""
        data = SymbolData(symbol="AAPL", max_len=10)
        
        for i in range(5):
            data.update_prices(100.0 + i)
        
        assert len(data.prices) == 5
        assert data.prices[-1] == 104.0
    
    def test_symbol_data_max_len(self):
        """測試最大長度限制"""
        data = SymbolData(symbol="AAPL", max_len=5)
        
        for i in range(10):
            data.update_prices(100.0 + i)
        
        # 應該只保留最後 5 個
        assert len(data.prices) == 5
        assert data.prices[0] == 105.0
        assert data.prices[-1] == 109.0
    
    def test_symbol_data_calculate_sma(self):
        """測試 SMA 計算"""
        data = SymbolData(symbol="AAPL", max_len=100)
        
        # 添加價格
        prices = [100, 102, 104, 106, 108]
        for p in prices:
            data.update_prices(p)
        
        # 計算 3 期 SMA
        sma = data.calculate_sma(3)
        
        # (104 + 106 + 108) / 3 = 106
        assert sma == pytest.approx(106.0, rel=0.01)
    
    def test_symbol_data_has_enough_data(self):
        """測試數據充足判斷"""
        data = SymbolData(symbol="AAPL", max_len=100)
        
        # 數據不足
        for i in range(3):
            data.update_prices(100.0 + i)
        
        data.update_sma(fast_period=5, slow_period=10)
        assert data.has_enough_data is False
        
        # 添加更多數據
        for i in range(10):
            data.update_prices(100.0 + i)
        
        data.update_sma(fast_period=5, slow_period=10)
        assert data.has_enough_data is True


# ============================================================
# 整合測試
# ============================================================

class TestStrategyIntegration:
    """策略整合測試"""
    
    def test_full_trading_cycle(self, mock_event_bus):
        """測試完整交易週期"""
        strategy = SMACrossStrategy(
            symbols=["AAPL"],
            event_bus=mock_event_bus,
            fast_period=3,
            slow_period=5,
            quantity=100,
        )
        strategy.initialize()
        strategy.start()
        
        # 模擬完整的價格序列
        # 先下跌
        for price in [160, 155, 150, 145, 140]:
            bar = create_bar_event("AAPL", close=price)
            strategy.on_bar(bar)
        
        # 然後上漲（觸發買入）
        for price in [142, 148, 155, 165, 175]:
            bar = create_bar_event("AAPL", close=price)
            strategy.on_bar(bar)
        
        # 模擬買入成交
        fill = create_fill_event("AAPL", OrderAction.BUY, 100, 165.0)
        strategy.on_fill(fill)
        
        # 驗證持倉
        assert strategy.has_position("AAPL")
        
        # 然後下跌（觸發賣出）
        for price in [170, 162, 155, 148, 140]:
            bar = create_bar_event("AAPL", close=price)
            strategy.on_bar(bar)
        
        # 停止策略
        strategy.stop()
        assert strategy.state == StrategyState.STOPPED
    
    def test_strategy_with_real_event_bus(self):
        """測試使用真實 EventBus"""
        event_bus = EventBus()
        received_signals = []
        
        def signal_handler(signal):
            received_signals.append(signal)
        
        event_bus.subscribe(EventType.SIGNAL, signal_handler)
        
        strategy = SMACrossStrategy(
            symbols=["AAPL"],
            event_bus=event_bus,
            fast_period=3,
            slow_period=5,
            quantity=100,
        )
        strategy.initialize()
        strategy.start()
        
        # 產生黃金交叉
        for price in [100, 98, 96, 94, 92, 90]:
            bar = create_bar_event("AAPL", close=price)
            strategy.on_bar(bar)
        
        for price in [92, 96, 100, 105, 112]:
            bar = create_bar_event("AAPL", close=price)
            strategy.on_bar(bar)
        
        # 信號應該通過 EventBus 傳遞
        buy_signals = [s for s in received_signals if s.action == OrderAction.BUY]
        assert len(buy_signals) >= 1


# ============================================================
# 執行測試
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])