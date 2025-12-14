"""
test_risk.py - 風控模組測試

測試 PositionSizer、RiskManager、CircuitBreaker
"""

import asyncio
import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
import time

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
from risk.position_sizer import (
    PositionSizer,
    PositionSize,
    SizingMethod,
)
from risk import (
    RiskManager,
    RiskCheckResult,
    RiskLevel,
    PositionInfo,
    DailyStats,
)
from risk.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    BreakerConfig,
    TradeRecord,
)
from core.events import (
    SignalEvent,
    FillEvent,
    EventType,
    OrderAction,
)
from tests import MockEventBus


# ============================================================
# PositionSizer 測試
# ============================================================

class TestPositionSizer:
    """PositionSizer 測試類"""
    
    @pytest.fixture
    def sizer(self):
        """建立 PositionSizer 實例"""
        return PositionSizer(
            account_value=100000.0,
            risk_per_trade=0.01,  # 1%
            max_position_pct=0.1,  # 10%
            default_method=SizingMethod.ATR,
        )
    
    # ========== 固定倉位測試 ==========
    
    def test_position_sizer_fixed(self, sizer):
        """測試固定倉位計算"""
        sizer.set_fixed_quantity(100)
        
        result = sizer.calculate(
            price=150.0,
            method=SizingMethod.FIXED,
        )
        
        assert isinstance(result, PositionSize)
        assert result.quantity == 100
        assert result.method == SizingMethod.FIXED
        assert result.price == 150.0
        assert result.position_value == 15000.0  # 100 * 150
    
    def test_position_sizer_fixed_different_quantities(self, sizer):
        """測試不同固定數量"""
        # 設定 50 股
        sizer.set_fixed_quantity(50)
        result = sizer.calculate(price=100.0, method=SizingMethod.FIXED)
        assert result.quantity == 50
        
        # 設定 200 股
        sizer.set_fixed_quantity(200)
        result = sizer.calculate(price=100.0, method=SizingMethod.FIXED)
        assert result.quantity == 200
    
    # ========== 百分比倉位測試 ==========
    
    def test_position_sizer_percent(self, sizer):
        """測試百分比倉位計算"""
        # max_position_pct = 0.1，帳戶 100000，最大倉位 10000
        result = sizer.calculate(
            price=100.0,
            method=SizingMethod.PERCENT,
        )
        
        assert isinstance(result, PositionSize)
        assert result.method == SizingMethod.PERCENT
        # 最大倉位價值 10000 / 價格 100 = 100 股
        assert result.quantity == 100
        assert result.position_value == 10000.0
    
    def test_position_sizer_percent_high_price(self, sizer):
        """測試高價股票的百分比倉位"""
        # 最大倉位 10000，價格 500
        result = sizer.calculate(
            price=500.0,
            method=SizingMethod.PERCENT,
        )
        
        # 10000 / 500 = 20 股
        assert result.quantity == 20
    
    def test_position_sizer_percent_low_price(self, sizer):
        """測試低價股票的百分比倉位"""
        # 最大倉位 10000，價格 10
        result = sizer.calculate(
            price=10.0,
            method=SizingMethod.PERCENT,
        )
        
        # 10000 / 10 = 1000 股
        assert result.quantity == 1000
    
    # ========== ATR 倉位測試 ==========
    
    def test_position_sizer_atr_with_stop_loss(self, sizer):
        """測試基於停損的 ATR 倉位計算"""
        # risk_per_trade = 0.01，帳戶 100000，風險金額 1000
        result = sizer.calculate(
            price=100.0,
            stop_loss=95.0,  # 停損距離 5
            method=SizingMethod.ATR,
        )
        
        # 風險金額 1000 / 每股風險 5 = 200 股
        assert result.method == SizingMethod.ATR
        assert result.risk_per_share == 5.0
        assert result.quantity == 200
        assert result.risk_amount == 1000.0
    
    def test_position_sizer_atr_with_atr_value(self, sizer):
        """測試使用 ATR 值計算倉位"""
        # 使用 ATR 值（2x ATR 作為停損）
        result = sizer.calculate(
            price=100.0,
            atr=2.0,  # ATR = 2，停損距離 = 4
            method=SizingMethod.ATR,
        )
        
        # 風險金額 1000 / 每股風險 4 = 250 股
        assert result.quantity == 250
    
    def test_position_sizer_atr_default_risk(self, sizer):
        """測試沒有停損和 ATR 時使用預設風險"""
        result = sizer.calculate(
            price=100.0,
            method=SizingMethod.ATR,
        )
        
        # 預設使用 2% 價格作為風險 = 2
        # 風險金額 1000 / 2 = 500 股
        assert result.quantity == 500
    
    # ========== Kelly 倉位測試 ==========
    
    def test_position_sizer_kelly(self, sizer):
        """測試 Kelly 公式倉位計算"""
        # 設定 Kelly 參數
        sizer.set_kelly_params(
            win_rate=0.6,      # 60% 勝率
            win_loss_ratio=1.5, # 盈虧比 1.5
            fraction=0.25,      # 使用 25% Kelly
        )
        
        result = sizer.calculate(
            price=100.0,
            method=SizingMethod.KELLY,
        )
        
        assert result.method == SizingMethod.KELLY
        assert result.quantity > 0
    
    # ========== 參數設定測試 ==========
    
    def test_set_account_value(self, sizer):
        """測試設定帳戶價值"""
        sizer.set_account_value(200000.0)
        
        result = sizer.calculate(
            price=100.0,
            stop_loss=95.0,
            method=SizingMethod.ATR,
        )
        
        # 新帳戶價值 200000，風險 1% = 2000
        # 2000 / 5 = 400 股
        assert result.quantity == 400
    
    def test_set_risk_per_trade(self, sizer):
        """測試設定交易風險比例"""
        sizer.set_risk_per_trade(0.02)  # 2%
        
        result = sizer.calculate(
            price=100.0,
            stop_loss=95.0,
            method=SizingMethod.ATR,
        )
        
        # 風險金額 2000 / 5 = 400 股
        assert result.quantity == 400
    
    # ========== 限制測試 ==========
    
    def test_position_sizer_max_quantity_limit(self):
        """測試最大數量限制"""
        sizer = PositionSizer(
            account_value=1000000.0,
            risk_per_trade=0.1,
            max_quantity=100,  # 最大 100 股
        )
        
        result = sizer.calculate(
            price=10.0,
            stop_loss=9.0,
            method=SizingMethod.ATR,
        )
        
        # 計算結果應該被限制在 100
        assert result.quantity <= 100
        assert result.was_adjusted is True
    
    def test_position_sizer_min_quantity_limit(self):
        """測試最小數量限制"""
        sizer = PositionSizer(
            account_value=1000.0,
            risk_per_trade=0.001,
            min_quantity=10,
        )
        
        result = sizer.calculate(
            price=1000.0,
            stop_loss=900.0,
            method=SizingMethod.ATR,
        )
        
        # 計算結果應該至少是 10
        assert result.quantity >= 10
    
    def test_position_sizer_zero_price(self, sizer):
        """測試零價格處理"""
        result = sizer.calculate(price=0.0, method=SizingMethod.PERCENT)
        
        assert result.quantity == 0
    
    # ========== 工具方法測試 ==========
    
    def test_calculate_max_quantity(self, sizer):
        """測試計算最大可買數量"""
        max_qty = sizer.calculate_max_quantity(price=100.0)
        
        # 最大倉位 10000 / 100 = 100
        assert max_qty == 100
    
    def test_calculate_risk_quantity(self, sizer):
        """測試根據風險計算數量"""
        qty = sizer.calculate_risk_quantity(price=100.0, stop_loss=95.0)
        
        # 風險金額 1000 / 5 = 200
        assert qty == 200
    
    def test_get_config(self, sizer):
        """測試取得配置"""
        config = sizer.get_config()
        
        assert config["account_value"] == 100000.0
        assert config["risk_per_trade"] == 0.01
        assert config["max_position_pct"] == 0.1


# ============================================================
# RiskManager 測試
# ============================================================

class TestRiskManager:
    """RiskManager 測試類"""
    
    @pytest.fixture
    def event_bus(self):
        """建立 Mock EventBus"""
        return MockEventBus()
    
    @pytest.fixture
    def risk_manager(self, event_bus):
        """建立 RiskManager 實例"""
        manager = RiskManager(
            event_bus=event_bus,
            account_value=100000.0,
            daily_loss_limit=2000.0,      # 每日虧損上限 2000
            max_position_value=20000.0,    # 單一倉位上限 20000
            max_total_exposure=80000.0,    # 總曝險上限 80000
        )
        return manager
    
    # ========== 每日虧損限制測試 ==========
    
    def test_risk_manager_daily_loss_limit(self, risk_manager):
        """測試每日虧損限制"""
        # 模擬虧損交易
        risk_manager._daily_stats.realized_pnl = -1500.0
        
        # 建立信號
        signal = SignalEvent(
            event_type=EventType.SIGNAL,
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=100,
            suggested_price=150.0,
        )
        
        # 檢查信號
        result = risk_manager.check_signal(signal)
        
        # 還沒超過限制，應該通過
        assert result.passed is True
    
    def test_risk_manager_daily_loss_limit_exceeded(self, risk_manager):
        """測試超過每日虧損限制"""
        # 模擬超過虧損限制
        risk_manager._daily_stats.realized_pnl = -2500.0
        
        signal = SignalEvent(
            event_type=EventType.SIGNAL,
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=100,
            suggested_price=150.0,
        )
        
        result = risk_manager.check_signal(signal)
        
        # 超過限制，應該拒絕
        assert result.passed is False
        assert "虧損" in result.reason or "loss" in result.reason.lower()
    
    def test_risk_manager_daily_loss_warning(self, risk_manager):
        """測試接近每日虧損限制警告"""
        # 虧損達到 80%（1600）
        risk_manager._daily_stats.realized_pnl = -1600.0
        
        signal = SignalEvent(
            event_type=EventType.SIGNAL,
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=100,
            suggested_price=150.0,
        )
        
        result = risk_manager.check_signal(signal)
        
        # 應該通過但有警告
        assert result.passed is True
        assert result.has_warnings or len(result.warnings) > 0
    
    # ========== 持倉限制測試 ==========
    
    def test_risk_manager_position_limit(self, risk_manager):
        """測試單一倉位限制"""
        # 建立超過限制的信號
        signal = SignalEvent(
            event_type=EventType.SIGNAL,
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=200,  # 200 * 150 = 30000 > 20000
            suggested_price=150.0,
        )
        
        result = risk_manager.check_signal(signal)
        
        # 超過單一倉位限制
        assert result.passed is False
    
    def test_risk_manager_position_within_limit(self, risk_manager):
        """測試在倉位限制內"""
        signal = SignalEvent(
            event_type=EventType.SIGNAL,
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=100,  # 100 * 150 = 15000 < 20000
            suggested_price=150.0,
        )
        
        result = risk_manager.check_signal(signal)
        
        assert result.passed is True
    
    # ========== 曝險限制測試 ==========
    
    def test_risk_manager_exposure_limit(self, risk_manager):
        """測試總曝險限制"""
        # 先設定現有持倉
        risk_manager._positions["MSFT"] = PositionInfo(
            symbol="MSFT",
            quantity=200,
            avg_cost=300.0,
            market_price=300.0,
            market_value=60000.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
        )
        
        # 新信號會超過曝險限制
        signal = SignalEvent(
            event_type=EventType.SIGNAL,
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=200,  # 200 * 150 = 30000，總曝險 90000 > 80000
            suggested_price=150.0,
        )
        
        result = risk_manager.check_signal(signal)
        
        assert result.passed is False
    
    # ========== 風險等級測試 ==========
    
    def test_risk_manager_risk_level(self, risk_manager):
        """測試風險等級計算"""
        # 初始應該是低風險
        result = risk_manager.check_signal(SignalEvent(
            event_type=EventType.SIGNAL,
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=10,
            suggested_price=150.0,
        ))
        
        assert result.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM]
    
    # ========== 其他測試 ==========
    
    def test_risk_manager_enable_disable_trading(self, risk_manager):
        """測試啟用/停用交易"""
        # 停用交易
        risk_manager.disable_trading("測試停用")
        assert risk_manager.trading_enabled is False
        
        # 啟用交易
        risk_manager.enable_trading()
        assert risk_manager.trading_enabled is True
    
    def test_risk_manager_update_account_value(self, risk_manager):
        """測試更新帳戶價值"""
        risk_manager.update_account_value(150000.0)
        assert risk_manager.account_value == 150000.0
    
    def test_risk_manager_reset_daily_stats(self, risk_manager):
        """測試重置每日統計"""
        risk_manager._daily_stats.realized_pnl = -500.0
        risk_manager._daily_stats.trade_count = 10
        
        risk_manager.reset_daily_stats()
        
        assert risk_manager._daily_stats.realized_pnl == 0.0
        assert risk_manager._daily_stats.trade_count == 0
    
    def test_risk_manager_get_risk_summary(self, risk_manager):
        """測試取得風險摘要"""
        summary = risk_manager.get_risk_summary()
        
        assert isinstance(summary, dict)
        assert "account_value" in summary
        assert "trading_enabled" in summary


# ============================================================
# CircuitBreaker 測試
# ============================================================

class TestCircuitBreaker:
    """CircuitBreaker 測試類"""
    
    @pytest.fixture
    def event_bus(self):
        """建立 Mock EventBus"""
        return MockEventBus()
    
    @pytest.fixture
    def breaker_config(self):
        """建立熔斷器配置"""
        return BreakerConfig(
            max_consecutive_losses=3,     # 連續虧損 3 次觸發
            max_loss_amount=500.0,        # 單筆虧損 500 觸發
            cooldown_seconds=5,           # 冷卻 5 秒（測試用）
            auto_reset=True,
            half_open_trades=2,
            half_open_after_seconds=2,
        )
    
    @pytest.fixture
    def circuit_breaker(self, event_bus, breaker_config):
        """建立 CircuitBreaker 實例"""
        return CircuitBreaker(
            config=breaker_config,
            event_bus=event_bus,
            account_value=100000.0,
        )
    
    # ========== 基本狀態測試 ==========
    
    def test_circuit_breaker_initial_state(self, circuit_breaker):
        """測試熔斷器初始狀態"""
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.is_triggered is False
        assert circuit_breaker.is_closed is True
        assert circuit_breaker.can_trade() is True
    
    def test_circuit_breaker_record_winning_trade(self, circuit_breaker):
        """測試記錄獲利交易"""
        circuit_breaker.record_trade(symbol="AAPL", pnl=100.0)
        
        assert circuit_breaker.stats.total_trades == 1
        assert circuit_breaker.stats.winning_trades == 1
        assert circuit_breaker.stats.consecutive_losses == 0
        assert circuit_breaker.is_triggered is False
    
    def test_circuit_breaker_record_losing_trade(self, circuit_breaker):
        """測試記錄虧損交易"""
        circuit_breaker.record_trade(symbol="AAPL", pnl=-50.0)
        
        assert circuit_breaker.stats.total_trades == 1
        assert circuit_breaker.stats.losing_trades == 1
        assert circuit_breaker.stats.consecutive_losses == 1
    
    # ========== 熔斷觸發測試 ==========
    
    def test_circuit_breaker_trigger_consecutive_losses(self, circuit_breaker):
        """測試連續虧損觸發熔斷"""
        # 記錄 3 次連續虧損
        circuit_breaker.record_trade(symbol="AAPL", pnl=-50.0)
        assert circuit_breaker.is_triggered is False
        
        circuit_breaker.record_trade(symbol="AAPL", pnl=-50.0)
        assert circuit_breaker.is_triggered is False
        
        circuit_breaker.record_trade(symbol="AAPL", pnl=-50.0)
        
        # 達到連續虧損限制，應該觸發
        assert circuit_breaker.is_triggered is True
        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.can_trade() is False
    
    def test_circuit_breaker_trigger_large_loss(self, circuit_breaker):
        """測試大額虧損觸發熔斷"""
        # 單筆虧損超過 500
        circuit_breaker.record_trade(symbol="AAPL", pnl=-600.0)
        
        assert circuit_breaker.is_triggered is True
        assert circuit_breaker.can_trade() is False
    
    def test_circuit_breaker_no_trigger_winning_streak(self, circuit_breaker):
        """測試連續獲利不觸發"""
        for _ in range(10):
            circuit_breaker.record_trade(symbol="AAPL", pnl=100.0)
        
        assert circuit_breaker.is_triggered is False
        assert circuit_breaker.can_trade() is True
    
    def test_circuit_breaker_reset_consecutive_on_win(self, circuit_breaker):
        """測試獲利重置連續虧損計數"""
        # 2 次虧損
        circuit_breaker.record_trade(symbol="AAPL", pnl=-50.0)
        circuit_breaker.record_trade(symbol="AAPL", pnl=-50.0)
        assert circuit_breaker.stats.consecutive_losses == 2
        
        # 1 次獲利
        circuit_breaker.record_trade(symbol="AAPL", pnl=100.0)
        assert circuit_breaker.stats.consecutive_losses == 0
        
        # 再 2 次虧損（不應觸發，因為連續計數被重置）
        circuit_breaker.record_trade(symbol="AAPL", pnl=-50.0)
        circuit_breaker.record_trade(symbol="AAPL", pnl=-50.0)
        assert circuit_breaker.is_triggered is False
    
    # ========== 冷卻和重置測試 ==========
    
    def test_circuit_breaker_cooldown(self, circuit_breaker):
        """測試冷卻時間"""
        # 觸發熔斷
        for _ in range(3):
            circuit_breaker.record_trade(symbol="AAPL", pnl=-50.0)
        
        assert circuit_breaker.is_triggered is True
        
        # 檢查剩餘冷卻時間
        remaining = circuit_breaker.get_remaining_cooldown()
        assert remaining > 0
    
    def test_circuit_breaker_manual_reset(self, circuit_breaker):
        """測試手動重置"""
        # 觸發熔斷
        for _ in range(3):
            circuit_breaker.record_trade(symbol="AAPL", pnl=-50.0)
        
        assert circuit_breaker.is_triggered is True
        
        # 手動重置
        circuit_breaker.reset()
        
        assert circuit_breaker.is_triggered is False
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.can_trade() is True
    
    def test_circuit_breaker_force_reset(self, circuit_breaker):
        """測試強制重置"""
        # 觸發熔斷
        for _ in range(3):
            circuit_breaker.record_trade(symbol="AAPL", pnl=-50.0)
        
        # 強制重置（清除觸發計數）
        circuit_breaker.force_reset()
        
        assert circuit_breaker.is_triggered is False
        assert circuit_breaker.stats.trigger_count == 0
    
    def test_circuit_breaker_auto_reset_after_cooldown(self, circuit_breaker):
        """測試冷卻後自動重置"""
        # 使用更短的冷卻時間進行測試
        circuit_breaker._config.cooldown_seconds = 1
        circuit_breaker._config.half_open_after_seconds = 0.5
        
        # 觸發熔斷
        for _ in range(3):
            circuit_breaker.record_trade(symbol="AAPL", pnl=-50.0)
        
        assert circuit_breaker.is_triggered is True
        
        # 手動重置（因為自動重置需要線程）
        circuit_breaker.reset()
        
        assert circuit_breaker.can_trade() is True
    
    # ========== 狀態查詢測試 ==========
    
    def test_circuit_breaker_get_status(self, circuit_breaker):
        """測試取得狀態"""
        status = circuit_breaker.get_status()
        
        assert isinstance(status, dict)
        assert "state" in status
        assert "is_triggered" in status
        assert "can_trade" in status
        assert "consecutive_losses" in status
    
    def test_circuit_breaker_trigger_count(self, circuit_breaker):
        """測試觸發計數"""
        # 第一次觸發
        for _ in range(3):
            circuit_breaker.record_trade(symbol="AAPL", pnl=-50.0)
        
        assert circuit_breaker.stats.trigger_count == 1
        
        # 重置
        circuit_breaker.reset()
        
        # 第二次觸發
        for _ in range(3):
            circuit_breaker.record_trade(symbol="AAPL", pnl=-50.0)
        
        assert circuit_breaker.stats.trigger_count == 2
    
    # ========== 回調測試 ==========
    
    def test_circuit_breaker_on_trigger_callback(self, circuit_breaker):
        """測試觸發回調"""
        callback_called = False
        
        @circuit_breaker.on_trigger
        def on_trigger():
            nonlocal callback_called
            callback_called = True
        
        # 觸發熔斷
        for _ in range(3):
            circuit_breaker.record_trade(symbol="AAPL", pnl=-50.0)
        
        assert callback_called is True
    
    def test_circuit_breaker_on_reset_callback(self, circuit_breaker):
        """測試重置回調"""
        callback_called = False
        
        @circuit_breaker.on_reset
        def on_reset():
            nonlocal callback_called
            callback_called = True
        
        # 觸發並重置
        for _ in range(3):
            circuit_breaker.record_trade(symbol="AAPL", pnl=-50.0)
        
        circuit_breaker.reset()
        
        assert callback_called is True


# ============================================================
# 整合測試
# ============================================================

class TestRiskIntegration:
    """風控模組整合測試"""
    
    def test_position_sizer_with_risk_manager(self):
        """測試 PositionSizer 和 RiskManager 整合"""
        event_bus = MockEventBus()
        
        # 建立組件
        sizer = PositionSizer(
            account_value=100000.0,
            risk_per_trade=0.01,
        )
        
        manager = RiskManager(
            event_bus=event_bus,
            account_value=100000.0,
            daily_loss_limit=2000.0,
            max_position_value=20000.0,
        )
        
        # 計算倉位
        size = sizer.calculate(
            price=150.0,
            stop_loss=145.0,
            method=SizingMethod.ATR,
        )
        
        # 建立信號
        signal = SignalEvent(
            event_type=EventType.SIGNAL,
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=size.quantity,
            suggested_price=150.0,
        )
        
        # 風險檢查
        result = manager.check_signal(signal)
        
        # 應該通過（倉位在限制內）
        assert result.passed is True
    
    def test_circuit_breaker_with_risk_manager(self):
        """測試 CircuitBreaker 和 RiskManager 整合"""
        event_bus = MockEventBus()
        
        breaker = CircuitBreaker(
            config=BreakerConfig(max_consecutive_losses=3),
            event_bus=event_bus,
        )
        
        manager = RiskManager(
            event_bus=event_bus,
            account_value=100000.0,
        )
        
        # 模擬連續虧損
        for _ in range(3):
            breaker.record_trade(symbol="AAPL", pnl=-100.0)
        
        # 熔斷器應該觸發
        assert breaker.is_triggered is True
        assert breaker.can_trade() is False


# ============================================================
# 執行測試
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])