"""
RiskManager 模組 - 風險管理器

提供全面的風險控制功能
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum, auto
from typing import Optional, Dict, List, Any
import threading

from core.events import (
    Event,
    EventType,
    SignalEvent,
    FillEvent,
    PositionEvent,
    RiskEvent,
    RiskEventType,
    OrderAction,
)
from core.event_bus import EventBus, get_event_bus
from config.settings import RiskConfig


# 設定 logger
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """風險等級"""
    
    LOW = auto()       # 低風險
    MEDIUM = auto()    # 中風險
    HIGH = auto()      # 高風險
    CRITICAL = auto()  # 危急


@dataclass
class RiskCheckResult:
    """風控檢查結果"""
    
    passed: bool                      # 是否通過
    risk_level: RiskLevel = RiskLevel.LOW
    reason: str = ""                  # 拒絕原因
    warnings: List[str] = field(default_factory=list)  # 警告訊息
    adjusted_quantity: Optional[int] = None  # 調整後的數量
    
    @property
    def has_warnings(self) -> bool:
        """是否有警告"""
        return len(self.warnings) > 0


@dataclass
class PositionInfo:
    """持倉資訊"""
    
    symbol: str
    quantity: int = 0
    avg_cost: float = 0.0
    market_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        return self.quantity == 0
    
    def update_from_fill(self, quantity: int, price: float) -> float:
        """根據成交更新持倉，返回已實現盈虧"""
        realized_pnl = 0.0
        
        if self.quantity == 0:
            self.quantity = quantity
            self.avg_cost = price
        elif (self.quantity > 0 and quantity > 0) or (self.quantity < 0 and quantity < 0):
            # 加倉
            total_cost = self.avg_cost * abs(self.quantity) + price * abs(quantity)
            self.quantity += quantity
            self.avg_cost = total_cost / abs(self.quantity)
        else:
            # 減倉或反向
            close_qty = min(abs(self.quantity), abs(quantity))
            realized_pnl = close_qty * (price - self.avg_cost) * (1 if self.quantity > 0 else -1)
            self.quantity += quantity
            
            if self.quantity != 0 and (self.quantity > 0) != (self.quantity - quantity > 0):
                self.avg_cost = price
        
        self.market_price = price
        self.market_value = abs(self.quantity) * price
        self.unrealized_pnl = self.quantity * (price - self.avg_cost) if self.quantity != 0 else 0
        self.realized_pnl += realized_pnl
        
        return realized_pnl


@dataclass
class DailyStats:
    """每日統計"""
    
    date: date
    
    # 盈虧
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    
    # 交易統計
    trade_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # 金額統計
    total_buy_value: float = 0.0
    total_sell_value: float = 0.0
    total_commission: float = 0.0
    
    # 高低水位
    peak_pnl: float = 0.0
    max_drawdown: float = 0.0
    
    def update_pnl(self, realized: float, unrealized: float) -> None:
        """更新盈虧"""
        self.realized_pnl = realized
        self.unrealized_pnl = unrealized
        self.total_pnl = realized + unrealized
        
        # 更新高低水位
        if self.total_pnl > self.peak_pnl:
            self.peak_pnl = self.total_pnl
        
        drawdown = self.peak_pnl - self.total_pnl
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
    
    def record_trade(self, is_buy: bool, value: float, pnl: float, commission: float) -> None:
        """記錄交易"""
        self.trade_count += 1
        self.total_commission += commission
        
        if is_buy:
            self.total_buy_value += value
        else:
            self.total_sell_value += value
        
        if pnl > 0:
            self.winning_trades += 1
        elif pnl < 0:
            self.losing_trades += 1


class RiskManager:
    """
    風險管理器
    
    提供全面的風險控制功能：
    - 信號風控檢查
    - 每日虧損上限
    - 持倉上限控制
    - 總曝險控制
    - 交易開關控制
    
    使用方式:
        risk_manager = RiskManager(config, event_bus)
        risk_manager.start()
        
        # 檢查信號
        result = risk_manager.check_signal(signal)
        if result.passed:
            # 執行交易
            pass
        else:
            print(f"風控拒絕: {result.reason}")
    """
    
    def __init__(
        self,
        config: Optional[RiskConfig] = None,
        event_bus: Optional[EventBus] = None,
        account_value: float = 100000.0,
    ):
        """
        初始化風險管理器
        
        Args:
            config: 風控配置
            event_bus: 事件總線
            account_value: 帳戶價值
        """
        self._config = config or RiskConfig()
        self._event_bus = event_bus or get_event_bus()
        self._account_value = account_value
        
        # 持倉管理: {symbol: PositionInfo}
        self._positions: Dict[str, PositionInfo] = {}
        
        # 每日統計
        self._daily_stats = DailyStats(date=date.today())
        
        # 交易控制
        self._trading_enabled = True
        self._disabled_reason = ""
        
        # 運行狀態
        self._running = False
        
        # 線程安全
        self._lock = threading.RLock()
        
        logger.info(
            f"RiskManager 初始化: max_daily_loss={self._config.max_daily_loss}, "
            f"max_position_size={self._config.max_position_size}"
        )
    
    # ========== 屬性 ==========
    
    @property
    def is_running(self) -> bool:
        """是否運行中"""
        return self._running
    
    @property
    def trading_enabled(self) -> bool:
        """交易是否啟用"""
        return self._trading_enabled
    
    @property
    def account_value(self) -> float:
        """帳戶價值"""
        return self._account_value
    
    @property
    def daily_pnl(self) -> float:
        """今日盈虧"""
        return self._daily_stats.total_pnl
    
    @property
    def total_exposure(self) -> float:
        """總曝險"""
        return sum(abs(pos.market_value) for pos in self._positions.values())
    
    @property
    def exposure_ratio(self) -> float:
        """曝險比率"""
        if self._account_value == 0:
            return 0.0
        return self.total_exposure / self._account_value
    
    # ========== 控制 ==========
    
    def start(self) -> None:
        """啟動風險管理器"""
        if self._running:
            logger.warning("RiskManager 已在運行中")
            return
        
        # 訂閱事件
        self._event_bus.subscribe(EventType.FILL, self._on_fill, priority=1)
        self._event_bus.subscribe(EventType.POSITION, self._on_position, priority=1)
        
        self._running = True
        logger.info("RiskManager 已啟動")
    
    def stop(self) -> None:
        """停止風險管理器"""
        if not self._running:
            return
        
        self._event_bus.unsubscribe(EventType.FILL, self._on_fill)
        self._event_bus.unsubscribe(EventType.POSITION, self._on_position)
        
        self._running = False
        logger.info("RiskManager 已停止")
    
    def enable_trading(self) -> None:
        """啟用交易"""
        with self._lock:
            self._trading_enabled = True
            self._disabled_reason = ""
            logger.info("交易已啟用")
    
    def disable_trading(self, reason: str = "") -> None:
        """停用交易"""
        with self._lock:
            self._trading_enabled = False
            self._disabled_reason = reason
            logger.warning(f"交易已停用: {reason}")
            
            self._emit_risk_event(
                RiskEventType.CIRCUIT_BREAKER_TRIGGERED,
                f"交易停用: {reason}",
            )
    
    # ========== 信號檢查 ==========
    
    def check_signal(self, signal: SignalEvent) -> RiskCheckResult:
        """
        檢查信號是否符合風控規則
        
        Args:
            signal: SignalEvent
            
        Returns:
            RiskCheckResult
        """
        warnings = []
        
        # 檢查交易是否啟用
        if not self._trading_enabled:
            return RiskCheckResult(
                passed=False,
                risk_level=RiskLevel.CRITICAL,
                reason=f"交易已停用: {self._disabled_reason}",
            )
        
        # 檢查每日虧損上限
        result = self._check_daily_loss_limit()
        if not result.passed:
            return result
        if result.has_warnings:
            warnings.extend(result.warnings)
        
        # 檢查單一標的持倉上限
        result = self._check_position_limit(signal)
        if not result.passed:
            return result
        if result.has_warnings:
            warnings.extend(result.warnings)
        
        # 檢查總曝險上限
        result = self._check_exposure_limit(signal)
        if not result.passed:
            return result
        if result.has_warnings:
            warnings.extend(result.warnings)
        
        # 檢查單筆交易金額
        result = self._check_trade_size(signal)
        if not result.passed:
            return result
        if result.has_warnings:
            warnings.extend(result.warnings)
        
        # 計算風險等級
        risk_level = self._calculate_risk_level()
        
        return RiskCheckResult(
            passed=True,
            risk_level=risk_level,
            warnings=warnings,
        )
    
    def _check_daily_loss_limit(self) -> RiskCheckResult:
        """檢查每日虧損上限"""
        max_loss = self._config.max_daily_loss
        current_loss = -self._daily_stats.total_pnl  # 虧損為正數
        
        if current_loss >= max_loss:
            self.disable_trading(f"達到每日虧損上限 ${max_loss}")
            
            self._emit_risk_event(
                RiskEventType.DAILY_LOSS_EXCEEDED,
                f"每日虧損 ${current_loss:.2f} 超過上限 ${max_loss}",
                current_value=current_loss,
                limit_value=max_loss,
            )
            
            return RiskCheckResult(
                passed=False,
                risk_level=RiskLevel.CRITICAL,
                reason=f"達到每日虧損上限 (${current_loss:.2f}/${max_loss})",
            )
        
        warnings = []
        # 警告：接近虧損上限
        if current_loss >= max_loss * 0.8:
            warnings.append(f"接近每日虧損上限 (${current_loss:.2f}/${max_loss})")
        
        return RiskCheckResult(passed=True, warnings=warnings)
    
    def _check_position_limit(self, signal: SignalEvent) -> RiskCheckResult:
        """檢查單一標的持倉上限"""
        symbol = signal.symbol
        max_position = self._config.max_position_size
        quantity = signal.suggested_quantity or 0
        
        # 取得當前持倉
        current_pos = self._positions.get(symbol)
        current_qty = current_pos.quantity if current_pos else 0
        
        # 計算交易後的持倉
        if signal.action == OrderAction.BUY:
            new_qty = current_qty + quantity
        else:
            new_qty = current_qty - quantity
        
        if abs(new_qty) > max_position:
            self._emit_risk_event(
                RiskEventType.POSITION_LIMIT_EXCEEDED,
                f"{symbol} 持倉 {new_qty} 超過上限 {max_position}",
                symbol=symbol,
                current_value=abs(new_qty),
                limit_value=max_position,
            )
            
            return RiskCheckResult(
                passed=False,
                risk_level=RiskLevel.HIGH,
                reason=f"持倉超過上限 ({abs(new_qty)}/{max_position})",
            )
        
        warnings = []
        # 警告：接近持倉上限
        if abs(new_qty) >= max_position * 0.8:
            warnings.append(f"{symbol} 接近持倉上限 ({abs(new_qty)}/{max_position})")
        
        return RiskCheckResult(passed=True, warnings=warnings)
    
    def _check_exposure_limit(self, signal: SignalEvent) -> RiskCheckResult:
        """檢查總曝險上限"""
        max_exposure = self._config.max_total_exposure
        
        # 估算交易後的曝險
        quantity = signal.suggested_quantity or 0
        price = signal.suggested_price or 0
        trade_value = quantity * price
        
        estimated_exposure = self.total_exposure + trade_value
        
        if estimated_exposure > max_exposure:
            self._emit_risk_event(
                RiskEventType.EXPOSURE_LIMIT_EXCEEDED,
                f"總曝險 ${estimated_exposure:.2f} 超過上限 ${max_exposure}",
                current_value=estimated_exposure,
                limit_value=max_exposure,
            )
            
            return RiskCheckResult(
                passed=False,
                risk_level=RiskLevel.HIGH,
                reason=f"總曝險超過上限 (${estimated_exposure:.2f}/${max_exposure})",
            )
        
        warnings = []
        if estimated_exposure >= max_exposure * 0.8:
            warnings.append(f"接近總曝險上限 (${estimated_exposure:.2f}/${max_exposure})")
        
        return RiskCheckResult(passed=True, warnings=warnings)
    
    def _check_trade_size(self, signal: SignalEvent) -> RiskCheckResult:
        """檢查單筆交易金額"""
        max_trade_value = self._config.max_order_value
        
        quantity = signal.suggested_quantity or 0
        price = signal.suggested_price or 0
        trade_value = quantity * price
        
        if trade_value > max_trade_value:
            return RiskCheckResult(
                passed=False,
                risk_level=RiskLevel.MEDIUM,
                reason=f"單筆交易金額超過上限 (${trade_value:.2f}/${max_trade_value})",
            )
        
        return RiskCheckResult(passed=True)
    
    def _calculate_risk_level(self) -> RiskLevel:
        """計算當前風險等級"""
        # 基於多個因素計算風險等級
        risk_score = 0
        
        # 因素 1: 曝險比率
        if self.exposure_ratio > 0.8:
            risk_score += 3
        elif self.exposure_ratio > 0.6:
            risk_score += 2
        elif self.exposure_ratio > 0.4:
            risk_score += 1
        
        # 因素 2: 每日虧損比率
        loss_ratio = -self._daily_stats.total_pnl / self._config.max_daily_loss
        if loss_ratio > 0.8:
            risk_score += 3
        elif loss_ratio > 0.5:
            risk_score += 2
        elif loss_ratio > 0.3:
            risk_score += 1
        
        # 因素 3: 最大回撤
        drawdown_ratio = self._daily_stats.max_drawdown / self._account_value
        if drawdown_ratio > 0.05:
            risk_score += 2
        elif drawdown_ratio > 0.02:
            risk_score += 1
        
        # 轉換為風險等級
        if risk_score >= 6:
            return RiskLevel.CRITICAL
        elif risk_score >= 4:
            return RiskLevel.HIGH
        elif risk_score >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    # ========== 事件處理 ==========
    
    def _on_fill(self, event: FillEvent) -> None:
        """處理成交事件"""
        with self._lock:
            symbol = event.symbol
            
            # 計算數量（買入為正，賣出為負）
            quantity = event.quantity
            if event.action == OrderAction.SELL:
                quantity = -quantity
            
            # 更新持倉
            if symbol not in self._positions:
                self._positions[symbol] = PositionInfo(symbol=symbol)
            
            pos = self._positions[symbol]
            realized_pnl = pos.update_from_fill(quantity, event.price)
            
            # 更新每日統計
            is_buy = event.action == OrderAction.BUY
            trade_value = event.quantity * event.price
            commission = event.commission or 0
            
            self._daily_stats.record_trade(is_buy, trade_value, realized_pnl, commission)
            self._daily_stats.realized_pnl += realized_pnl
            
            # 重新計算未實現盈虧
            total_unrealized = sum(p.unrealized_pnl for p in self._positions.values())
            self._daily_stats.update_pnl(
                self._daily_stats.realized_pnl,
                total_unrealized,
            )
            
            logger.debug(
                f"成交更新: {symbol} qty={event.quantity} @ {event.price}, "
                f"realized_pnl={realized_pnl:.2f}, daily_pnl={self._daily_stats.total_pnl:.2f}"
            )
    
    def _on_position(self, event: PositionEvent) -> None:
        """處理持倉更新事件"""
        with self._lock:
            symbol = event.symbol
            
            if symbol not in self._positions:
                self._positions[symbol] = PositionInfo(symbol=symbol)
            
            pos = self._positions[symbol]
            pos.quantity = event.quantity
            pos.avg_cost = event.avg_cost
            pos.market_price = event.market_price or pos.market_price
            pos.market_value = event.market_value or abs(pos.quantity * pos.market_price)
            pos.unrealized_pnl = event.unrealized_pnl or 0
    
    def _emit_risk_event(
        self,
        sub_type: RiskEventType,
        message: str,
        symbol: Optional[str] = None,
        strategy_id: Optional[str] = None,
        current_value: Optional[float] = None,
        limit_value: Optional[float] = None,
        action_taken: Optional[str] = None,
    ) -> None:
        """發布風控事件"""
        risk_event = RiskEvent(
            event_type=EventType.RISK,
            sub_type=sub_type,
            message=message,
            symbol=symbol,
            strategy_id=strategy_id,
            current_value=current_value,
            limit_value=limit_value,
            action_taken=action_taken,
        )
        
        self._event_bus.publish(risk_event)
        logger.warning(f"風控事件: {sub_type.name} - {message}")
    
    # ========== 查詢方法 ==========
    
    def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """取得持倉資訊"""
        return self._positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, PositionInfo]:
        """取得所有持倉"""
        return self._positions.copy()
    
    def get_daily_pnl(self) -> float:
        """取得今日盈虧"""
        return self._daily_stats.total_pnl
    
    def get_daily_stats(self) -> DailyStats:
        """取得今日統計"""
        return self._daily_stats
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """取得風險摘要"""
        return {
            "trading_enabled": self._trading_enabled,
            "disabled_reason": self._disabled_reason,
            "risk_level": self._calculate_risk_level().name,
            "account_value": self._account_value,
            "total_exposure": self.total_exposure,
            "exposure_ratio": f"{self.exposure_ratio:.2%}",
            "daily_pnl": self._daily_stats.total_pnl,
            "daily_realized_pnl": self._daily_stats.realized_pnl,
            "daily_unrealized_pnl": self._daily_stats.unrealized_pnl,
            "max_drawdown": self._daily_stats.max_drawdown,
            "trade_count": self._daily_stats.trade_count,
            "win_rate": (
                f"{self._daily_stats.winning_trades / self._daily_stats.trade_count:.2%}"
                if self._daily_stats.trade_count > 0 else "N/A"
            ),
            "position_count": len([p for p in self._positions.values() if not p.is_flat]),
        }
    
    # ========== 工具方法 ==========
    
    def update_account_value(self, value: float) -> None:
        """更新帳戶價值"""
        self._account_value = value
    
    def reset_daily_stats(self) -> None:
        """重置每日統計（每日開盤時調用）"""
        with self._lock:
            self._daily_stats = DailyStats(date=date.today())
            
            # 重置持倉的已實現盈虧
            for pos in self._positions.values():
                pos.realized_pnl = 0.0
            
            # 重新啟用交易
            if not self._trading_enabled and "每日虧損" in self._disabled_reason:
                self.enable_trading()
            
            logger.info("每日統計已重置")
    
    def sync_positions(self, positions: Dict[str, Dict[str, Any]]) -> None:
        """同步持倉（從 IB 獲取後調用）"""
        with self._lock:
            for symbol, data in positions.items():
                if symbol not in self._positions:
                    self._positions[symbol] = PositionInfo(symbol=symbol)
                
                pos = self._positions[symbol]
                pos.quantity = data.get("quantity", 0)
                pos.avg_cost = data.get("avg_cost", 0)
                pos.market_price = data.get("market_price", 0)
                pos.market_value = data.get("market_value", 0)
                pos.unrealized_pnl = data.get("unrealized_pnl", 0)
