"""
三角套利策略
Triangular Arbitrage Strategy

整合現貨與期貨的貴金屬三角套利策略
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .config import (
    TriangularArbitrageConfig,
    TriangleType,
    TRIANGLE_DEFINITIONS,
    SPOT_SYMBOLS,
    FUTURES_SYMBOLS,
    DEFAULT_CONFIG,
)
from .calculator import (
    TriangleCalculator,
    TrianglePositionManager,
    TriangleSignal,
    TriangleState,
)

logger = logging.getLogger(__name__)


class TriangularArbitrageStrategy:
    """
    三角套利策略主類
    
    支援：
    - 四種貴金屬（XAU, XAG, XPT, XPD）
    - 現貨和期貨
    - 多三角同時監控
    - 自動進出場管理
    """
    
    def __init__(
        self,
        config: Optional[TriangularArbitrageConfig] = None,
        event_bus: Optional[Any] = None,
    ):
        """
        初始化策略
        
        Args:
            config: 策略配置
            event_bus: 事件總線（用於與系統其他模組通訊）
        """
        self.config = config or DEFAULT_CONFIG
        self.event_bus = event_bus
        
        # 計算器和部位管理器
        self.calculator = TriangleCalculator(
            lookback_period=self.config.lookback_period,
            use_log_deviation=True,
        )
        self.position_manager = TrianglePositionManager()
        
        # 價格快取
        self._prices: Dict[str, float] = {}
        self._last_price_update: Optional[datetime] = None
        
        # 策略狀態
        self._is_running = False
        self._is_warming_up = True
        self._warmup_count = 0
        
        # 每日統計
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._last_trade_date: Optional[datetime] = None
        
        # 信號歷史
        self._signal_history: List[TriangleSignal] = []
        
        logger.info(f"TriangularArbitrageStrategy initialized")
        logger.info(f"Enabled triangles: {[t.value for t in self.config.enabled_triangles]}")
    
    # ==================== 主要接口 ====================
    
    def start(self) -> None:
        """啟動策略"""
        self._is_running = True
        self._is_warming_up = True
        self._warmup_count = 0
        logger.info("Strategy started")
    
    def stop(self) -> None:
        """停止策略"""
        self._is_running = False
        logger.info("Strategy stopped")
    
    def on_bar(self, bar_data: Dict[str, Any]) -> None:
        """
        處理 K 線數據（主要入口）
        
        Args:
            bar_data: K線數據，格式:
                {
                    "symbol": "XAUUSD",
                    "timestamp": datetime,
                    "open": float,
                    "high": float,
                    "low": float,
                    "close": float,
                    "volume": float,
                }
        """
        if not self._is_running:
            return
        
        # 更新價格
        symbol = bar_data.get("symbol", "")
        close_price = bar_data.get("close", 0)
        timestamp = bar_data.get("timestamp", datetime.utcnow())
        
        # 將 symbol 轉換為 asset key
        asset_key = self._symbol_to_asset(symbol)
        if asset_key and close_price > 0:
            self._prices[asset_key] = close_price
            self._last_price_update = timestamp
        
        # 檢查是否所有價格都已更新
        if not self._has_all_prices():
            return
        
        # 更新計算器
        self.calculator.update_prices(self._prices, timestamp)
        
        # 預熱期
        if self._is_warming_up:
            self._warmup_count += 1
            if self._warmup_count >= self.config.warmup_bars:
                self._is_warming_up = False
                logger.info("Warmup completed, strategy is now active")
            return
        
        # 重置每日統計
        self._check_new_day(timestamp)
        
        # 檢查時間過濾
        if not self._is_trading_time(timestamp):
            return
        
        # 更新持倉
        self.position_manager.update_positions(self._prices, timestamp)
        
        # 檢查出場條件
        self._check_exits(timestamp)
        
        # 檢查進場條件
        self._check_entries(timestamp)
    
    def on_tick(self, tick_data: Dict[str, Any]) -> None:
        """
        處理 Tick 數據（可選，用於更精確執行）
        """
        symbol = tick_data.get("symbol", "")
        price = tick_data.get("price", 0)
        
        asset_key = self._symbol_to_asset(symbol)
        if asset_key and price > 0:
            self._prices[asset_key] = price
    
    # ==================== 進出場邏輯 ====================
    
    def _check_entries(self, timestamp: datetime) -> None:
        """檢查進場條件"""
        # 檢查是否已達最大持倉數
        if self.position_manager.get_position_count() >= self.config.max_triangles:
            return
        
        # 檢查總曝險
        current_exposure = self.position_manager.get_total_exposure(self._prices)
        max_exposure = self.config.capital_per_triangle * self.config.max_triangles * self.config.max_exposure_pct
        if current_exposure >= max_exposure:
            return
        
        # 生成信號
        signals = self.calculator.generate_signals(
            entry_zscore=self.config.entry_zscore,
            min_deviation_pct=self.config.min_deviation_pct,
            enabled_triangles=self.config.enabled_triangles,
        )
        
        for signal in signals:
            # 檢查是否已有該三角的持倉
            if self.position_manager.has_position(signal.triangle_type):
                continue
            
            # 檢查每日虧損限制
            if self._daily_pnl < -self.config.daily_loss_limit_pct * self.config.capital_per_triangle:
                logger.warning("Daily loss limit reached, no new entries")
                break
            
            # 計算部位
            positions = self.calculator.calculate_positions(
                signal=signal,
                capital=self.config.capital_per_triangle,
                prices=self._prices,
                method="equal_dollar",
            )
            
            if not positions:
                continue
            
            # 開倉
            self._execute_entry(signal, positions, timestamp)
            
            # 只開一個新倉位（避免同時開太多）
            break
    
    def _check_exits(self, timestamp: datetime) -> None:
        """檢查出場條件"""
        positions_to_close = []
        positions_to_scale = []
        
        for tri_type, position in self.position_manager.open_positions.items():
            state = self.calculator.get_triangle_state(tri_type)
            
            # === 獲利出場 ===
            if abs(state.zscore) < self.config.exit_zscore:
                positions_to_close.append((tri_type, "profit_target"))
                continue
            
            # === 時間出場 ===
            if position.holding_days >= self.config.max_holding_days:
                positions_to_close.append((tri_type, "time_stop"))
                continue
            
            # === 逐層減倉/停損 ===
            for threshold_z, scale_pct in self.config.position_scale_out:
                if abs(state.zscore) > threshold_z:
                    if scale_pct >= 1.0:
                        # 全部平倉
                        positions_to_close.append((tri_type, "stop_loss"))
                    elif position.scale_out_level < len(self.config.position_scale_out):
                        # 部分減倉
                        positions_to_scale.append((tri_type, scale_pct))
                    break
            
            # === 單腿停損 ===
            if abs(position.current_pnl_pct) > self.config.single_leg_stop_pct:
                positions_to_close.append((tri_type, "single_leg_stop"))
        
        # 執行減倉
        for tri_type, scale_pct in positions_to_scale:
            self._execute_scale_out(tri_type, scale_pct, timestamp)
        
        # 執行平倉
        for tri_type, reason in positions_to_close:
            self._execute_exit(tri_type, reason, timestamp)
    
    # ==================== 執行函數 ====================
    
    def _execute_entry(
        self,
        signal: TriangleSignal,
        positions: Dict[str, float],
        timestamp: datetime,
    ) -> None:
        """執行進場"""
        logger.info(f"=== ENTRY SIGNAL ===")
        logger.info(f"Triangle: {signal.triangle_type.value}")
        logger.info(f"Direction: {signal.direction}")
        logger.info(f"Z-Score: {signal.zscore:.2f}")
        logger.info(f"Deviation: {signal.deviation_pct:.3f}%")
        logger.info(f"Positions: {positions}")
        
        # 記錄信號
        self._signal_history.append(signal)
        
        # 開倉
        self.position_manager.open_position(
            tri_type=signal.triangle_type,
            signal=signal,
            prices=self._prices,
        )
        
        # 發送訂單事件（如果有 event_bus）
        if self.event_bus:
            for symbol, units in positions.items():
                action = "BUY" if units > 0 else "SELL"
                self._emit_order_signal(
                    symbol=symbol,
                    action=action,
                    quantity=abs(units),
                    timestamp=timestamp,
                    reason=f"Triangle entry: {signal.triangle_type.value}",
                )
        
        self._daily_trades += 1
    
    def _execute_exit(
        self,
        tri_type: TriangleType,
        reason: str,
        timestamp: datetime,
    ) -> None:
        """執行出場"""
        position = self.position_manager.open_positions.get(tri_type)
        if not position:
            return
        
        logger.info(f"=== EXIT SIGNAL ===")
        logger.info(f"Triangle: {tri_type.value}")
        logger.info(f"Reason: {reason}")
        logger.info(f"PnL: {position.current_pnl:.2f}")
        logger.info(f"Holding days: {position.holding_days}")
        
        # 發送平倉訂單
        if self.event_bus:
            for symbol, units in position.positions.items():
                # 平倉方向與持倉相反
                action = "SELL" if units > 0 else "BUY"
                self._emit_order_signal(
                    symbol=symbol,
                    action=action,
                    quantity=abs(units),
                    timestamp=timestamp,
                    reason=f"Triangle exit: {tri_type.value}, {reason}",
                )
        
        # 更新統計
        pnl = self.position_manager.close_position(tri_type, self._prices, reason)
        self._daily_pnl += pnl
        self._daily_trades += 1
    
    def _execute_scale_out(
        self,
        tri_type: TriangleType,
        scale_pct: float,
        timestamp: datetime,
    ) -> None:
        """執行減倉"""
        position = self.position_manager.open_positions.get(tri_type)
        if not position:
            return
        
        logger.info(f"=== SCALE OUT ===")
        logger.info(f"Triangle: {tri_type.value}")
        logger.info(f"Scale: {scale_pct*100:.0f}%")
        
        # 發送減倉訂單
        if self.event_bus:
            for symbol, units in position.positions.items():
                reduce_units = abs(units) * scale_pct
                action = "SELL" if units > 0 else "BUY"
                self._emit_order_signal(
                    symbol=symbol,
                    action=action,
                    quantity=reduce_units,
                    timestamp=timestamp,
                    reason=f"Triangle scale out: {tri_type.value}",
                )
        
        self.position_manager.scale_out_position(tri_type, scale_pct)
    
    def _emit_order_signal(
        self,
        symbol: str,
        action: str,
        quantity: float,
        timestamp: datetime,
        reason: str,
    ) -> None:
        """發送訂單信號到事件總線"""
        if not self.event_bus:
            return
        
        # 這裡需要根據你的系統架構調整
        # 假設使用 SignalEvent
        try:
            from core.events import SignalEvent, OrderAction
            
            order_action = OrderAction.BUY if action == "BUY" else OrderAction.SELL
            
            signal_event = SignalEvent(
                strategy_id=self.config.strategy_id,
                symbol=symbol,
                action=order_action,
                suggested_quantity=quantity,
                timestamp=timestamp,
                metadata={"reason": reason},
            )
            
            self.event_bus.publish(signal_event)
            
        except ImportError:
            logger.warning("Could not import SignalEvent, order not emitted")
    
    # ==================== 輔助函數 ====================
    
    def _symbol_to_asset(self, symbol: str) -> Optional[str]:
        """將 IB symbol 轉換為 asset key"""
        for asset, spot_symbol in SPOT_SYMBOLS.items():
            if symbol == spot_symbol or symbol.startswith(spot_symbol):
                return asset
        return None
    
    def _has_all_prices(self) -> bool:
        """檢查是否有所有需要的價格"""
        required_assets = set()
        for tri_type in self.config.enabled_triangles:
            defn = TRIANGLE_DEFINITIONS[tri_type]
            required_assets.add(defn.asset_a)
            required_assets.add(defn.asset_b)
            required_assets.add(defn.asset_c)
        
        for asset in required_assets:
            if asset not in self._prices or self._prices[asset] <= 0:
                return False
        
        return True
    
    def _is_trading_time(self, timestamp: datetime) -> bool:
        """檢查是否在交易時間內"""
        hour = timestamp.hour
        start_hour, end_hour = self.config.trading_hours_utc
        
        if start_hour <= end_hour:
            return start_hour <= hour < end_hour
        else:
            # 跨午夜
            return hour >= start_hour or hour < end_hour
    
    def _check_new_day(self, timestamp: datetime) -> None:
        """檢查是否新的一天，重置每日統計"""
        if self._last_trade_date is None or timestamp.date() != self._last_trade_date.date():
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._last_trade_date = timestamp
            logger.info(f"New trading day: {timestamp.date()}")
    
    # ==================== 狀態查詢 ====================
    
    def get_status(self) -> Dict[str, Any]:
        """取得策略狀態"""
        return {
            "is_running": self._is_running,
            "is_warming_up": self._is_warming_up,
            "warmup_progress": f"{self._warmup_count}/{self.config.warmup_bars}",
            "open_positions": self.position_manager.get_position_count(),
            "total_pnl": self.position_manager.total_pnl,
            "daily_pnl": self._daily_pnl,
            "daily_trades": self._daily_trades,
            "prices": self._prices.copy(),
        }
    
    def get_triangle_states(self) -> Dict[str, Dict[str, Any]]:
        """取得所有三角的狀態"""
        result = {}
        
        for tri_type in self.config.enabled_triangles:
            state = self.calculator.get_triangle_state(tri_type)
            result[tri_type.value] = {
                "is_valid": state.is_valid,
                "ratio_ab": round(state.ratio_ab, 4),
                "ratio_bc": round(state.ratio_bc, 4),
                "ratio_ac": round(state.ratio_ac, 4),
                "implied_ac": round(state.implied_ac, 4),
                "deviation_pct": round(state.deviation_pct, 4),
                "zscore": round(state.zscore, 2),
                "has_position": self.position_manager.has_position(tri_type),
            }
        
        return result
    
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """取得當前持倉"""
        positions = []
        
        for tri_type, pos in self.position_manager.open_positions.items():
            positions.append({
                "triangle": tri_type.value,
                "entry_time": pos.entry_time.isoformat() if pos.entry_time else None,
                "entry_zscore": round(pos.entry_zscore, 2),
                "holding_days": pos.holding_days,
                "current_pnl": round(pos.current_pnl, 2),
                "current_pnl_pct": round(pos.current_pnl_pct, 2),
                "positions": {k: round(v, 4) for k, v in pos.positions.items()},
            })
        
        return positions
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """取得績效摘要"""
        closed = self.position_manager.closed_positions
        
        if not closed:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_pnl": 0,
            }
        
        wins = [p for p in closed if p.current_pnl > 0]
        losses = [p for p in closed if p.current_pnl <= 0]
        
        total_pnl = sum(p.current_pnl for p in closed)
        avg_pnl = total_pnl / len(closed) if closed else 0
        avg_holding = np.mean([p.holding_days for p in closed]) if closed else 0
        
        return {
            "total_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(closed) * 100 if closed else 0,
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(avg_pnl, 2),
            "avg_holding_days": round(avg_holding, 1),
            "best_trade": round(max(p.current_pnl for p in closed), 2) if closed else 0,
            "worst_trade": round(min(p.current_pnl for p in closed), 2) if closed else 0,
        }


# ==================== 策略工廠函數 ====================

def create_strategy(
    config: Optional[Dict[str, Any]] = None,
    event_bus: Optional[Any] = None,
) -> TriangularArbitrageStrategy:
    """
    創建策略實例的工廠函數
    
    Args:
        config: 配置字典（可選）
        event_bus: 事件總線
        
    Returns:
        策略實例
    """
    if config:
        # 從字典創建配置
        strategy_config = TriangularArbitrageConfig(
            strategy_id=config.get("strategy_id", "triangular_arbitrage"),
            enabled_triangles=[
                TriangleType[t] for t in config.get("enabled_triangles", ["T1_XAU_XAG_XPT"])
            ],
            lookback_period=config.get("lookback_period", 120),
            entry_zscore=config.get("entry_zscore", 2.0),
            exit_zscore=config.get("exit_zscore", 0.5),
            stop_zscore=config.get("stop_zscore", 3.5),
            capital_per_triangle=config.get("capital_per_triangle", 50000),
            max_triangles=config.get("max_triangles", 3),
        )
    else:
        strategy_config = DEFAULT_CONFIG
    
    return TriangularArbitrageStrategy(
        config=strategy_config,
        event_bus=event_bus,
    )
