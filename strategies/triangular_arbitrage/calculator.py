"""
三角套利計算工具
Triangular Arbitrage Calculator
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import logging

from .config import (
    TriangleType, 
    TriangleDefinition, 
    TRIANGLE_DEFINITIONS,
    SPOT_SYMBOLS,
    FUTURES_SYMBOLS,
)

logger = logging.getLogger(__name__)


@dataclass
class TriangleState:
    """單一三角的狀態"""
    triangle_type: TriangleType
    definition: TriangleDefinition
    
    # 當前價格
    price_a: float = 0.0
    price_b: float = 0.0
    price_c: float = 0.0
    
    # 計算結果
    ratio_ab: float = 0.0      # A/B
    ratio_bc: float = 0.0      # B/C
    ratio_ac: float = 0.0      # A/C (實際)
    implied_ac: float = 0.0    # A/C (隱含)
    deviation: float = 0.0     # 偏離度
    deviation_pct: float = 0.0 # 偏離百分比
    zscore: float = 0.0        # Z-Score
    
    # 歷史數據
    deviation_history: deque = field(default_factory=lambda: deque(maxlen=500))
    
    # 時間戳
    last_update: Optional[datetime] = None
    
    @property
    def is_valid(self) -> bool:
        """檢查數據是否有效"""
        return (
            self.price_a > 0 and 
            self.price_b > 0 and 
            self.price_c > 0 and
            self.last_update is not None
        )


@dataclass
class TriangleSignal:
    """三角交易信號"""
    triangle_type: TriangleType
    timestamp: datetime
    
    # 信號方向
    direction: str  # "short_deviation" or "long_deviation"
    
    # 信號強度
    zscore: float
    deviation_pct: float
    
    # 建議部位
    positions: Dict[str, float] = field(default_factory=dict)
    # 格式: {"XAUUSD": -100, "XAGUSD": 5000, "XPTUSD": 50}
    
    # 預期
    expected_profit_pct: float = 0.0
    half_life_days: float = 0.0
    
    # 信號評分 (用於排序)
    score: float = 0.0


class TriangleCalculator:
    """三角套利計算器"""
    
    def __init__(
        self,
        lookback_period: int = 120,
        use_log_deviation: bool = True,
    ):
        """
        初始化
        
        Args:
            lookback_period: Z-Score 計算回顧期
            use_log_deviation: 是否使用對數偏離（更穩定）
        """
        self.lookback_period = lookback_period
        self.use_log_deviation = use_log_deviation
        
        # 各三角的狀態
        self.triangle_states: Dict[TriangleType, TriangleState] = {}
        
        # 初始化所有三角狀態
        for tri_type, definition in TRIANGLE_DEFINITIONS.items():
            self.triangle_states[tri_type] = TriangleState(
                triangle_type=tri_type,
                definition=definition,
            )
    
    def update_prices(
        self,
        prices: Dict[str, float],
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        更新價格並重新計算所有三角
        
        Args:
            prices: 價格字典 {"XAU": 2000.0, "XAG": 25.0, ...}
            timestamp: 時間戳
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        for tri_type, state in self.triangle_states.items():
            defn = state.definition
            
            # 取得價格
            price_a = prices.get(defn.asset_a)
            price_b = prices.get(defn.asset_b)
            price_c = prices.get(defn.asset_c)
            
            if price_a and price_b and price_c:
                state.price_a = price_a
                state.price_b = price_b
                state.price_c = price_c
                state.last_update = timestamp
                
                # 計算比率和偏離
                self._calculate_deviation(state)
    
    def _calculate_deviation(self, state: TriangleState) -> None:
        """
        計算單一三角的偏離度
        
        策略邏輯：追蹤 A/B 比率相對於其歷史均值的偏離
        當比率過度偏離歷史均值時，預期會回歸
        """
        # 計算比率
        state.ratio_ab = state.price_a / state.price_b
        state.ratio_bc = state.price_b / state.price_c
        state.ratio_ac = state.price_a / state.price_c
        
        # 隱含比率（用於參考）
        state.implied_ac = state.ratio_ab * state.ratio_bc
        
        # 使用主要比率 (A/B) 的對數值作為偏離追蹤指標
        if self.use_log_deviation:
            current_log_ratio = np.log(state.ratio_ab)
        else:
            current_log_ratio = state.ratio_ab
        
        # 更新歷史
        state.deviation_history.append(current_log_ratio)
        
        # 計算 Z-Score（基於比率偏離歷史均值的程度）
        if len(state.deviation_history) >= self.lookback_period:
            history = list(state.deviation_history)[-self.lookback_period:]
            mean = np.mean(history)
            std = np.std(history)
            
            if std > 0:
                state.zscore = (current_log_ratio - mean) / std
                # 偏離度 = 當前值相對於均值的百分比變化
                if self.use_log_deviation:
                    state.deviation = current_log_ratio - mean
                    state.deviation_pct = (np.exp(state.deviation) - 1) * 100
                else:
                    state.deviation = (current_log_ratio - mean) / mean
                    state.deviation_pct = state.deviation * 100
            else:
                state.zscore = 0.0
                state.deviation = 0.0
                state.deviation_pct = 0.0
        else:
            state.zscore = 0.0
            state.deviation = 0.0
            state.deviation_pct = 0.0
    def get_triangle_state(self, tri_type: TriangleType) -> TriangleState:
        """取得指定三角的狀態"""
        return self.triangle_states[tri_type]
    
    def get_all_states(self) -> Dict[TriangleType, TriangleState]:
        """取得所有三角狀態"""
        return self.triangle_states
    
    def generate_signals(
        self,
        entry_zscore: float = 2.0,
        min_deviation_pct: float = 0.5,
        enabled_triangles: Optional[List[TriangleType]] = None,
    ) -> List[TriangleSignal]:
        """
        生成交易信號
        
        Args:
            entry_zscore: 進場 Z-Score 門檻
            min_deviation_pct: 最小偏離百分比
            enabled_triangles: 啟用的三角列表
            
        Returns:
            符合條件的信號列表（按評分排序）
        """
        signals = []
        
        for tri_type, state in self.triangle_states.items():
            # 檢查是否啟用
            if enabled_triangles and tri_type not in enabled_triangles:
                continue
            
            # 檢查數據是否有效
            if not state.is_valid:
                continue
            
            # 檢查是否有足夠歷史數據
            if len(state.deviation_history) < self.lookback_period:
                continue
            
            # 檢查信號條件
            if abs(state.zscore) < entry_zscore:
                continue
            
            if abs(state.deviation_pct) < min_deviation_pct:
                continue
            
            # 確定方向
            if state.zscore > 0:
                direction = "short_deviation"
            else:
                direction = "long_deviation"
            
            # 計算信號評分
            score = self._calculate_signal_score(state)
            
            signal = TriangleSignal(
                triangle_type=tri_type,
                timestamp=state.last_update,
                direction=direction,
                zscore=state.zscore,
                deviation_pct=state.deviation_pct,
                expected_profit_pct=abs(state.deviation_pct) * 0.7,  # 假設能捕捉 70%
                half_life_days=state.definition.half_life_days,
                score=score,
            )
            
            signals.append(signal)
        
        # 按評分排序
        signals.sort(key=lambda x: x.score, reverse=True)
        
        return signals
    
    def _calculate_signal_score(self, state: TriangleState) -> float:
        """
        計算信號評分
        
        考慮因素：
        - Z-Score 絕對值（越大越好）
        - 流動性評分
        - 歷史半衰期（越短越好）
        """
        z_score_factor = min(abs(state.zscore), 5) / 5  # 0-1
        liquidity_factor = state.definition.liquidity_score / 5  # 0-1
        half_life_factor = max(0, 1 - state.definition.half_life_days / 20)  # 0-1
        
        # 加權組合
        score = (
            z_score_factor * 0.5 +
            liquidity_factor * 0.3 +
            half_life_factor * 0.2
        )
        
        return score
    
    def calculate_positions(
        self,
        signal: TriangleSignal,
        capital: float,
        prices: Dict[str, float],
        method: str = "equal_dollar",
    ) -> Dict[str, float]:
        """
        計算三腿部位
        
        Args:
            signal: 交易信號
            capital: 分配給此三角的資金
            prices: 當前價格
            method: 部位計算方法
                - "equal_dollar": 等美元價值
                - "volatility_weighted": 波動率加權
                
        Returns:
            部位字典 {"XAUUSD": 數量, ...}
        """
        defn = TRIANGLE_DEFINITIONS[signal.triangle_type]
        
        # 取得現貨代碼
        symbol_a = SPOT_SYMBOLS[defn.asset_a]
        symbol_b = SPOT_SYMBOLS[defn.asset_b]
        symbol_c = SPOT_SYMBOLS[defn.asset_c]
        
        price_a = prices.get(defn.asset_a, 0)
        price_b = prices.get(defn.asset_b, 0)
        price_c = prices.get(defn.asset_c, 0)
        
        if not (price_a > 0 and price_b > 0 and price_c > 0):
            return {}
        
        if method == "equal_dollar":
            # 每腿等美元價值
            capital_per_leg = capital / 3
            
            units_a = capital_per_leg / price_a
            units_b = capital_per_leg / price_b
            units_c = capital_per_leg / price_c
            
        else:
            # 預設使用等美元
            capital_per_leg = capital / 3
            units_a = capital_per_leg / price_a
            units_b = capital_per_leg / price_b
            units_c = capital_per_leg / price_c
        
        # 根據方向設定正負
        if signal.direction == "short_deviation":
            # 實際 > 隱含，做空 A/C ratio
            # 需要：做空 A，做多 B，做多 C
            positions = {
                symbol_a: -units_a,
                symbol_b: +units_b,
                symbol_c: +units_c,
            }
        else:
            # 實際 < 隱含，做多 A/C ratio
            positions = {
                symbol_a: +units_a,
                symbol_b: -units_b,
                symbol_c: -units_c,
            }
        
        signal.positions = positions
        return positions


class TrianglePositionManager:
    """三角部位管理器"""
    
    @dataclass
    class TrianglePosition:
        """單一三角持倉"""
        triangle_type: TriangleType
        entry_time: datetime
        entry_zscore: float
        entry_deviation: float
        positions: Dict[str, float]  # symbol -> units
        entry_prices: Dict[str, float]  # symbol -> price
        
        # 狀態追蹤
        current_pnl: float = 0.0
        current_pnl_pct: float = 0.0
        holding_days: int = 0
        scale_out_level: int = 0  # 已減倉次數
    
    def __init__(self):
        self.open_positions: Dict[TriangleType, TrianglePositionManager.TrianglePosition] = {}
        self.closed_positions: List[TrianglePositionManager.TrianglePosition] = []
        self.total_pnl: float = 0.0
    
    def has_position(self, tri_type: TriangleType) -> bool:
        """檢查是否已有該三角的持倉"""
        return tri_type in self.open_positions
    
    def open_position(
        self,
        tri_type: TriangleType,
        signal: TriangleSignal,
        prices: Dict[str, float],
    ) -> None:
        """開倉"""
        if self.has_position(tri_type):
            logger.warning(f"Already have position for {tri_type}")
            return
        
        defn = TRIANGLE_DEFINITIONS[tri_type]
        entry_prices = {
            SPOT_SYMBOLS[defn.asset_a]: prices.get(defn.asset_a, 0),
            SPOT_SYMBOLS[defn.asset_b]: prices.get(defn.asset_b, 0),
            SPOT_SYMBOLS[defn.asset_c]: prices.get(defn.asset_c, 0),
        }
        
        position = self.TrianglePosition(
            triangle_type=tri_type,
            entry_time=signal.timestamp,
            entry_zscore=signal.zscore,
            entry_deviation=signal.deviation_pct,
            positions=signal.positions.copy(),
            entry_prices=entry_prices,
        )
        
        self.open_positions[tri_type] = position
        logger.info(f"Opened triangle position: {tri_type.value}")
    
    def close_position(
        self,
        tri_type: TriangleType,
        prices: Dict[str, float],
        reason: str = "signal",
    ) -> float:
        """
        平倉
        
        Returns:
            實現損益
        """
        if not self.has_position(tri_type):
            return 0.0
        
        position = self.open_positions[tri_type]
        
        # 計算最終 PnL
        pnl = self._calculate_pnl(position, prices)
        position.current_pnl = pnl
        
        # 移到已平倉列表
        self.closed_positions.append(position)
        del self.open_positions[tri_type]
        
        self.total_pnl += pnl
        
        logger.info(f"Closed triangle position: {tri_type.value}, PnL: {pnl:.2f}, Reason: {reason}")
        
        return pnl
    
    def scale_out_position(
        self,
        tri_type: TriangleType,
        scale_pct: float,
    ) -> None:
        """減倉"""
        if not self.has_position(tri_type):
            return
        
        position = self.open_positions[tri_type]
        
        for symbol in position.positions:
            position.positions[symbol] *= (1 - scale_pct)
        
        position.scale_out_level += 1
        logger.info(f"Scaled out {scale_pct*100:.0f}% of {tri_type.value}")
    
    def update_positions(
        self,
        prices: Dict[str, float],
        current_time: datetime,
    ) -> None:
        """更新所有持倉狀態"""
        for tri_type, position in self.open_positions.items():
            # 更新 PnL
            pnl = self._calculate_pnl(position, prices)
            position.current_pnl = pnl
            
            # 計算初始名義價值
            initial_value = sum(
                abs(units) * position.entry_prices.get(symbol, 0)
                for symbol, units in position.positions.items()
            )
            if initial_value > 0:
                position.current_pnl_pct = (pnl / initial_value) * 100
            
            # 更新持倉天數
            if position.entry_time:
                delta = current_time - position.entry_time
                position.holding_days = delta.days
    
    def _calculate_pnl(
        self,
        position: 'TrianglePositionManager.TrianglePosition',
        current_prices: Dict[str, float],
    ) -> float:
        """計算持倉損益"""
        pnl = 0.0
        
        defn = TRIANGLE_DEFINITIONS[position.triangle_type]
        
        # 將 asset key 轉為 symbol
        price_map = {
            SPOT_SYMBOLS[defn.asset_a]: current_prices.get(defn.asset_a, 0),
            SPOT_SYMBOLS[defn.asset_b]: current_prices.get(defn.asset_b, 0),
            SPOT_SYMBOLS[defn.asset_c]: current_prices.get(defn.asset_c, 0),
        }
        
        for symbol, units in position.positions.items():
            entry_price = position.entry_prices.get(symbol, 0)
            current_price = price_map.get(symbol, 0)
            
            if entry_price > 0 and current_price > 0:
                # units > 0 表示做多，< 0 表示做空
                pnl += units * (current_price - entry_price)
        
        return pnl
    
    def get_position_count(self) -> int:
        """取得當前持倉數量"""
        return len(self.open_positions)
    
    def get_total_exposure(self, prices: Dict[str, float]) -> float:
        """計算總曝險"""
        exposure = 0.0
        
        for position in self.open_positions.values():
            defn = TRIANGLE_DEFINITIONS[position.triangle_type]
            price_map = {
                SPOT_SYMBOLS[defn.asset_a]: prices.get(defn.asset_a, 0),
                SPOT_SYMBOLS[defn.asset_b]: prices.get(defn.asset_b, 0),
                SPOT_SYMBOLS[defn.asset_c]: prices.get(defn.asset_c, 0),
            }
            
            for symbol, units in position.positions.items():
                price = price_map.get(symbol, 0)
                exposure += abs(units) * price
        
        return exposure