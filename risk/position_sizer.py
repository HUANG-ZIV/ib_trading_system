"""
PositionSizer 模組 - 倉位計算器

提供多種倉位計算方法
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Dict, Any
import math

from core.events import SignalEvent


# 設定 logger
logger = logging.getLogger(__name__)


class SizingMethod(Enum):
    """倉位計算方法"""
    
    FIXED = auto()           # 固定數量
    PERCENT = auto()         # 帳戶百分比
    ATR = auto()             # 基於 ATR 和停損距離
    KELLY = auto()           # 凱利公式
    RISK_PARITY = auto()     # 風險平價
    VOLATILITY = auto()      # 波動率調整


@dataclass
class PositionSize:
    """倉位計算結果"""
    
    quantity: int              # 建議數量
    risk_amount: float         # 風險金額
    method: SizingMethod       # 使用的計算方法
    
    # 計算細節
    price: float = 0.0         # 進場價格
    stop_loss: Optional[float] = None  # 停損價格
    risk_per_share: float = 0.0        # 每股風險
    
    # 額外資訊
    position_value: float = 0.0        # 倉位價值
    account_risk_pct: float = 0.0      # 帳戶風險百分比
    
    # 調整資訊
    was_adjusted: bool = False          # 是否被調整
    adjustment_reason: str = ""         # 調整原因
    original_quantity: int = 0          # 原始數量（調整前）
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            "quantity": self.quantity,
            "risk_amount": self.risk_amount,
            "method": self.method.name,
            "price": self.price,
            "stop_loss": self.stop_loss,
            "risk_per_share": self.risk_per_share,
            "position_value": self.position_value,
            "account_risk_pct": f"{self.account_risk_pct:.2%}",
            "was_adjusted": self.was_adjusted,
            "adjustment_reason": self.adjustment_reason,
        }


class PositionSizer:
    """
    倉位計算器
    
    提供多種倉位計算方法，根據風險參數計算適當的交易數量
    
    使用方式:
        sizer = PositionSizer(
            account_value=100000,
            risk_per_trade=0.01,  # 每筆交易風險 1%
            default_method=SizingMethod.ATR,
        )
        
        # 計算倉位
        size = sizer.calculate(
            price=150.0,
            stop_loss=145.0,
        )
        print(f"建議數量: {size.quantity}")
    """
    
    def __init__(
        self,
        account_value: float = 100000.0,
        risk_per_trade: float = 0.01,
        max_position_pct: float = 0.1,
        default_method: SizingMethod = SizingMethod.ATR,
        min_quantity: int = 1,
        max_quantity: int = 10000,
        round_lot_size: int = 1,
    ):
        """
        初始化倉位計算器
        
        Args:
            account_value: 帳戶價值
            risk_per_trade: 每筆交易的風險比例（0.01 = 1%）
            max_position_pct: 單一倉位最大佔帳戶比例
            default_method: 預設計算方法
            min_quantity: 最小數量
            max_quantity: 最大數量
            round_lot_size: 整批數量（用於向下取整）
        """
        self._account_value = account_value
        self._risk_per_trade = risk_per_trade
        self._max_position_pct = max_position_pct
        self._default_method = default_method
        self._min_quantity = min_quantity
        self._max_quantity = max_quantity
        self._round_lot_size = round_lot_size
        
        # 固定數量模式的數量
        self._fixed_quantity = 100
        
        # Kelly 公式參數
        self._kelly_win_rate = 0.5
        self._kelly_win_loss_ratio = 1.5
        self._kelly_fraction = 0.25  # 使用部分 Kelly
        
        # 波動率參數
        self._target_volatility = 0.02  # 目標日波動率 2%
        
        logger.info(
            f"PositionSizer 初始化: account={account_value}, "
            f"risk_per_trade={risk_per_trade:.2%}, method={default_method.name}"
        )
    
    # ========== 參數設定 ==========
    
    def set_account_value(self, value: float) -> None:
        """設定帳戶價值"""
        self._account_value = value
        logger.debug(f"帳戶價值更新: {value}")
    
    def set_risk_per_trade(self, risk: float) -> None:
        """設定每筆交易風險比例"""
        self._risk_per_trade = risk
        logger.debug(f"交易風險比例更新: {risk:.2%}")
    
    def set_fixed_quantity(self, quantity: int) -> None:
        """設定固定數量"""
        self._fixed_quantity = quantity
    
    def set_kelly_params(
        self,
        win_rate: float,
        win_loss_ratio: float,
        fraction: float = 0.25,
    ) -> None:
        """
        設定 Kelly 公式參數
        
        Args:
            win_rate: 勝率 (0-1)
            win_loss_ratio: 盈虧比
            fraction: Kelly 分數（建議 0.25-0.5）
        """
        self._kelly_win_rate = win_rate
        self._kelly_win_loss_ratio = win_loss_ratio
        self._kelly_fraction = fraction
    
    def set_target_volatility(self, volatility: float) -> None:
        """設定目標波動率"""
        self._target_volatility = volatility
    
    # ========== 主要計算方法 ==========
    
    def calculate(
        self,
        price: float,
        stop_loss: Optional[float] = None,
        atr: Optional[float] = None,
        volatility: Optional[float] = None,
        method: Optional[SizingMethod] = None,
        signal: Optional[SignalEvent] = None,
    ) -> PositionSize:
        """
        計算倉位大小
        
        Args:
            price: 進場價格
            stop_loss: 停損價格
            atr: ATR 值（用於 ATR 方法）
            volatility: 波動率（用於波動率方法）
            method: 計算方法，None 使用預設
            signal: SignalEvent（可從中提取參數）
            
        Returns:
            PositionSize 結果
        """
        # 從 signal 提取參數
        if signal:
            price = signal.suggested_price or price
            stop_loss = signal.stop_loss or stop_loss
        
        # 選擇方法
        sizing_method = method or self._default_method
        
        # 根據方法計算
        if sizing_method == SizingMethod.FIXED:
            result = self._fixed_size(price)
        elif sizing_method == SizingMethod.PERCENT:
            result = self._percent_size(price)
        elif sizing_method == SizingMethod.ATR:
            result = self._atr_size(price, stop_loss, atr)
        elif sizing_method == SizingMethod.KELLY:
            result = self._kelly_size(price, stop_loss)
        elif sizing_method == SizingMethod.VOLATILITY:
            result = self._volatility_size(price, volatility)
        elif sizing_method == SizingMethod.RISK_PARITY:
            result = self._risk_parity_size(price, volatility)
        else:
            result = self._fixed_size(price)
        
        # 應用限制
        result = self._apply_limits(result, price)
        
        return result
    
    def calculate_from_signal(self, signal: SignalEvent) -> PositionSize:
        """從 SignalEvent 計算倉位"""
        return self.calculate(
            price=signal.suggested_price or 0,
            stop_loss=signal.stop_loss,
            signal=signal,
        )
    
    # ========== 各種計算方法 ==========
    
    def _fixed_size(self, price: float) -> PositionSize:
        """
        固定數量方法
        
        直接返回預設的固定數量
        """
        quantity = self._fixed_quantity
        position_value = quantity * price
        risk_amount = self._account_value * self._risk_per_trade
        
        return PositionSize(
            quantity=quantity,
            risk_amount=risk_amount,
            method=SizingMethod.FIXED,
            price=price,
            position_value=position_value,
            account_risk_pct=position_value / self._account_value if self._account_value > 0 else 0,
        )
    
    def _percent_size(self, price: float) -> PositionSize:
        """
        帳戶百分比方法
        
        根據帳戶價值的百分比計算倉位
        """
        if price <= 0:
            return self._empty_result(SizingMethod.PERCENT, price)
        
        # 計算最大倉位價值
        max_position_value = self._account_value * self._max_position_pct
        
        # 計算數量
        quantity = int(max_position_value / price)
        position_value = quantity * price
        
        # 風險金額（假設使用預設風險比例）
        risk_amount = position_value * self._risk_per_trade
        
        return PositionSize(
            quantity=quantity,
            risk_amount=risk_amount,
            method=SizingMethod.PERCENT,
            price=price,
            position_value=position_value,
            account_risk_pct=position_value / self._account_value,
        )
    
    def _atr_size(
        self,
        price: float,
        stop_loss: Optional[float] = None,
        atr: Optional[float] = None,
    ) -> PositionSize:
        """
        ATR / 停損距離方法
        
        根據停損距離計算倉位，使每筆交易的風險金額固定
        
        公式: quantity = risk_amount / risk_per_share
        """
        if price <= 0:
            return self._empty_result(SizingMethod.ATR, price)
        
        # 計算風險金額
        risk_amount = self._account_value * self._risk_per_trade
        
        # 計算每股風險
        if stop_loss is not None and stop_loss > 0:
            risk_per_share = abs(price - stop_loss)
        elif atr is not None and atr > 0:
            # 使用 2x ATR 作為預設停損距離
            risk_per_share = atr * 2
        else:
            # 預設使用 2% 的價格作為風險
            risk_per_share = price * 0.02
        
        if risk_per_share <= 0:
            return self._empty_result(SizingMethod.ATR, price)
        
        # 計算數量
        quantity = int(risk_amount / risk_per_share)
        position_value = quantity * price
        
        return PositionSize(
            quantity=quantity,
            risk_amount=risk_amount,
            method=SizingMethod.ATR,
            price=price,
            stop_loss=stop_loss,
            risk_per_share=risk_per_share,
            position_value=position_value,
            account_risk_pct=risk_amount / self._account_value,
        )
    
    def _kelly_size(
        self,
        price: float,
        stop_loss: Optional[float] = None,
    ) -> PositionSize:
        """
        Kelly 公式方法
        
        Kelly % = W - (1-W)/R
        其中 W = 勝率, R = 盈虧比
        """
        if price <= 0:
            return self._empty_result(SizingMethod.KELLY, price)
        
        # 計算 Kelly 百分比
        w = self._kelly_win_rate
        r = self._kelly_win_loss_ratio
        
        if r <= 0:
            kelly_pct = 0
        else:
            kelly_pct = w - (1 - w) / r
        
        # 使用部分 Kelly（降低風險）
        kelly_pct = max(0, kelly_pct) * self._kelly_fraction
        
        # 計算倉位價值
        position_value = self._account_value * kelly_pct
        quantity = int(position_value / price)
        
        # 計算風險金額
        if stop_loss and stop_loss > 0:
            risk_per_share = abs(price - stop_loss)
            risk_amount = quantity * risk_per_share
        else:
            risk_amount = position_value * self._risk_per_trade
            risk_per_share = price * self._risk_per_trade
        
        return PositionSize(
            quantity=quantity,
            risk_amount=risk_amount,
            method=SizingMethod.KELLY,
            price=price,
            stop_loss=stop_loss,
            risk_per_share=risk_per_share,
            position_value=position_value,
            account_risk_pct=kelly_pct,
        )
    
    def _volatility_size(
        self,
        price: float,
        volatility: Optional[float] = None,
    ) -> PositionSize:
        """
        波動率調整方法
        
        根據資產波動率調整倉位大小，使組合波動率接近目標
        """
        if price <= 0:
            return self._empty_result(SizingMethod.VOLATILITY, price)
        
        # 使用提供的波動率或預設值
        vol = volatility or 0.02  # 預設 2% 日波動率
        
        if vol <= 0:
            vol = 0.02
        
        # 計算調整係數
        vol_adjustment = self._target_volatility / vol
        
        # 基礎倉位價值
        base_position_value = self._account_value * self._max_position_pct
        
        # 調整後的倉位價值
        adjusted_value = base_position_value * vol_adjustment
        
        # 限制最大倉位
        max_value = self._account_value * self._max_position_pct * 2  # 最多 2 倍
        adjusted_value = min(adjusted_value, max_value)
        
        quantity = int(adjusted_value / price)
        position_value = quantity * price
        
        # 風險金額
        risk_amount = position_value * vol  # 預期日風險
        
        return PositionSize(
            quantity=quantity,
            risk_amount=risk_amount,
            method=SizingMethod.VOLATILITY,
            price=price,
            position_value=position_value,
            account_risk_pct=position_value / self._account_value,
        )
    
    def _risk_parity_size(
        self,
        price: float,
        volatility: Optional[float] = None,
    ) -> PositionSize:
        """
        風險平價方法
        
        使每個倉位的風險貢獻相等
        """
        if price <= 0:
            return self._empty_result(SizingMethod.RISK_PARITY, price)
        
        # 預設波動率
        vol = volatility or 0.02
        
        if vol <= 0:
            vol = 0.02
        
        # 目標風險貢獻
        target_risk = self._account_value * self._risk_per_trade
        
        # 計算需要的倉位價值
        # risk = position_value * volatility
        # position_value = risk / volatility
        position_value = target_risk / vol
        
        # 限制
        max_value = self._account_value * self._max_position_pct
        position_value = min(position_value, max_value)
        
        quantity = int(position_value / price)
        position_value = quantity * price
        risk_amount = position_value * vol
        
        return PositionSize(
            quantity=quantity,
            risk_amount=risk_amount,
            method=SizingMethod.RISK_PARITY,
            price=price,
            position_value=position_value,
            account_risk_pct=risk_amount / self._account_value,
        )
    
    # ========== 輔助方法 ==========
    
    def _empty_result(self, method: SizingMethod, price: float) -> PositionSize:
        """建立空結果"""
        return PositionSize(
            quantity=0,
            risk_amount=0,
            method=method,
            price=price,
        )
    
    def _apply_limits(self, result: PositionSize, price: float) -> PositionSize:
        """應用數量限制"""
        original_qty = result.quantity
        adjusted = False
        reason = ""
        
        # 最小數量
        if result.quantity < self._min_quantity:
            result.quantity = self._min_quantity
            adjusted = True
            reason = f"低於最小數量 {self._min_quantity}"
        
        # 最大數量
        if result.quantity > self._max_quantity:
            result.quantity = self._max_quantity
            adjusted = True
            reason = f"超過最大數量 {self._max_quantity}"
        
        # 最大倉位價值
        max_value = self._account_value * self._max_position_pct
        if result.quantity * price > max_value:
            result.quantity = int(max_value / price)
            adjusted = True
            reason = f"超過最大倉位價值 ${max_value:.0f}"
        
        # 整批取整
        if self._round_lot_size > 1:
            result.quantity = (result.quantity // self._round_lot_size) * self._round_lot_size
            if result.quantity != original_qty:
                adjusted = True
                reason = f"整批取整 {self._round_lot_size}"
        
        # 更新結果
        if adjusted:
            result.was_adjusted = True
            result.adjustment_reason = reason
            result.original_quantity = original_qty
            result.position_value = result.quantity * price
            
            if self._account_value > 0:
                result.account_risk_pct = result.position_value / self._account_value
        
        return result
    
    # ========== 工具方法 ==========
    
    def calculate_max_quantity(self, price: float) -> int:
        """計算最大可買數量"""
        if price <= 0:
            return 0
        
        max_value = self._account_value * self._max_position_pct
        return int(max_value / price)
    
    def calculate_risk_quantity(
        self,
        price: float,
        stop_loss: float,
    ) -> int:
        """根據風險計算數量"""
        if price <= 0 or stop_loss <= 0:
            return 0
        
        risk_amount = self._account_value * self._risk_per_trade
        risk_per_share = abs(price - stop_loss)
        
        if risk_per_share <= 0:
            return 0
        
        return int(risk_amount / risk_per_share)
    
    def get_config(self) -> Dict[str, Any]:
        """取得配置"""
        return {
            "account_value": self._account_value,
            "risk_per_trade": self._risk_per_trade,
            "max_position_pct": self._max_position_pct,
            "default_method": self._default_method.name,
            "min_quantity": self._min_quantity,
            "max_quantity": self._max_quantity,
            "round_lot_size": self._round_lot_size,
            "fixed_quantity": self._fixed_quantity,
            "kelly_win_rate": self._kelly_win_rate,
            "kelly_win_loss_ratio": self._kelly_win_loss_ratio,
            "kelly_fraction": self._kelly_fraction,
            "target_volatility": self._target_volatility,
        }