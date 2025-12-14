"""
CircuitBreaker 模組 - 熔斷機制

提供交易熔斷保護功能
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Optional, Dict, List, Callable, Any
import threading

from core.events import (
    EventType,
    FillEvent,
    RiskEvent,
    RiskEventType,
    OrderAction,
)
from core.event_bus import EventBus, get_event_bus


# 設定 logger
logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """熔斷狀態"""
    
    CLOSED = auto()      # 正常（允許交易）
    OPEN = auto()        # 熔斷（禁止交易）
    HALF_OPEN = auto()   # 半開（試探性允許）


@dataclass
class BreakerConfig:
    """熔斷器配置"""
    
    # 虧損觸發條件
    max_consecutive_losses: int = 5       # 最大連續虧損次數
    max_loss_amount: float = 1000.0       # 最大虧損金額
    max_loss_percent: float = 0.02        # 最大虧損百分比（相對帳戶）
    
    # 時間窗口觸發
    max_losses_in_window: int = 10        # 時間窗口內最大虧損次數
    window_minutes: int = 60              # 時間窗口（分鐘）
    
    # 冷卻設定
    cooldown_seconds: int = 300           # 冷卻時間（秒）
    auto_reset: bool = True               # 是否自動重置
    
    # 半開狀態設定
    half_open_trades: int = 3             # 半開狀態允許的交易次數
    half_open_after_seconds: int = 60     # 多久後進入半開狀態
    
    # 累進冷卻
    progressive_cooldown: bool = True     # 是否啟用累進冷卻
    cooldown_multiplier: float = 1.5      # 冷卻時間乘數
    max_cooldown_seconds: int = 3600      # 最大冷卻時間


@dataclass
class TradeRecord:
    """交易記錄"""
    
    timestamp: datetime
    symbol: str
    pnl: float
    is_win: bool
    
    @classmethod
    def from_fill(cls, fill: FillEvent, pnl: float) -> "TradeRecord":
        """從 FillEvent 建立"""
        return cls(
            timestamp=fill.timestamp,
            symbol=fill.symbol,
            pnl=pnl,
            is_win=pnl >= 0,
        )


@dataclass
class BreakerStats:
    """熔斷器統計"""
    
    # 交易統計
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0
    
    # 盈虧統計
    total_pnl: float = 0.0
    window_pnl: float = 0.0
    window_losses: int = 0
    
    # 熔斷統計
    trigger_count: int = 0
    last_trigger_time: Optional[datetime] = None
    last_reset_time: Optional[datetime] = None
    
    # 當前狀態
    current_cooldown_seconds: int = 0


class CircuitBreaker:
    """
    熔斷器
    
    監控交易表現，在達到風險閾值時觸發熔斷保護
    
    使用方式:
        breaker = CircuitBreaker(config, event_bus)
        breaker.start()
        
        # 檢查是否可交易
        if breaker.can_trade():
            # 執行交易
            pass
        
        # 手動記錄交易（如果不訂閱 FillEvent）
        breaker.record_trade(symbol="AAPL", pnl=-50.0)
    """
    
    def __init__(
        self,
        config: Optional[BreakerConfig] = None,
        event_bus: Optional[EventBus] = None,
        account_value: float = 100000.0,
    ):
        """
        初始化熔斷器
        
        Args:
            config: 熔斷器配置
            event_bus: 事件總線
            account_value: 帳戶價值（用於計算百分比）
        """
        self._config = config or BreakerConfig()
        self._event_bus = event_bus or get_event_bus()
        self._account_value = account_value
        
        # 狀態
        self._state = CircuitState.CLOSED
        self._stats = BreakerStats()
        
        # 交易歷史（用於時間窗口計算）
        self._trade_history: List[TradeRecord] = []
        
        # 冷卻計時
        self._trigger_time: Optional[datetime] = None
        self._cooldown_end_time: Optional[datetime] = None
        self._cooldown_task: Optional[asyncio.Task] = None
        self._cooldown_timer: Optional[threading.Timer] = None
        
        # 半開狀態計數
        self._half_open_trade_count: int = 0
        
        # 運行狀態
        self._running = False
        
        # 線程安全
        self._lock = threading.RLock()
        
        # 回調
        self._on_trigger_callbacks: List[Callable[[], None]] = []
        self._on_reset_callbacks: List[Callable[[], None]] = []
        
        logger.info(
            f"CircuitBreaker 初始化: max_consecutive_losses={self._config.max_consecutive_losses}, "
            f"cooldown={self._config.cooldown_seconds}s"
        )
    
    # ========== 屬性 ==========
    
    @property
    def state(self) -> CircuitState:
        """當前狀態"""
        return self._state
    
    @property
    def is_triggered(self) -> bool:
        """是否已觸發熔斷"""
        return self._state == CircuitState.OPEN
    
    @property
    def is_closed(self) -> bool:
        """是否正常（允許交易）"""
        return self._state == CircuitState.CLOSED
    
    @property
    def is_half_open(self) -> bool:
        """是否半開狀態"""
        return self._state == CircuitState.HALF_OPEN
    
    @property
    def stats(self) -> BreakerStats:
        """統計資訊"""
        return self._stats
    
    # ========== 控制 ==========
    
    def start(self) -> None:
        """啟動熔斷器"""
        if self._running:
            logger.warning("CircuitBreaker 已在運行中")
            return
        
        # 訂閱 FillEvent
        self._event_bus.subscribe(EventType.FILL, self._on_fill, priority=0)
        
        self._running = True
        logger.info("CircuitBreaker 已啟動")
    
    def stop(self) -> None:
        """停止熔斷器"""
        if not self._running:
            return
        
        self._event_bus.unsubscribe(EventType.FILL, self._on_fill)
        
        # 取消冷卻計時器
        self._cancel_cooldown_timer()
        
        self._running = False
        logger.info("CircuitBreaker 已停止")
    
    # ========== 交易檢查 ==========
    
    def can_trade(self) -> bool:
        """
        檢查是否可以交易
        
        Returns:
            是否允許交易
        """
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            
            if self._state == CircuitState.HALF_OPEN:
                # 半開狀態：允許有限的交易
                return self._half_open_trade_count < self._config.half_open_trades
            
            # OPEN 狀態：不允許交易
            return False
    
    def check_and_allow(self) -> bool:
        """
        檢查並計數（用於半開狀態）
        
        Returns:
            是否允許交易
        """
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_trade_count < self._config.half_open_trades:
                    self._half_open_trade_count += 1
                    return True
                return False
            
            return self.can_trade()
    
    # ========== 交易記錄 ==========
    
    def record_trade(
        self,
        symbol: str,
        pnl: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        記錄交易結果
        
        Args:
            symbol: 標的代碼
            pnl: 盈虧
            timestamp: 時間戳
        """
        with self._lock:
            now = timestamp or datetime.now()
            is_win = pnl >= 0
            
            # 建立記錄
            record = TradeRecord(
                timestamp=now,
                symbol=symbol,
                pnl=pnl,
                is_win=is_win,
            )
            self._trade_history.append(record)
            
            # 更新統計
            self._stats.total_trades += 1
            self._stats.total_pnl += pnl
            
            if is_win:
                self._stats.winning_trades += 1
                self._stats.consecutive_losses = 0
                
                # 半開狀態：如果盈利，可能回到 CLOSED
                if self._state == CircuitState.HALF_OPEN:
                    self._on_half_open_win()
            else:
                self._stats.losing_trades += 1
                self._stats.consecutive_losses += 1
                
                # 更新最大連續虧損
                if self._stats.consecutive_losses > self._stats.max_consecutive_losses:
                    self._stats.max_consecutive_losses = self._stats.consecutive_losses
                
                # 半開狀態：如果虧損，回到 OPEN
                if self._state == CircuitState.HALF_OPEN:
                    self._on_half_open_loss()
            
            # 計算時間窗口內的統計
            self._update_window_stats()
            
            # 檢查是否需要觸發熔斷
            if self._state == CircuitState.CLOSED:
                self._check_trigger_conditions(pnl)
            
            logger.debug(
                f"記錄交易: {symbol} pnl={pnl:.2f}, "
                f"consecutive_losses={self._stats.consecutive_losses}"
            )
    
    def _on_fill(self, event: FillEvent) -> None:
        """處理 FillEvent"""
        # 注意：這裡需要外部提供 PnL 資訊
        # FillEvent 本身可能沒有完整的 PnL 資訊
        # 這裡假設可以從某處計算或獲取
        pass
    
    def record_fill(self, fill: FillEvent, realized_pnl: float) -> None:
        """
        記錄成交和盈虧
        
        Args:
            fill: FillEvent
            realized_pnl: 已實現盈虧
        """
        self.record_trade(
            symbol=fill.symbol,
            pnl=realized_pnl,
            timestamp=fill.timestamp,
        )
    
    # ========== 觸發條件檢查 ==========
    
    def _check_trigger_conditions(self, last_pnl: float) -> None:
        """檢查是否滿足觸發條件"""
        should_trigger = False
        trigger_reason = ""
        
        # 條件 1: 連續虧損次數
        if self._stats.consecutive_losses >= self._config.max_consecutive_losses:
            should_trigger = True
            trigger_reason = f"連續虧損 {self._stats.consecutive_losses} 次"
        
        # 條件 2: 單筆虧損金額
        if last_pnl < 0 and abs(last_pnl) >= self._config.max_loss_amount:
            should_trigger = True
            trigger_reason = f"單筆虧損 ${abs(last_pnl):.2f} 超過閾值"
        
        # 條件 3: 虧損百分比
        if self._account_value > 0:
            loss_pct = abs(last_pnl) / self._account_value
            if last_pnl < 0 and loss_pct >= self._config.max_loss_percent:
                should_trigger = True
                trigger_reason = f"單筆虧損 {loss_pct:.2%} 超過閾值"
        
        # 條件 4: 時間窗口內虧損次數
        if self._stats.window_losses >= self._config.max_losses_in_window:
            should_trigger = True
            trigger_reason = f"時間窗口內虧損 {self._stats.window_losses} 次"
        
        if should_trigger:
            self._trigger(trigger_reason)
    
    def _update_window_stats(self) -> None:
        """更新時間窗口統計"""
        now = datetime.now()
        window_start = now - timedelta(minutes=self._config.window_minutes)
        
        # 過濾時間窗口內的交易
        window_trades = [
            t for t in self._trade_history
            if t.timestamp >= window_start
        ]
        
        # 計算窗口統計
        self._stats.window_pnl = sum(t.pnl for t in window_trades)
        self._stats.window_losses = sum(1 for t in window_trades if not t.is_win)
        
        # 清理過期記錄（保留最近 1000 筆）
        if len(self._trade_history) > 1000:
            self._trade_history = self._trade_history[-1000:]
    
    # ========== 熔斷觸發 ==========
    
    def _trigger(self, reason: str) -> None:
        """觸發熔斷"""
        with self._lock:
            if self._state == CircuitState.OPEN:
                return  # 已經是 OPEN 狀態
            
            self._state = CircuitState.OPEN
            self._trigger_time = datetime.now()
            self._stats.trigger_count += 1
            self._stats.last_trigger_time = self._trigger_time
            
            # 計算冷卻時間（累進）
            cooldown = self._config.cooldown_seconds
            if self._config.progressive_cooldown:
                cooldown = int(cooldown * (self._config.cooldown_multiplier ** (self._stats.trigger_count - 1)))
                cooldown = min(cooldown, self._config.max_cooldown_seconds)
            
            self._stats.current_cooldown_seconds = cooldown
            self._cooldown_end_time = self._trigger_time + timedelta(seconds=cooldown)
            
            logger.warning(f"熔斷觸發: {reason}, 冷卻 {cooldown} 秒")
            
            # 發布風控事件
            self._emit_risk_event(
                RiskEventType.CIRCUIT_BREAKER_TRIGGERED,
                f"熔斷觸發: {reason}",
                action_taken=f"冷卻 {cooldown} 秒",
            )
            
            # 執行回調
            for callback in self._on_trigger_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"觸發回調錯誤: {e}")
            
            # 啟動冷卻計時器
            if self._config.auto_reset:
                self._start_cooldown_timer(cooldown)
    
    def _start_cooldown_timer(self, seconds: int) -> None:
        """啟動冷卻計時器"""
        self._cancel_cooldown_timer()
        
        # 先進入半開狀態的計時
        half_open_delay = self._config.half_open_after_seconds
        if half_open_delay < seconds:
            self._cooldown_timer = threading.Timer(
                half_open_delay,
                self._enter_half_open,
            )
            self._cooldown_timer.daemon = True
            self._cooldown_timer.start()
        else:
            self._cooldown_timer = threading.Timer(
                seconds,
                self.reset,
            )
            self._cooldown_timer.daemon = True
            self._cooldown_timer.start()
    
    def _cancel_cooldown_timer(self) -> None:
        """取消冷卻計時器"""
        if self._cooldown_timer:
            self._cooldown_timer.cancel()
            self._cooldown_timer = None
    
    def _enter_half_open(self) -> None:
        """進入半開狀態"""
        with self._lock:
            if self._state != CircuitState.OPEN:
                return
            
            self._state = CircuitState.HALF_OPEN
            self._half_open_trade_count = 0
            
            logger.info("熔斷器進入半開狀態")
            
            # 設定完全重置的計時器
            remaining = self.get_remaining_cooldown()
            if remaining > 0:
                self._cooldown_timer = threading.Timer(
                    remaining,
                    self.reset,
                )
                self._cooldown_timer.daemon = True
                self._cooldown_timer.start()
    
    def _on_half_open_win(self) -> None:
        """半開狀態盈利"""
        # 可以考慮提前回到 CLOSED 狀態
        if self._half_open_trade_count >= self._config.half_open_trades:
            self.reset()
    
    def _on_half_open_loss(self) -> None:
        """半開狀態虧損"""
        # 回到 OPEN 狀態
        self._state = CircuitState.OPEN
        self._half_open_trade_count = 0
        
        logger.warning("半開狀態虧損，回到熔斷狀態")
    
    # ========== 重置 ==========
    
    def reset(self) -> None:
        """重置熔斷器（冷卻結束後調用）"""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return
            
            self._state = CircuitState.CLOSED
            self._stats.consecutive_losses = 0
            self._stats.last_reset_time = datetime.now()
            self._half_open_trade_count = 0
            
            self._cancel_cooldown_timer()
            
            logger.info("熔斷器已重置")
            
            # 發布風控事件
            self._emit_risk_event(
                RiskEventType.CIRCUIT_BREAKER_RESET,
                "熔斷器已重置",
            )
            
            # 執行回調
            for callback in self._on_reset_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"重置回調錯誤: {e}")
    
    def force_reset(self) -> None:
        """強制重置（手動調用）"""
        with self._lock:
            self._stats.trigger_count = 0  # 重置觸發計數
            self._stats.current_cooldown_seconds = self._config.cooldown_seconds
            self.reset()
            
            logger.info("熔斷器強制重置")
    
    # ========== 查詢方法 ==========
    
    def get_remaining_cooldown(self) -> float:
        """
        取得剩餘冷卻時間（秒）
        
        Returns:
            剩餘秒數，0 表示無冷卻
        """
        if self._cooldown_end_time is None:
            return 0.0
        
        remaining = (self._cooldown_end_time - datetime.now()).total_seconds()
        return max(0.0, remaining)
    
    def get_status(self) -> Dict[str, Any]:
        """取得熔斷器狀態"""
        return {
            "state": self._state.name,
            "is_triggered": self.is_triggered,
            "can_trade": self.can_trade(),
            "consecutive_losses": self._stats.consecutive_losses,
            "max_consecutive_losses": self._stats.max_consecutive_losses,
            "trigger_count": self._stats.trigger_count,
            "last_trigger_time": (
                self._stats.last_trigger_time.isoformat()
                if self._stats.last_trigger_time else None
            ),
            "remaining_cooldown": self.get_remaining_cooldown(),
            "current_cooldown_seconds": self._stats.current_cooldown_seconds,
            "window_losses": self._stats.window_losses,
            "total_trades": self._stats.total_trades,
            "total_pnl": self._stats.total_pnl,
        }
    
    # ========== 事件發布 ==========
    
    def _emit_risk_event(
        self,
        sub_type: RiskEventType,
        message: str,
        action_taken: Optional[str] = None,
    ) -> None:
        """發布風控事件"""
        risk_event = RiskEvent(
            event_type=EventType.RISK,
            sub_type=sub_type,
            message=message,
            action_taken=action_taken,
        )
        
        self._event_bus.publish(risk_event)
    
    # ========== 回調註冊 ==========
    
    def on_trigger(self, callback: Callable[[], None]) -> Callable:
        """註冊觸發回調"""
        self._on_trigger_callbacks.append(callback)
        return callback
    
    def on_reset(self, callback: Callable[[], None]) -> Callable:
        """註冊重置回調"""
        self._on_reset_callbacks.append(callback)
        return callback
    
    # ========== 工具方法 ==========
    
    def set_account_value(self, value: float) -> None:
        """設定帳戶價值"""
        self._account_value = value
    
    def update_config(self, **kwargs) -> None:
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)