"""
FeedHandler 模組 - 市場數據接收器

管理市場數據訂閱，接收 Tick 和 Bar 數據並發布到事件總線
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, List, Set, Callable, Any

from ib_insync import (
    IB,
    Contract,
    Ticker,
    BarData as IBBarData,
    RealTimeBar,
    util,
)

from core.events import (
    TickEvent,
    BarEvent,
    EventType,
)
from core.event_bus import EventBus, get_event_bus
from core.connection import IBConnection


# 設定 logger
logger = logging.getLogger(__name__)


class SubscriptionType(Enum):
    """訂閱類型"""
    
    TICK = "tick"           # Tick 數據（報價）
    REALTIME_BAR = "rtb"    # 即時 5 秒 Bar
    BOTH = "both"           # 同時訂閱 Tick 和 Bar


@dataclass
class SubscriptionInfo:
    """訂閱資訊"""
    
    contract: Contract
    symbol: str
    subscription_type: SubscriptionType
    
    # 訂閱狀態
    ticker: Optional[Ticker] = None
    realtime_bars_handle: Optional[Any] = None
    
    # 統計
    tick_count: int = 0
    bar_count: int = 0
    last_tick_time: Optional[datetime] = None
    last_bar_time: Optional[datetime] = None
    subscribed_time: datetime = field(default_factory=datetime.now)
    
    @property
    def is_active(self) -> bool:
        """訂閱是否活躍"""
        return self.ticker is not None or self.realtime_bars_handle is not None


class FeedHandler:
    """
    市場數據接收器
    
    管理市場數據訂閱，接收 IB 的 Tick 和 Bar 數據，
    轉換為事件發布到事件總線
    
    使用方式:
        feed_handler = FeedHandler(connection, event_bus)
        
        # 訂閱 Tick 數據
        await feed_handler.subscribe(contract, SubscriptionType.TICK)
        
        # 訂閱即時 Bar
        await feed_handler.subscribe(contract, SubscriptionType.REALTIME_BAR)
        
        # 取得歷史數據
        bars = await feed_handler.get_historical_bars(contract, "1 day", "1 M")
    """
    
    def __init__(
        self,
        connection: IBConnection,
        event_bus: Optional[EventBus] = None,
    ):
        """
        初始化數據接收器
        
        Args:
            connection: IB 連接管理器
            event_bus: 事件總線，None 使用全局單例
        """
        self._connection = connection
        self._ib = connection.ib
        self._event_bus = event_bus or get_event_bus()
        
        # 訂閱管理: {symbol: SubscriptionInfo}
        self._subscriptions: Dict[str, SubscriptionInfo] = {}
        
        # 設置 IB 事件監聽
        self._setup_ib_events()
        
        logger.debug("FeedHandler 初始化完成")
    
    # ========== 屬性 ==========
    
    @property
    def subscriptions(self) -> Dict[str, SubscriptionInfo]:
        """取得所有訂閱"""
        return self._subscriptions.copy()
    
    @property
    def subscribed_symbols(self) -> List[str]:
        """取得已訂閱的標的列表"""
        return list(self._subscriptions.keys())
    
    @property
    def subscription_count(self) -> int:
        """取得訂閱數量"""
        return len(self._subscriptions)
    
    # ========== 事件設置 ==========
    
    def _setup_ib_events(self) -> None:
        """設置 IB 事件監聽"""
        # pendingTickersEvent: 當有新的 tick 數據時觸發
        self._ib.pendingTickersEvent += self._on_pending_tickers
        
        logger.debug("IB 事件監聽已設置")
    
    def _cleanup_ib_events(self) -> None:
        """清理 IB 事件監聽"""
        self._ib.pendingTickersEvent -= self._on_pending_tickers
    
    # ========== 訂閱管理 ==========
    
    async def subscribe(
        self,
        contract: Contract,
        subscription_type: SubscriptionType = SubscriptionType.TICK,
        generic_tick_list: str = "",
    ) -> bool:
        """
        訂閱市場數據
        
        Args:
            contract: IB 合約
            subscription_type: 訂閱類型
            generic_tick_list: 額外的 Tick 類型（IB generic tick list）
            
        Returns:
            是否訂閱成功
        """
        symbol = contract.symbol
        
        if symbol in self._subscriptions:
            logger.warning(f"已訂閱 {symbol}，跳過")
            return True
        
        try:
            # 建立訂閱資訊
            sub_info = SubscriptionInfo(
                contract=contract,
                symbol=symbol,
                subscription_type=subscription_type,
            )
            
            # 訂閱 Tick 數據
            if subscription_type in [SubscriptionType.TICK, SubscriptionType.BOTH]:
                ticker = self._ib.reqMktData(
                    contract,
                    genericTickList=generic_tick_list,
                    snapshot=False,
                    regulatorySnapshot=False,
                )
                sub_info.ticker = ticker
                logger.info(f"訂閱 Tick 數據: {symbol}")
            
            # 訂閱即時 5 秒 Bar
            if subscription_type in [SubscriptionType.REALTIME_BAR, SubscriptionType.BOTH]:
                # 根據合約類型選擇 whatToShow
                # 外匯和商品用 MIDPOINT，股票用 TRADES
                what_to_show = "MIDPOINT" if contract.secType in ["CASH", "CMDTY"] else "TRADES"
                
                bars = self._ib.reqRealTimeBars(
                    contract,
                    barSize=5,  # 只支援 5 秒
                    whatToShow=what_to_show,
                    useRTH=False,
                )
                sub_info.realtime_bars_handle = bars
                
                # 註冊 bar 更新回調
                bars.updateEvent += lambda bars, has_new: self._on_realtime_bar(
                    symbol, bars, has_new
                )
                
                logger.info(f"訂閱即時 Bar: {symbol}")
            
            self._subscriptions[symbol] = sub_info
            return True
            
        except Exception as e:
            logger.error(f"訂閱 {symbol} 失敗: {e}")
            return False
    
    async def subscribe_many(
        self,
        contracts: List[Contract],
        subscription_type: SubscriptionType = SubscriptionType.TICK,
    ) -> Dict[str, bool]:
        """
        批量訂閱
        
        Args:
            contracts: 合約列表
            subscription_type: 訂閱類型
            
        Returns:
            {symbol: 是否成功} 的字典
        """
        results = {}
        for contract in contracts:
            success = await self.subscribe(contract, subscription_type)
            results[contract.symbol] = success
            # 避免過快訂閱觸發 IB 限制
            await asyncio.sleep(0.1)
        
        return results
    
    def unsubscribe(self, symbol: str) -> bool:
        """
        取消訂閱
        
        Args:
            symbol: 標的代碼
            
        Returns:
            是否成功取消
        """
        sub_info = self._subscriptions.get(symbol)
        if sub_info is None:
            logger.warning(f"未訂閱 {symbol}，無需取消")
            return False
        
        try:
            # 取消 Tick 訂閱
            if sub_info.ticker is not None:
                self._ib.cancelMktData(sub_info.contract)
                logger.info(f"取消 Tick 訂閱: {symbol}")
            
            # 取消即時 Bar 訂閱
            if sub_info.realtime_bars_handle is not None:
                self._ib.cancelRealTimeBars(sub_info.realtime_bars_handle)
                logger.info(f"取消即時 Bar 訂閱: {symbol}")
            
            del self._subscriptions[symbol]
            return True
            
        except Exception as e:
            logger.error(f"取消訂閱 {symbol} 失敗: {e}")
            return False
    
    def unsubscribe_all(self) -> int:
        """
        取消所有訂閱
        
        Returns:
            取消的訂閱數量
        """
        symbols = list(self._subscriptions.keys())
        count = 0
        
        for symbol in symbols:
            if self.unsubscribe(symbol):
                count += 1
        
        logger.info(f"取消所有訂閱: {count} 個")
        return count
    
    # ========== 數據處理回調 ==========
    
    def _on_pending_tickers(self, tickers: Set[Ticker]) -> None:
        """
        處理 pendingTickersEvent
        
        當有新的 tick 數據時，IB 會觸發此事件
        """
        for ticker in tickers:
            symbol = ticker.contract.symbol
            sub_info = self._subscriptions.get(symbol)
            
            if sub_info is None:
                continue
            
            # 更新統計
            sub_info.tick_count += 1
            sub_info.last_tick_time = datetime.now()
            
            # 建立 TickEvent
            tick_event = TickEvent(
                event_type=EventType.TICK,
                symbol=symbol,
                bid=ticker.bid if ticker.bid > 0 else None,
                ask=ticker.ask if ticker.ask > 0 else None,
                last=ticker.last if ticker.last > 0 else None,
                bid_size=ticker.bidSize if ticker.bidSize > 0 else None,
                ask_size=ticker.askSize if ticker.askSize > 0 else None,
                last_size=ticker.lastSize if ticker.lastSize > 0 else None,
                volume=ticker.volume if ticker.volume > 0 else None,
                open=ticker.open if ticker.open > 0 else None,
                high=ticker.high if ticker.high > 0 else None,
                low=ticker.low if ticker.low > 0 else None,
                close=ticker.close if ticker.close > 0 else None,
            )
            
            # 發布事件
            self._event_bus.publish(tick_event)
    
    def _on_realtime_bar(
        self,
        symbol: str,
        bars: List[RealTimeBar],
        has_new: bool,
    ) -> None:
        """
        處理即時 Bar 更新
        
        Args:
            symbol: 標的代碼
            bars: Bar 列表
            has_new: 是否有新的 Bar
        """
        if not has_new or not bars:
            return
        
        sub_info = self._subscriptions.get(symbol)
        if sub_info is None:
            return
        
        # 取得最新的 Bar
        bar = bars[-1]
        
        # 更新統計
        sub_info.bar_count += 1
        sub_info.last_bar_time = datetime.now()
        
        # 建立 BarEvent
        bar_event = BarEvent(
            event_type=EventType.BAR,
            symbol=symbol,
            open=bar.open_,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=int(bar.volume),
            bar_size="5 secs",
            bar_start=bar.time,
            vwap=bar.wap if hasattr(bar, "wap") else None,
            trade_count=bar.count if hasattr(bar, "count") else None,
        )
        
        # 發布事件
        self._event_bus.publish(bar_event)
    
    # ========== 歷史數據 ==========
    
    async def get_historical_bars(
        self,
        contract: Contract,
        bar_size: str = "1 min",
        duration: str = "1 D",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
        end_datetime: Optional[datetime] = None,
        keep_up_to_date: bool = False,
    ) -> List[BarEvent]:
        """
        獲取歷史 Bar 數據
        
        Args:
            contract: IB 合約
            bar_size: Bar 大小，如 "1 min", "5 mins", "1 hour", "1 day"
            duration: 數據時間範圍，如 "1 D", "1 W", "1 M", "1 Y"
            what_to_show: 數據類型，如 "TRADES", "MIDPOINT", "BID", "ASK"
            use_rth: 是否只取正規交易時段數據
            end_datetime: 結束時間，None 為當前時間
            keep_up_to_date: 是否持續更新
            
        Returns:
            BarEvent 列表
        """
        symbol = contract.symbol
        end_dt = end_datetime or datetime.now()
        
        logger.info(
            f"請求歷史數據: {symbol}, bar_size={bar_size}, "
            f"duration={duration}, what_to_show={what_to_show}"
        )
        
        try:
            bars = await self._ib.reqHistoricalDataAsync(
                contract,
                endDateTime=end_dt,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=1,
                keepUpToDate=keep_up_to_date,
            )
            
            # 轉換為 BarEvent 列表
            bar_events = []
            for bar in bars:
                bar_event = BarEvent(
                    event_type=EventType.BAR,
                    symbol=symbol,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=int(bar.volume),
                    bar_size=bar_size,
                    bar_start=bar.date,
                    vwap=bar.average if hasattr(bar, "average") else None,
                    trade_count=bar.barCount if hasattr(bar, "barCount") else None,
                )
                bar_events.append(bar_event)
            
            logger.info(f"取得歷史數據: {symbol}, {len(bar_events)} bars")
            return bar_events
            
        except Exception as e:
            logger.error(f"取得歷史數據失敗 ({symbol}): {e}")
            return []
    
    def get_historical_bars_sync(
        self,
        contract: Contract,
        bar_size: str = "1 min",
        duration: str = "1 D",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
        end_datetime: Optional[datetime] = None,
    ) -> List[BarEvent]:
        """
        獲取歷史 Bar 數據（同步版本）
        
        參數同 get_historical_bars()
        """
        symbol = contract.symbol
        end_dt = end_datetime or datetime.now()
        
        try:
            bars = self._ib.reqHistoricalData(
                contract,
                endDateTime=end_dt,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=1,
            )
            
            bar_events = []
            for bar in bars:
                bar_event = BarEvent(
                    event_type=EventType.BAR,
                    symbol=symbol,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=int(bar.volume),
                    bar_size=bar_size,
                    bar_start=bar.date,
                    vwap=bar.average if hasattr(bar, "average") else None,
                    trade_count=bar.barCount if hasattr(bar, "barCount") else None,
                )
                bar_events.append(bar_event)
            
            return bar_events
            
        except Exception as e:
            logger.error(f"取得歷史數據失敗 ({symbol}): {e}")
            return []
    
    async def get_historical_ticks(
        self,
        contract: Contract,
        start_datetime: datetime,
        end_datetime: Optional[datetime] = None,
        number_of_ticks: int = 1000,
        what_to_show: str = "TRADES",
        use_rth: bool = True,
    ) -> List[TickEvent]:
        """
        獲取歷史 Tick 數據
        
        Args:
            contract: IB 合約
            start_datetime: 開始時間
            end_datetime: 結束時間，None 為當前時間
            number_of_ticks: 最大 Tick 數量
            what_to_show: "TRADES", "BID_ASK", "MIDPOINT"
            use_rth: 是否只取正規交易時段
            
        Returns:
            TickEvent 列表
        """
        symbol = contract.symbol
        end_dt = end_datetime or datetime.now()
        
        logger.info(f"請求歷史 Tick: {symbol}, {start_datetime} - {end_dt}")
        
        try:
            ticks = await self._ib.reqHistoricalTicksAsync(
                contract,
                startDateTime=start_datetime,
                endDateTime=end_dt,
                numberOfTicks=number_of_ticks,
                whatToShow=what_to_show,
                useRth=use_rth,
            )
            
            tick_events = []
            for tick in ticks:
                tick_event = TickEvent(
                    event_type=EventType.TICK,
                    symbol=symbol,
                    last=tick.price if hasattr(tick, "price") else None,
                    last_size=tick.size if hasattr(tick, "size") else None,
                    timestamp=tick.time if hasattr(tick, "time") else datetime.now(),
                )
                tick_events.append(tick_event)
            
            logger.info(f"取得歷史 Tick: {symbol}, {len(tick_events)} ticks")
            return tick_events
            
        except Exception as e:
            logger.error(f"取得歷史 Tick 失敗 ({symbol}): {e}")
            return []
    
    # ========== 快照 ==========
    
    async def get_snapshot(self, contract: Contract) -> Optional[TickEvent]:
        """
        獲取市場數據快照
        
        Args:
            contract: IB 合約
            
        Returns:
            TickEvent 或 None
        """
        symbol = contract.symbol
        
        try:
            ticker = await self._ib.reqTickersAsync(contract)
            
            if not ticker:
                return None
            
            t = ticker[0]
            return TickEvent(
                event_type=EventType.TICK,
                symbol=symbol,
                bid=t.bid if t.bid > 0 else None,
                ask=t.ask if t.ask > 0 else None,
                last=t.last if t.last > 0 else None,
                bid_size=t.bidSize if t.bidSize > 0 else None,
                ask_size=t.askSize if t.askSize > 0 else None,
                last_size=t.lastSize if t.lastSize > 0 else None,
                volume=t.volume if t.volume > 0 else None,
            )
            
        except Exception as e:
            logger.error(f"取得快照失敗 ({symbol}): {e}")
            return None
    
    # ========== 統計 ==========
    
    def get_subscription_stats(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        取得訂閱統計資訊
        
        Args:
            symbol: 標的代碼
            
        Returns:
            統計資訊字典
        """
        sub_info = self._subscriptions.get(symbol)
        if sub_info is None:
            return None
        
        return {
            "symbol": symbol,
            "subscription_type": sub_info.subscription_type.value,
            "tick_count": sub_info.tick_count,
            "bar_count": sub_info.bar_count,
            "last_tick_time": sub_info.last_tick_time,
            "last_bar_time": sub_info.last_bar_time,
            "subscribed_time": sub_info.subscribed_time,
            "is_active": sub_info.is_active,
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """取得所有訂閱的統計資訊"""
        return {
            symbol: self.get_subscription_stats(symbol)
            for symbol in self._subscriptions
        }
    
    # ========== 清理 ==========
    
    def shutdown(self) -> None:
        """關閉數據接收器"""
        self.unsubscribe_all()
        self._cleanup_ib_events()
        logger.info("FeedHandler 已關閉")