#!/usr/bin/env python3
"""
run_live.py - 實盤交易啟動腳本

用於快速啟動實盤交易環境
"""

# ============================================================
# Python 3.14+ 相容性修復（必須在最前面）
# ============================================================
import asyncio
import sys

# 在導入 ib_insync 之前設定 event loop
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# 嘗試使用 nest_asyncio 來允許嵌套的 event loop
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    print("提示: 安裝 nest_asyncio 可以改善相容性: pip install nest_asyncio")

import logging
import signal
import os
from datetime import datetime
from typing import Optional, List

# 確保專案路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 載入配置（使用穩健的導入方式）
from config.settings import get_settings
from config.trading_modes import TradingMode
settings = get_settings()

from config.symbols import US_TECH_STOCKS, US_ETFS

# 核心組件
from core.events import EventType
from core.event_bus import EventBus
from core.connection import IBConnection, ConnectionConfig
from core.contracts import ContractFactory

# 引擎
from engine.strategy_engine import StrategyEngine
from engine.execution_engine import ExecutionEngine

# 數據
from data.feed_handler import FeedHandler, SubscriptionType
from data.bar_aggregator import BarAggregator
from data.cache import MarketDataCache

# 風控
from risk.manager import RiskManager
from risk.position_sizer import PositionSizer, SizingMethod
from risk.circuit_breaker import CircuitBreaker, BreakerConfig

# 策略
from strategies.base import StrategyConfig
from strategies.examples.sma_cross import SMACrossStrategy
from strategies.examples.test_strategy import TestStrategy

# 工具
from utils.logger import setup_logger, get_logger
from utils.notifier import Notifier, NotificationConfig, NotificationLevel
from utils.time_utils import (
    format_duration,
)
from utils.market_hours import (
    is_market_open,
    time_until_market_open,
    get_eastern_time,
    get_taiwan_time,
)
from utils.performance import PerformanceMonitor


# ============================================================
# 配置
# ============================================================

# 外匯交易標的（免費數據）
LIVE_SYMBOLS = [
    "XAUUSD",    # 黃金（商品）
    "EUR/USD",   # 歐元/美元
    "GBP/USD",   # 英鎊/美元
    "USD/JPY",   # 美元/日圓
    "AUD/USD",   # 澳幣/美元
    "USD/CHF",   # 美元/瑞士法郎
]

# 策略配置
STRATEGY_CONFIG = {
    "test_strategy": {
        "enabled": True,
        "symbols": ["XAUUSD", "USD/JPY"],
        "params": {
            "trigger_bars": 3,
            "auto_close_bars": 2,
            "quantity": 1,
        },
    },
    "sma_cross": {
        "enabled": False,
        "symbols": LIVE_SYMBOLS,
        "params": {
            "fast_period": 10,
            "slow_period": 20,
            "position_size": 20000,  # 外匯通常用較大單位
        },
    },
}


# ============================================================
# 實盤交易運行器
# ============================================================

class LiveTrader:
    """
    實盤交易運行器
    
    簡化的實盤交易啟動流程
    """
    
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        use_paper: bool = True,
    ):
        """
        初始化
        
        Args:
            symbols: 交易標的列表
            use_paper: 是否使用模擬帳戶（安全起見預設 True）
        """
        self._symbols = symbols or LIVE_SYMBOLS
        self._use_paper = use_paper
        
        # 組件
        self._event_bus: Optional[EventBus] = None
        self._connection: Optional[IBConnection] = None
        self._contract_factory: Optional[ContractFactory] = None
        self._feed_handler: Optional[FeedHandler] = None
        self._bar_aggregator: Optional[BarAggregator] = None
        self._cache: Optional[MarketDataCache] = None
        self._strategy_engine: Optional[StrategyEngine] = None
        self._execution_engine: Optional[ExecutionEngine] = None
        self._risk_manager: Optional[RiskManager] = None
        self._position_sizer: Optional[PositionSizer] = None
        self._circuit_breaker: Optional[CircuitBreaker] = None
        self._notifier: Optional[Notifier] = None
        self._performance_monitor: Optional[PerformanceMonitor] = None
        
        # 狀態
        self._running = False
        self._shutdown_event: Optional[asyncio.Event] = None
        self._start_time: Optional[datetime] = None
        
        # Logger
        self._logger = get_logger("LiveTrader")
    
    async def initialize(self) -> bool:
        """
        初始化所有組件
        
        Returns:
            是否成功
        """
        self._logger.info("=" * 60)
        self._logger.info("實盤交易系統初始化中...")
        self._logger.info(f"模式: {'模擬帳戶' if self._use_paper else '實盤帳戶'}")
        self._logger.info(f"標的數量: {len(self._symbols)}")
        self._logger.info("=" * 60)
        
        try:
            # 1. 事件總線
            self._logger.info("初始化 EventBus...")
            self._event_bus = EventBus()
            await self._event_bus.start()
            
            # 2. 通知服務
            self._logger.info("初始化 Notifier...")
            self._notifier = Notifier(NotificationConfig.from_env())
            await self._notifier.initialize()
            
            # 3. 性能監控
            self._logger.info("初始化 PerformanceMonitor...")
            self._performance_monitor = PerformanceMonitor(
                report_interval=300,
                enable_system_monitoring=True,
            )
            self._performance_monitor.start()
            
            # 4. 市場數據快取
            self._logger.info("初始化 MarketDataCache...")
            self._cache = MarketDataCache(
                event_bus=self._event_bus,
                tick_cache_size=10000,
                bar_cache_size=1000,
            )
            
            # 5. IB 連接
            self._logger.info("初始化 IBConnection...")
            # 取得連接埠（支援不同的屬性名）
            if self._use_paper:
                port = getattr(settings.ib, 'paper_port', None) or getattr(settings.ib, 'port', 7497)
            else:
                port = getattr(settings.ib, 'live_port', None) or getattr(settings.ib, 'port', 7496)
            
            ib_config = ConnectionConfig(
                host=settings.ib.host,
                port=port,
                client_id=settings.ib.client_id,
                timeout=settings.ib.timeout,
            )
            self._connection = IBConnection(ib_config)
            self._contract_factory = ContractFactory()
            
            # 6. 數據處理
            self._logger.info("初始化 FeedHandler...")
            self._feed_handler = FeedHandler(
                connection=self._connection,
                event_bus=self._event_bus,
            )
            
            self._logger.info("初始化 BarAggregator...")
            self._bar_aggregator = BarAggregator(
                event_bus=self._event_bus,
            )
            
            # 7. 風控組件
            self._logger.info("初始化風控組件...")
            
            # 預設帳戶價值（可從環境變數或設定檔讀取）
            account_value = float(os.getenv("ACCOUNT_VALUE", "100000"))
            
            self._risk_manager = RiskManager(
                config=settings.risk,
                event_bus=self._event_bus,
                account_value=account_value,
            )
            
            self._position_sizer = PositionSizer(
                account_value=account_value,
                risk_per_trade=float(os.getenv("RISK_PER_TRADE", "0.01")),
                max_position_pct=float(os.getenv("MAX_POSITION_PCT", "0.1")),
            )
            
            self._circuit_breaker = CircuitBreaker(
                config=BreakerConfig(
                    max_consecutive_losses=settings.risk.circuit_breaker_threshold,
                    cooldown_seconds=settings.risk.circuit_breaker_cooldown,
                ),
                event_bus=self._event_bus,
            )
            
            # 8. 策略引擎
            self._logger.info("初始化 StrategyEngine...")
            self._strategy_engine = StrategyEngine(
                event_bus=self._event_bus,
            )
            
            # 9. 執行引擎
            self._logger.info("初始化 ExecutionEngine...")
            self._execution_engine = ExecutionEngine(
                connection=self._connection,
                event_bus=self._event_bus,
                risk_manager=self._risk_manager,
            )
            
            self._logger.info("所有組件初始化完成")
            return True
            
        except Exception as e:
            self._logger.error(f"初始化失敗: {e}")
            await self._send_alert(f"系統初始化失敗: {e}", NotificationLevel.CRITICAL)
            return False
    
    async def connect(self) -> bool:
        """
        連接到 IB
        
        Returns:
            是否成功
        """
        self._logger.info("連接到 Interactive Brokers...")
        
        # 使用 ib_insync 的 util.run 來處理連接
        from ib_insync import util
        
        try:
            # 使用同步連接（ib_insync 內部會處理 event loop）
            ib = self._connection.ib
            ib.connect(
                host=self._connection.config.host,
                port=self._connection.config.port,
                clientId=self._connection.config.client_id,
                readonly=self._connection.config.readonly,
                timeout=self._connection.config.timeout,
            )
            success = ib.isConnected()
        except Exception as e:
            self._logger.error(f"連接錯誤: {e}")
            success = False
        
        if not success:
            self._logger.error("IB 連接失敗")
            self._logger.error("請確認：")
            self._logger.error("  1. TWS 或 IB Gateway 已啟動")
            self._logger.error("  2. API 連接已啟用（端口 7497）")
            self._logger.error("  3. 允許來自 localhost 的連接")
            await self._send_alert("IB 連接失敗", NotificationLevel.ERROR)
            return False
        
        self._logger.info("IB 連接成功")
        
        # 取得帳戶資訊
        await self._fetch_account_info()
        
        # 同步現有持倉
        await self._sync_ib_positions()
        
        # 訂閱標的
        await self._subscribe_symbols()
        
        return True
    
    async def _fetch_account_info(self) -> None:
        """取得帳戶資訊"""
        try:
            ib = self._connection.ib
            
            # 取得帳戶摘要
            account_values = ib.accountSummary()
            
            # 更新帳戶價值
            for av in account_values:
                if av.tag == "NetLiquidation":
                    account_value = float(av.value)
                    self._risk_manager.update_account_value(account_value)
                    self._position_sizer.set_account_value(account_value)
                    self._logger.info(f"帳戶淨值: ${account_value:,.2f}")
                    break
                    
        except Exception as e:
            self._logger.warning(f"取得帳戶資訊失敗: {e}")
    
    async def _sync_ib_positions(self) -> None:
        """從 IB 同步現有持倉"""
        try:
            ib = self._connection.ib
            
            # 取得 IB 持倉
            ib_positions = ib.positions()
            
            if not ib_positions:
                self._logger.info("目前無持倉")
                return
            
            # 轉換為 RiskManager 需要的格式
            positions_data = {}
            for pos in ib_positions:
                symbol = pos.contract.symbol
                positions_data[symbol] = {
                    "quantity": int(pos.position),
                    "avg_cost": float(pos.avgCost),
                    "market_value": float(pos.position * pos.avgCost),
                }
                self._logger.info(
                    f"同步持倉: {symbol} = {int(pos.position)} @ ${pos.avgCost:.2f}"
                )
            
            # 同步到 RiskManager
            self._risk_manager.sync_positions(positions_data)
            self._logger.info(f"已同步 {len(positions_data)} 個持倉")
            
        except Exception as e:
            self._logger.warning(f"同步持倉失敗: {e}")
    
    async def _subscribe_symbols(self) -> None:
        """訂閱交易標的"""
        self._logger.info(f"訂閱 {len(self._symbols)} 個標的...")
        
        subscribed = 0
        
        for symbol in self._symbols:
            try:
                # 判斷標的類型
                if symbol in ["XAUUSD", "XAGUSD"]:
                    # 商品
                    contract = self._contract_factory.commodity(symbol)
                elif "/" in symbol:
                    # 外匯：EUR/USD -> base=EUR, quote=USD
                    parts = symbol.split("/")
                    base = parts[0]
                    quote = parts[1] if len(parts) > 1 else "USD"
                    contract = self._contract_factory.forex(base, quote)
                else:
                    # 股票
                    contract = self._contract_factory.stock(symbol)
                
                # 訂閱即時數據
                success = await self._feed_handler.subscribe(
                    contract=contract,
                    subscription_type=SubscriptionType.REALTIME_BAR,
                )
                
                if success:
                    subscribed += 1
                    self._logger.debug(f"已訂閱: {symbol}")
                else:
                    self._logger.warning(f"訂閱 {symbol} 返回失敗")
                
            except Exception as e:
                self._logger.error(f"訂閱 {symbol} 失敗: {e}")
        
        self._logger.info(f"成功訂閱 {subscribed}/{len(self._symbols)} 個標的")
    
    async def _load_strategies(self) -> None:
        """載入並啟動策略"""
        self._logger.info("載入策略...")
        
        # 測試策略
        test_config = STRATEGY_CONFIG.get("test_strategy", {})
        
        if test_config.get("enabled", False):
            params = test_config.get("params", {})
            
            strategy = TestStrategy(
                strategy_id="test_strategy_live",
                symbols=test_config.get("symbols", ["XAUUSD", "USD/JPY"]),
                trigger_bars=params.get("trigger_bars", 3),
                auto_close_bars=params.get("auto_close_bars", 2),
                quantity=params.get("quantity", 1),
            )
            
            self._strategy_engine.add_strategy(strategy)
            strategy.initialize()
            strategy.start()
            
            self._logger.info(f"已載入策略: test_strategy_live")
        
        # SMA 交叉策略
        sma_config = STRATEGY_CONFIG.get("sma_cross", {})
        
        if sma_config.get("enabled", False):
            params = sma_config.get("params", {})
            
            strategy = SMACrossStrategy(
                strategy_id="sma_cross_live",
                symbols=sma_config.get("symbols", self._symbols),
                fast_period=params.get("fast_period", 10),
                slow_period=params.get("slow_period", 20),
                quantity=params.get("position_size", 100),
            )
            
            self._strategy_engine.add_strategy(strategy)
            strategy.initialize()
            strategy.start()
            
            self._logger.info(f"已載入策略: sma_cross_live")
    
    async def run(self) -> None:
        """
        主運行循環
        """
        self._running = True
        self._start_time = datetime.now()
        self._shutdown_event = asyncio.Event()
        
        # 顯示啟動資訊
        self._print_startup_info()
        
        # 發送啟動通知
        await self._send_alert(
            f"實盤交易系統已啟動\n"
            f"模式: {'模擬' if self._use_paper else '實盤'}\n"
            f"標的數: {len(self._symbols)}\n"
            f"美東時間: {get_eastern_time().strftime('%H:%M:%S')}",
            NotificationLevel.INFO,
        )
        
        try:
            # 啟動組件
            self._risk_manager.start()
            self._circuit_breaker.start()
            self._strategy_engine.start()
            self._execution_engine.start()
            self._bar_aggregator.start()
            
            # 載入策略
            await self._load_strategies()
            
            # 主循環
            loop_count = 0
            
            while not self._shutdown_event.is_set():
                await self._main_loop_tick(loop_count)
                loop_count += 1
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            self._logger.info("收到取消信號")
        except Exception as e:
            self._logger.error(f"運行錯誤: {e}")
            await self._send_alert(f"系統運行錯誤: {e}", NotificationLevel.CRITICAL)
    
    async def _main_loop_tick(self, loop_count: int) -> None:
        """主循環每秒執行"""
        # 每 60 秒輸出狀態
        if loop_count % 60 == 0 and loop_count > 0:
            await self._print_status()
        
        # 每 5 分鐘檢查連接
        if loop_count % 300 == 0:
            if not self._connection.is_connected:
                self._logger.warning("IB 連接斷開，嘗試重連...")
                await self._send_alert("IB 連接斷開，正在重連...", NotificationLevel.WARNING)
                self._connection.reconnect()
        
        # 記錄性能
        self._performance_monitor.record_event("main_loop")
        
        # 檢查熔斷狀態
        if self._circuit_breaker.is_triggered:
            if loop_count % 60 == 0:
                remaining = self._circuit_breaker.get_remaining_cooldown()
                self._logger.warning(f"熔斷中，剩餘冷卻時間: {remaining:.0f}s")
    
    async def _print_status(self) -> None:
        """輸出狀態"""
        et = get_eastern_time()
        tw = get_taiwan_time()
        market_status = "開盤" if is_market_open() else "休市"
        
        uptime = datetime.now() - self._start_time
        
        self._logger.info(
            f"狀態: 運行 {format_duration(uptime)} | "
            f"市場: {market_status} | "
            f"美東: {et.strftime('%H:%M')} | "
            f"台灣: {tw.strftime('%H:%M')}"
        )
    
    def _print_startup_info(self) -> None:
        """輸出啟動資訊"""
        et = get_eastern_time()
        tw = get_taiwan_time()
        
        self._logger.info("=" * 60)
        self._logger.info("實盤交易系統已啟動")
        self._logger.info("=" * 60)
        self._logger.info(f"交易模式: {'模擬帳戶' if self._use_paper else '實盤帳戶'}")
        self._logger.info(f"交易標的: {', '.join(self._symbols[:5])}{'...' if len(self._symbols) > 5 else ''}")
        self._logger.info(f"美東時間: {et.strftime('%Y-%m-%d %H:%M:%S')}")
        self._logger.info(f"台灣時間: {tw.strftime('%Y-%m-%d %H:%M:%S')}")
        self._logger.info(f"市場狀態: {'開盤中' if is_market_open() else '休市'}")
        
        if not is_market_open():
            remaining = time_until_market_open()
            self._logger.info(f"距離開盤: {format_duration(remaining)}")
        
        self._logger.info("=" * 60)
        self._logger.info("按 Ctrl+C 停止系統")
        self._logger.info("=" * 60)
    
    async def shutdown(self) -> None:
        """關閉系統"""
        if not self._running:
            return
        
        self._running = False
        
        self._logger.info("=" * 60)
        self._logger.info("系統關閉中...")
        self._logger.info("=" * 60)
        
        try:
            # 停止策略引擎
            if self._strategy_engine:
                self._logger.info("停止策略引擎...")
                self._strategy_engine.stop()
            
            # 取消未成交訂單
            if self._execution_engine:
                self._logger.info("取消未成交訂單...")
                self._execution_engine.cancel_all_orders()  # 同步方法，不需要 await
                self._execution_engine.stop()
            
            # 停止風控組件
            if self._circuit_breaker:
                self._circuit_breaker.stop()
            if self._risk_manager:
                self._risk_manager.stop()
            
            # 停止數據處理
            if self._bar_aggregator:
                self._bar_aggregator.stop()
            if self._feed_handler:
                self._feed_handler.unsubscribe_all()  # 同步方法，不需要 await
            
            # 斷開 IB 連接
            if self._connection and self._connection.is_connected:
                self._logger.info("斷開 IB 連接...")
                self._connection.disconnect()
            
            # 停止性能監控
            if self._performance_monitor:
                self._performance_monitor.stop()
            
            # 停止事件總線
            if self._event_bus:
                await self._event_bus.stop()
            
            # 發送關閉通知
            if self._notifier and self._start_time:
                uptime = datetime.now() - self._start_time
                await self._notifier.alert(
                    f"實盤交易系統已關閉\n運行時間: {format_duration(uptime)}",
                    level=NotificationLevel.INFO,
                )
                await self._notifier.shutdown()
            
            self._logger.info("系統已安全關閉")
            
        except Exception as e:
            self._logger.error(f"關閉過程中發生錯誤: {e}")
    
    def request_shutdown(self) -> None:
        """請求關閉"""
        if self._shutdown_event:
            self._shutdown_event.set()
    
    async def _send_alert(self, message: str, level: NotificationLevel) -> None:
        """發送告警"""
        if self._notifier:
            await self._notifier.alert(message, level=level)


# ============================================================
# 主程式
# ============================================================

# 全局實例
_trader: Optional[LiveTrader] = None


def signal_handler(signum: int, frame) -> None:
    """信號處理器"""
    print(f"\n收到信號: {signal.Signals(signum).name}")
    if _trader:
        _trader.request_shutdown()


async def main() -> int:
    """主程式"""
    global _trader
    
    # 設定日誌
    setup_logger(
        log_dir="logs",
        log_level="INFO",
        console_output=True,
        file_output=True,
    )
    
    logger = get_logger("main")
    
    # 解析命令列參數
    import argparse
    parser = argparse.ArgumentParser(description="實盤交易啟動腳本")
    parser.add_argument(
        "--live",
        action="store_true",
        help="使用實盤帳戶（預設使用模擬帳戶）",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="交易標的，用逗號分隔",
    )
    args = parser.parse_args()
    
    # 解析標的
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    
    # 安全確認
    if args.live:
        logger.warning("=" * 60)
        logger.warning("警告：即將使用【實盤帳戶】進行交易！")
        logger.warning("=" * 60)
        confirm = input("確認使用實盤帳戶？(yes/no): ")
        if confirm.lower() != "yes":
            logger.info("已取消")
            return 0
    
    # 建立交易器
    _trader = LiveTrader(
        symbols=symbols,
        use_paper=not args.live,
    )
    
    # 設定信號處理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 初始化
        if not await _trader.initialize():
            return 1
        
        # 連接
        if not await _trader.connect():
            return 2
        
        # 運行
        await _trader.run()
        
        return 0
        
    except Exception as e:
        logger.error(f"錯誤: {e}")
        return 3
    finally:
        if _trader:
            await _trader.shutdown()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)