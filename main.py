#!/usr/bin/env python3
"""
IB Trading System - 主程式入口

Interactive Brokers 自動交易系統
"""

import argparse
import asyncio
import logging
import signal
import sys
import os
from datetime import datetime
from enum import Enum, auto
from typing import Optional, List

# 確保專案路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 載入配置
from config.settings import settings, TradingMode
from config.symbols import SymbolConfig, DEFAULT_WATCHLIST

# 核心組件
from core.events import EventType
from core.event_bus import EventBus
from core.connection import IBConnection, IBConfig
from core.contracts import ContractFactory

# 引擎
from engine.strategy_engine import StrategyEngine
from engine.execution_engine import ExecutionEngine

# 數據
from data.feed_handler import FeedHandler
from data.bar_aggregator import BarAggregator
from data.database import Database
from data.cache import MarketDataCache

# 風控
from risk import RiskManager
from risk.position_sizer import PositionSizer, SizingMethod
from risk.circuit_breaker import CircuitBreaker, BreakerConfig

# 工具
from utils.logger import setup_logger, get_logger
from utils.notifier import Notifier, NotificationConfig, NotificationLevel
from utils.time_utils import (
    is_market_open,
    time_until_market_open,
    format_duration,
    get_eastern_time,
)
from utils.performance import PerformanceMonitor

# 策略
from strategies.registry import StrategyRegistry
from strategies.examples.sma_cross import SMACrossStrategy
from strategies.examples.tick_scalper import TickScalperStrategy


# 設定 logger
logger = get_logger(__name__)


class SystemState(Enum):
    """系統狀態"""
    
    STOPPED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    ERROR = auto()


class TradingSystem:
    """
    交易系統主類
    
    整合所有組件，提供統一的啟動和管理接口
    """
    
    def __init__(
        self,
        mode: TradingMode = TradingMode.PAPER,
        symbols: Optional[List[str]] = None,
        config_file: Optional[str] = None,
    ):
        """
        初始化交易系統
        
        Args:
            mode: 交易模式 (LIVE/PAPER/BACKTEST)
            symbols: 交易標的列表
            config_file: 配置檔路徑
        """
        self._mode = mode
        self._symbols = symbols or DEFAULT_WATCHLIST
        self._config_file = config_file
        
        # 系統狀態
        self._state = SystemState.STOPPED
        self._start_time: Optional[datetime] = None
        
        # 核心組件
        self._event_bus: Optional[EventBus] = None
        self._connection: Optional[IBConnection] = None
        self._contract_factory: Optional[ContractFactory] = None
        
        # 引擎
        self._strategy_engine: Optional[StrategyEngine] = None
        self._execution_engine: Optional[ExecutionEngine] = None
        
        # 數據
        self._feed_handler: Optional[FeedHandler] = None
        self._bar_aggregator: Optional[BarAggregator] = None
        self._database: Optional[Database] = None
        self._cache: Optional[MarketDataCache] = None
        
        # 風控
        self._risk_manager: Optional[RiskManager] = None
        self._position_sizer: Optional[PositionSizer] = None
        self._circuit_breaker: Optional[CircuitBreaker] = None
        
        # 工具
        self._notifier: Optional[Notifier] = None
        self._performance_monitor: Optional[PerformanceMonitor] = None
        
        # 策略註冊表
        self._strategy_registry: Optional[StrategyRegistry] = None
        
        # 事件循環
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._shutdown_event: Optional[asyncio.Event] = None
        
        logger.info(f"TradingSystem 初始化: mode={mode.value}, symbols={len(self._symbols)}")
    
    # ========== 屬性 ==========
    
    @property
    def state(self) -> SystemState:
        """系統狀態"""
        return self._state
    
    @property
    def is_running(self) -> bool:
        """是否運行中"""
        return self._state == SystemState.RUNNING
    
    @property
    def uptime(self) -> Optional[float]:
        """運行時間（秒）"""
        if self._start_time:
            return (datetime.now() - self._start_time).total_seconds()
        return None
    
    # ========== 初始化 ==========
    
    async def setup(self) -> bool:
        """
        初始化所有組件
        
        Returns:
            是否成功
        """
        logger.info("=" * 60)
        logger.info("IB Trading System 啟動中...")
        logger.info("=" * 60)
        
        self._state = SystemState.STARTING
        
        try:
            # 1. 事件總線
            logger.info("[1/10] 初始化 EventBus...")
            self._event_bus = EventBus()
            await self._event_bus.start()
            
            # 2. 通知服務
            logger.info("[2/10] 初始化 Notifier...")
            self._notifier = Notifier(NotificationConfig.from_env())
            await self._notifier.initialize()
            
            # 3. 性能監控
            logger.info("[3/10] 初始化 PerformanceMonitor...")
            self._performance_monitor = PerformanceMonitor(
                report_interval=300,  # 5 分鐘報告一次
                enable_system_monitoring=True,
            )
            self._performance_monitor.start()
            
            # 4. 數據庫和快取
            logger.info("[4/10] 初始化 Database & Cache...")
            self._database = Database(db_path=settings.database.path)
            await self._database.initialize()
            
            self._cache = MarketDataCache(
                max_ticks=settings.cache.max_ticks,
                max_bars=settings.cache.max_bars,
            )
            
            # 5. IB 連接
            logger.info("[5/10] 初始化 IBConnection...")
            ib_config = IBConfig(
                host=settings.ib.host,
                port=settings.ib.paper_port if self._mode == TradingMode.PAPER else settings.ib.live_port,
                client_id=settings.ib.client_id,
                timeout=settings.ib.timeout,
                readonly=(self._mode == TradingMode.BACKTEST),
            )
            self._connection = IBConnection(ib_config)
            self._contract_factory = ContractFactory()
            
            # 6. 數據處理
            logger.info("[6/10] 初始化 FeedHandler & BarAggregator...")
            self._feed_handler = FeedHandler(
                ib=self._connection.ib,
                event_bus=self._event_bus,
            )
            
            self._bar_aggregator = BarAggregator(
                event_bus=self._event_bus,
            )
            
            # 7. 風控組件
            logger.info("[7/10] 初始化風控組件...")
            self._risk_manager = RiskManager(
                event_bus=self._event_bus,
                account_value=settings.risk.account_value,
                daily_loss_limit=settings.risk.daily_loss_limit,
                max_position_value=settings.risk.max_position_value,
                max_total_exposure=settings.risk.max_total_exposure,
            )
            
            self._position_sizer = PositionSizer(
                account_value=settings.risk.account_value,
                risk_per_trade=settings.risk.risk_per_trade,
                max_position_pct=settings.risk.max_position_pct,
            )
            
            self._circuit_breaker = CircuitBreaker(
                config=BreakerConfig(
                    max_consecutive_losses=settings.risk.max_consecutive_losses,
                    max_loss_percent=settings.risk.max_drawdown,
                    cooldown_seconds=settings.risk.circuit_breaker_cooldown,
                ),
                event_bus=self._event_bus,
                account_value=settings.risk.account_value,
            )
            
            # 8. 策略引擎
            logger.info("[8/10] 初始化 StrategyEngine...")
            self._strategy_engine = StrategyEngine(
                event_bus=self._event_bus,
            )
            
            # 9. 執行引擎
            logger.info("[9/10] 初始化 ExecutionEngine...")
            self._execution_engine = ExecutionEngine(
                ib=self._connection.ib,
                event_bus=self._event_bus,
                risk_manager=self._risk_manager,
            )
            
            # 10. 策略註冊
            logger.info("[10/10] 註冊策略...")
            self._strategy_registry = StrategyRegistry()
            self._register_strategies()
            
            logger.info("=" * 60)
            logger.info("所有組件初始化完成")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"初始化失敗: {e}")
            self._state = SystemState.ERROR
            await self._notifier.alert(
                f"系統初始化失敗: {e}",
                level=NotificationLevel.CRITICAL,
            )
            return False
    
    def _register_strategies(self) -> None:
        """註冊可用策略"""
        # 註冊內建策略
        self._strategy_registry.register(
            "sma_cross",
            SMACrossStrategy,
            description="SMA 均線交叉策略",
            category="trend",
        )
        
        self._strategy_registry.register(
            "tick_scalper",
            TickScalperStrategy,
            description="Tick 高頻剝頭皮策略",
            category="scalping",
        )
        
        logger.info(f"已註冊 {len(self._strategy_registry.list_strategies())} 個策略")
    
    # ========== 連接 ==========
    
    async def connect(self) -> bool:
        """
        連接到 IB
        
        Returns:
            是否成功
        """
        if self._mode == TradingMode.BACKTEST:
            logger.info("回測模式，跳過 IB 連接")
            return True
        
        logger.info("連接到 Interactive Brokers...")
        
        success = self._connection.connect()
        
        if success:
            logger.info("IB 連接成功")
            
            # 訂閱合約
            await self._subscribe_symbols()
            
            return True
        else:
            logger.error("IB 連接失敗")
            await self._notifier.alert(
                "IB 連接失敗",
                level=NotificationLevel.ERROR,
            )
            return False
    
    async def _subscribe_symbols(self) -> None:
        """訂閱交易標的"""
        logger.info(f"訂閱 {len(self._symbols)} 個標的...")
        
        for symbol in self._symbols:
            try:
                # 建立合約
                contract = self._contract_factory.create_stock(symbol)
                
                # 訂閱數據
                await self._feed_handler.subscribe_realtime(
                    contract=contract,
                    tick_types=["Last", "BidAsk"],
                )
                
                logger.debug(f"已訂閱: {symbol}")
                
            except Exception as e:
                logger.error(f"訂閱 {symbol} 失敗: {e}")
    
    # ========== 運行 ==========
    
    async def run(self) -> None:
        """
        主運行循環
        """
        if self._state != SystemState.STARTING:
            logger.error(f"無法啟動，當前狀態: {self._state}")
            return
        
        self._state = SystemState.RUNNING
        self._start_time = datetime.now()
        self._shutdown_event = asyncio.Event()
        
        logger.info("=" * 60)
        logger.info(f"交易系統已啟動 - 模式: {self._mode.value}")
        logger.info(f"監控標的: {', '.join(self._symbols[:5])}{'...' if len(self._symbols) > 5 else ''}")
        logger.info("=" * 60)
        
        # 發送啟動通知
        await self._notifier.alert(
            f"交易系統已啟動\n模式: {self._mode.value}\n標的數: {len(self._symbols)}",
            level=NotificationLevel.INFO,
            title="系統啟動",
        )
        
        try:
            # 啟動組件
            self._risk_manager.start()
            self._circuit_breaker.start()
            self._strategy_engine.start()
            self._execution_engine.start()
            self._bar_aggregator.start()
            
            # 載入並啟動策略
            await self._load_strategies()
            
            # 主循環
            while not self._shutdown_event.is_set():
                await self._main_loop_tick()
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            logger.info("收到取消信號")
        except Exception as e:
            logger.error(f"運行錯誤: {e}")
            self._state = SystemState.ERROR
            await self._notifier.alert(
                f"系統運行錯誤: {e}",
                level=NotificationLevel.CRITICAL,
            )
        finally:
            await self.shutdown()
    
    async def _main_loop_tick(self) -> None:
        """主循環每秒執行"""
        # 檢查市場狀態
        if not is_market_open(include_extended=True):
            # 非交易時段
            remaining = time_until_market_open()
            if remaining.total_seconds() > 0 and int(remaining.total_seconds()) % 3600 == 0:
                logger.info(f"市場休市中，距離開盤: {format_duration(remaining)}")
        
        # 檢查連接狀態
        if self._mode != TradingMode.BACKTEST:
            if not self._connection.is_connected:
                logger.warning("IB 連接斷開，嘗試重連...")
                self._connection.reconnect()
        
        # 記錄性能指標
        self._performance_monitor.record_event("main_loop_tick")
    
    async def _load_strategies(self) -> None:
        """載入並啟動策略"""
        # 這裡可以根據配置載入不同策略
        # 目前使用 SMA 交叉作為示例
        
        from strategies.base import StrategyConfig
        
        strategy_config = StrategyConfig(
            name="sma_cross_default",
            symbols=self._symbols[:5],  # 限制前 5 個標的
            enabled=True,
            params={
                "fast_period": 10,
                "slow_period": 20,
            },
        )
        
        strategy = SMACrossStrategy(
            symbols=strategy_config.symbols,
            config=strategy_config,
            event_bus=self._event_bus,
        )
        
        self._strategy_engine.add_strategy(strategy)
        strategy.initialize()
        strategy.start()
        
        logger.info(f"已載入策略: {strategy_config.name}")
    
    # ========== 關閉 ==========
    
    async def shutdown(self) -> None:
        """
        優雅關閉系統
        """
        if self._state == SystemState.STOPPED:
            return
        
        logger.info("=" * 60)
        logger.info("系統關閉中...")
        logger.info("=" * 60)
        
        self._state = SystemState.STOPPING
        
        try:
            # 1. 停止策略引擎
            if self._strategy_engine:
                logger.info("停止策略引擎...")
                self._strategy_engine.stop()
            
            # 2. 取消所有未成交訂單
            if self._execution_engine:
                logger.info("取消未成交訂單...")
                await self._execution_engine.cancel_all_orders()
                self._execution_engine.stop()
            
            # 3. 停止風控組件
            if self._circuit_breaker:
                self._circuit_breaker.stop()
            if self._risk_manager:
                self._risk_manager.stop()
            
            # 4. 停止數據處理
            if self._bar_aggregator:
                self._bar_aggregator.stop()
            if self._feed_handler:
                await self._feed_handler.unsubscribe_all()
            
            # 5. 關閉數據庫
            if self._database:
                await self._database.close()
            
            # 6. 斷開 IB 連接
            if self._connection and self._connection.is_connected:
                logger.info("斷開 IB 連接...")
                self._connection.disconnect()
            
            # 7. 停止性能監控
            if self._performance_monitor:
                self._performance_monitor.stop()
            
            # 8. 停止事件總線
            if self._event_bus:
                await self._event_bus.stop()
            
            # 9. 關閉通知服務
            if self._notifier:
                await self._notifier.alert(
                    f"交易系統已關閉\n運行時間: {format_duration(datetime.now() - self._start_time) if self._start_time else 'N/A'}",
                    level=NotificationLevel.INFO,
                    title="系統關閉",
                )
                await self._notifier.shutdown()
            
            logger.info("=" * 60)
            logger.info("系統已安全關閉")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"關閉過程中發生錯誤: {e}")
        finally:
            self._state = SystemState.STOPPED
    
    def request_shutdown(self) -> None:
        """請求關閉（從外部調用）"""
        if self._shutdown_event:
            self._shutdown_event.set()
    
    # ========== 狀態查詢 ==========
    
    def get_status(self) -> dict:
        """取得系統狀態"""
        return {
            "state": self._state.name,
            "mode": self._mode.value,
            "uptime": self.uptime,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "symbols": len(self._symbols),
            "ib_connected": self._connection.is_connected if self._connection else False,
            "market_open": is_market_open(),
            "eastern_time": get_eastern_time().strftime("%Y-%m-%d %H:%M:%S"),
        }


# ============================================================
# 命令列解析
# ============================================================

def parse_args() -> argparse.Namespace:
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description="IB Trading System - Interactive Brokers 自動交易系統",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 模擬交易模式
  python main.py --mode paper
  
  # 實盤交易模式
  python main.py --mode live --symbols AAPL,GOOGL,MSFT
  
  # 回測模式
  python main.py --mode backtest --start 2023-01-01 --end 2023-12-31
        """,
    )
    
    # 交易模式
    parser.add_argument(
        "-m", "--mode",
        type=str,
        choices=["live", "paper", "backtest"],
        default="paper",
        help="交易模式 (預設: paper)",
    )
    
    # 交易標的
    parser.add_argument(
        "-s", "--symbols",
        type=str,
        default=None,
        help="交易標的，用逗號分隔 (例: AAPL,GOOGL,MSFT)",
    )
    
    # 配置檔
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="配置檔路徑",
    )
    
    # 日誌等級
    parser.add_argument(
        "-l", "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日誌等級 (預設: INFO)",
    )
    
    # 日誌目錄
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="日誌目錄 (預設: logs)",
    )
    
    # 回測參數
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="回測開始日期 (YYYY-MM-DD)",
    )
    
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="回測結束日期 (YYYY-MM-DD)",
    )
    
    # 其他選項
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="乾跑模式，不執行實際交易",
    )
    
    parser.add_argument(
        "-v", "--version",
        action="version",
        version="IB Trading System v1.0.0",
    )
    
    return parser.parse_args()


# ============================================================
# 主程式
# ============================================================

# 全局系統實例（用於信號處理）
_system: Optional[TradingSystem] = None


def signal_handler(signum: int, frame) -> None:
    """信號處理器"""
    sig_name = signal.Signals(signum).name
    logger.info(f"收到信號: {sig_name}")
    
    if _system:
        _system.request_shutdown()


async def main() -> int:
    """
    主程式入口
    
    Returns:
        退出碼
    """
    global _system
    
    # 解析參數
    args = parse_args()
    
    # 設定日誌
    setup_logger(
        log_dir=args.log_dir,
        log_level=args.log_level,
        console_output=True,
        file_output=True,
    )
    
    logger.info("=" * 60)
    logger.info("IB Trading System v1.0.0")
    logger.info("=" * 60)
    
    # 解析交易模式
    mode_map = {
        "live": TradingMode.LIVE,
        "paper": TradingMode.PAPER,
        "backtest": TradingMode.BACKTEST,
    }
    mode = mode_map[args.mode]
    
    # 解析標的
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    
    # 建立系統
    _system = TradingSystem(
        mode=mode,
        symbols=symbols,
        config_file=args.config,
    )
    
    # 設定信號處理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 初始化
        if not await _system.setup():
            logger.error("系統初始化失敗")
            return 1
        
        # 連接
        if not await _system.connect():
            logger.error("IB 連接失敗")
            return 2
        
        # 運行
        await _system.run()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("收到鍵盤中斷")
        return 0
    except Exception as e:
        logger.error(f"系統錯誤: {e}")
        return 3
    finally:
        if _system:
            await _system.shutdown()


def run() -> None:
    """程式入口點"""
    exit_code = asyncio.run(main())
    sys.exit(exit_code)


if __name__ == "__main__":
    run()