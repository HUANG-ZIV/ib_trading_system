#!/usr/bin/env python3
"""
三角套利即時交易執行腳本
Triangular Arbitrage Live Trading Runner

用於連接 IB 並執行三角套利策略
"""

import asyncio
import signal
import sys
import logging
from datetime import datetime
from typing import Dict, Optional, Any
from pathlib import Path

# 添加專案根目錄到 path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from ib_insync import IB, Forex, util
    HAS_IB = True
except ImportError:
    HAS_IB = False
    print("Error: ib_insync not installed. Install with: pip install ib_insync")
    sys.exit(1)

from strategies.triangular_arbitrage import (
    TriangularArbitrageStrategy,
    TriangularArbitrageConfig,
    TriangleType,
    SPOT_SYMBOLS,
)

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/triangular_{datetime.now().strftime('%Y%m%d')}.log"),
    ]
)
logger = logging.getLogger(__name__)


class TriangularArbitrageLiveRunner:
    """三角套利即時交易執行器"""
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        config: Optional[TriangularArbitrageConfig] = None,
    ):
        """
        初始化
        
        Args:
            host: IB Gateway/TWS 主機
            port: 端口
            client_id: 客戶端 ID
            config: 策略配置
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        
        # IB 連接
        self.ib = IB()
        
        # 策略配置
        self.config = config or TriangularArbitrageConfig(
            enabled_triangles=[
                TriangleType.T1_XAU_XAG_XPT,
                TriangleType.T2_XAU_XAG_XPD,
            ],
            entry_zscore=2.0,
            exit_zscore=0.5,
            capital_per_triangle=50000,
            max_triangles=2,
        )
        
        # 策略實例
        self.strategy = TriangularArbitrageStrategy(config=self.config)
        
        # 運行狀態
        self._running = False
        self._tickers = {}
        
        # 價格快取
        self._latest_prices: Dict[str, float] = {}
        
    async def connect(self) -> bool:
        """連接到 IB"""
        try:
            await self.ib.connectAsync(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
            )
            logger.info(f"Connected to IB at {self.host}:{self.port}")
            
            # 獲取帳戶資訊
            accounts = self.ib.managedAccounts()
            logger.info(f"Managed accounts: {accounts}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    def disconnect(self) -> None:
        """斷開連接"""
        self._running = False
        
        # 取消訂閱
        for symbol, ticker in self._tickers.items():
            contract = Forex(symbol)
            self.ib.cancelMktData(contract)
        
        self.ib.disconnect()
        logger.info("Disconnected from IB")
    
    async def subscribe_market_data(self) -> None:
        """訂閱市場數據"""
        symbols = list(SPOT_SYMBOLS.values())
        
        for symbol in symbols:
            try:
                contract = Forex(symbol)
                
                # 驗證合約
                qualified = await self.ib.qualifyContractsAsync(contract)
                if not qualified:
                    logger.warning(f"Could not qualify contract: {symbol}")
                    continue
                
                # 訂閱數據
                ticker = self.ib.reqMktData(contract, "", False, False)
                self._tickers[symbol] = ticker
                
                # 設置回調
                ticker.updateEvent += lambda t, s=symbol: self._on_price_update(s, t)
                
                logger.info(f"Subscribed to {symbol}")
                
            except Exception as e:
                logger.error(f"Error subscribing to {symbol}: {e}")
    
    def _on_price_update(self, symbol: str, ticker) -> None:
        """價格更新回調"""
        if ticker.midpoint():
            # 找到 asset key
            for asset, spot_symbol in SPOT_SYMBOLS.items():
                if spot_symbol == symbol:
                    self._latest_prices[asset] = ticker.midpoint()
                    break
    
    async def run(self) -> None:
        """主運行循環"""
        self._running = True
        self.strategy.start()
        
        logger.info("=" * 50)
        logger.info("Triangular Arbitrage Strategy Started")
        logger.info(f"Enabled triangles: {[t.value for t in self.config.enabled_triangles]}")
        logger.info(f"Entry Z-Score: {self.config.entry_zscore}")
        logger.info(f"Exit Z-Score: {self.config.exit_zscore}")
        logger.info(f"Capital per triangle: ${self.config.capital_per_triangle:,}")
        logger.info("=" * 50)
        
        # 訂閱數據
        await self.subscribe_market_data()
        
        # 等待初始數據
        logger.info("Waiting for initial market data...")
        await asyncio.sleep(5)
        
        # 主循環
        update_interval = 60  # 每分鐘更新一次
        last_status_time = datetime.now()
        
        while self._running:
            try:
                # 檢查是否有完整價格
                if len(self._latest_prices) >= 4:
                    # 模擬 bar 數據（使用當前價格）
                    timestamp = datetime.utcnow()
                    
                    for asset, price in self._latest_prices.items():
                        bar_data = {
                            "symbol": SPOT_SYMBOLS[asset],
                            "timestamp": timestamp,
                            "open": price,
                            "high": price,
                            "low": price,
                            "close": price,
                            "volume": 0,
                        }
                        self.strategy.on_bar(bar_data)
                
                # 定期顯示狀態
                if (datetime.now() - last_status_time).seconds >= 300:
                    self._print_status()
                    last_status_time = datetime.now()
                
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5)
        
        self.strategy.stop()
    
    def _print_status(self) -> None:
        """打印狀態"""
        status = self.strategy.get_status()
        triangle_states = self.strategy.get_triangle_states()
        positions = self.strategy.get_open_positions()
        
        logger.info("=" * 40)
        logger.info("STATUS UPDATE")
        logger.info(f"Running: {status['is_running']}")
        logger.info(f"Warming up: {status['is_warming_up']} ({status['warmup_progress']})")
        logger.info(f"Open positions: {status['open_positions']}")
        logger.info(f"Daily PnL: ${status['daily_pnl']:.2f}")
        logger.info(f"Total PnL: ${status['total_pnl']:.2f}")
        
        logger.info("\nTriangle States:")
        for name, state in triangle_states.items():
            logger.info(f"  {name}: Z={state['zscore']:.2f}, Dev={state['deviation_pct']:.3f}%")
        
        if positions:
            logger.info("\nOpen Positions:")
            for pos in positions:
                logger.info(f"  {pos['triangle']}: PnL=${pos['current_pnl']:.2f} ({pos['holding_days']}d)")
        
        logger.info("=" * 40)


async def main():
    """主函數"""
    # 創建日誌目錄
    Path("logs").mkdir(exist_ok=True)
    
    # 配置
    config = TriangularArbitrageConfig(
        enabled_triangles=[
            TriangleType.T1_XAU_XAG_XPT,
            TriangleType.T2_XAU_XAG_XPD,
        ],
        lookback_period=120,
        entry_zscore=2.0,
        exit_zscore=0.5,
        stop_zscore=3.5,
        min_deviation_pct=0.5,
        capital_per_triangle=50000,
        max_triangles=2,
        warmup_bars=150,
    )
    
    runner = TriangularArbitrageLiveRunner(
        host="127.0.0.1",
        port=7497,  # TWS Paper Trading
        client_id=1,
        config=config,
    )
    
    # 設置信號處理
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        runner.disconnect()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 連接並運行
    if await runner.connect():
        try:
            await runner.run()
        finally:
            runner.disconnect()
    else:
        logger.error("Failed to connect, exiting")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
