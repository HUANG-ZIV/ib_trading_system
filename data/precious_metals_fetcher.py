"""
IB 貴金屬數據獲取工具
IB Precious Metals Data Fetcher

用於獲取黃金、白銀、鉑金、鈀金的歷史和即時數據
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from pathlib import Path

try:
    from ib_insync import IB, Forex, Future, Contract, util
    HAS_IB = True
except ImportError:
    HAS_IB = False
    print("Warning: ib_insync not installed. Install with: pip install ib_insync")

logger = logging.getLogger(__name__)


@dataclass
class PreciousMetalContract:
    """貴金屬合約定義"""
    symbol: str           # 代碼
    asset_key: str        # 資產鍵 (XAU, XAG, etc.)
    contract_type: str    # spot / futures
    exchange: str         # 交易所
    currency: str         # 報價貨幣
    multiplier: int       # 合約乘數
    
    def to_ib_contract(self) -> Optional['Contract']:
        """轉換為 IB 合約"""
        if not HAS_IB:
            return None
        
        if self.contract_type == "spot":
            # 現貨外匯
            return Forex(self.symbol)
        elif self.contract_type == "futures":
            # 期貨
            return Future(
                symbol=self.symbol,
                exchange=self.exchange,
                currency=self.currency,
            )
        return None


# 預定義合約
PRECIOUS_METAL_CONTRACTS = {
    # 現貨
    "XAUUSD": PreciousMetalContract("XAUUSD", "XAU", "spot", "IDEALPRO", "USD", 1),
    "XAGUSD": PreciousMetalContract("XAGUSD", "XAG", "spot", "IDEALPRO", "USD", 1),
    "XPTUSD": PreciousMetalContract("XPTUSD", "XPT", "spot", "IDEALPRO", "USD", 1),
    "XPDUSD": PreciousMetalContract("XPDUSD", "XPD", "spot", "IDEALPRO", "USD", 1),
    
    # 期貨（需要指定到期月份）
    "GC": PreciousMetalContract("GC", "XAU", "futures", "COMEX", "USD", 100),
    "SI": PreciousMetalContract("SI", "XAG", "futures", "COMEX", "USD", 5000),
    "PL": PreciousMetalContract("PL", "XPT", "futures", "NYMEX", "USD", 50),
    "PA": PreciousMetalContract("PA", "XPD", "futures", "NYMEX", "USD", 100),
}


class IBPreciousMetalsFetcher:
    """IB 貴金屬數據獲取器"""
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 10,
    ):
        """
        初始化
        
        Args:
            host: IB Gateway/TWS 主機
            port: 端口 (7497=TWS Paper, 7496=TWS Live, 4002=Gateway Paper, 4001=Gateway Live)
            client_id: 客戶端 ID
        """
        if not HAS_IB:
            raise ImportError("ib_insync is required. Install with: pip install ib_insync")
        
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()
        self._connected = False
    
    async def connect(self) -> bool:
        """連接到 IB"""
        try:
            await self.ib.connectAsync(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
            )
            self._connected = True
            logger.info(f"Connected to IB at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            return False
    
    def connect_sync(self) -> bool:
        """同步連接到 IB"""
        try:
            self.ib.connect(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
            )
            self._connected = True
            logger.info(f"Connected to IB at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            return False
    
    def disconnect(self) -> None:
        """斷開連接"""
        if self._connected:
            self.ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IB")
    
    def get_spot_contract(self, symbol: str) -> Optional[Contract]:
        """取得現貨合約"""
        if symbol in PRECIOUS_METAL_CONTRACTS:
            return PRECIOUS_METAL_CONTRACTS[symbol].to_ib_contract()
        
        # 嘗試直接創建 Forex
        return Forex(symbol)
    
    def get_futures_contract(
        self,
        symbol: str,
        expiry: Optional[str] = None,
    ) -> Optional[Contract]:
        """
        取得期貨合約
        
        Args:
            symbol: 期貨代碼 (GC, SI, PL, PA)
            expiry: 到期月份 (YYYYMM)，若不指定則使用最近月
        """
        if symbol not in PRECIOUS_METAL_CONTRACTS:
            return None
        
        contract_def = PRECIOUS_METAL_CONTRACTS[symbol]
        
        contract = Future(
            symbol=symbol,
            exchange=contract_def.exchange,
            currency=contract_def.currency,
        )
        
        if expiry:
            contract.lastTradeDateOrContractMonth = expiry
        
        # 讓 IB 解析合約詳情
        details = self.ib.reqContractDetails(contract)
        if details:
            return details[0].contract
        
        return contract
    
    def fetch_historical_data(
        self,
        symbols: List[str] = None,
        duration: str = "1 Y",
        bar_size: str = "1 hour",
        end_datetime: str = "",
        what_to_show: str = "MIDPOINT",
    ) -> pd.DataFrame:
        """
        獲取歷史數據
        
        Args:
            symbols: 要獲取的商品列表，預設為四種貴金屬現貨
            duration: 數據時長 (e.g., "1 Y", "6 M", "30 D")
            bar_size: K線週期 (e.g., "1 hour", "1 day", "5 mins")
            end_datetime: 結束時間，空字串表示當前
            what_to_show: 價格類型 (MIDPOINT, BID, ASK, TRADES)
            
        Returns:
            DataFrame with columns: datetime, XAU, XAG, XPT, XPD
        """
        if not self._connected:
            raise RuntimeError("Not connected to IB")
        
        if symbols is None:
            symbols = ["XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD"]
        
        all_data = {}
        
        for symbol in symbols:
            contract = self.get_spot_contract(symbol)
            if not contract:
                logger.warning(f"Could not create contract for {symbol}")
                continue
            
            try:
                bars = self.ib.reqHistoricalData(
                    contract=contract,
                    endDateTime=end_datetime,
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow=what_to_show,
                    useRTH=False,
                    formatDate=1,
                )
                
                if bars:
                    df = util.df(bars)
                    # 使用 asset key
                    asset_key = PRECIOUS_METAL_CONTRACTS[symbol].asset_key
                    all_data[asset_key] = df.set_index("date")["close"]
                    logger.info(f"Fetched {len(bars)} bars for {symbol}")
                else:
                    logger.warning(f"No data returned for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
        
        if not all_data:
            return pd.DataFrame()
        
        # 合併數據
        result = pd.DataFrame(all_data)
        result.index.name = "datetime"
        result = result.reset_index()
        
        # 處理缺失值
        result = result.ffill().bfill()
        
        return result
    
    def fetch_realtime_prices(
        self,
        symbols: List[str] = None,
    ) -> Dict[str, float]:
        """
        獲取即時價格
        
        Args:
            symbols: 商品列表
            
        Returns:
            價格字典 {"XAU": 2000.0, "XAG": 25.0, ...}
        """
        if not self._connected:
            raise RuntimeError("Not connected to IB")
        
        if symbols is None:
            symbols = ["XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD"]
        
        prices = {}
        
        for symbol in symbols:
            contract = self.get_spot_contract(symbol)
            if not contract:
                continue
            
            try:
                ticker = self.ib.reqMktData(contract, "", False, False)
                self.ib.sleep(1)  # 等待數據
                
                if ticker.midpoint():
                    asset_key = PRECIOUS_METAL_CONTRACTS[symbol].asset_key
                    prices[asset_key] = ticker.midpoint()
                
                self.ib.cancelMktData(contract)
                
            except Exception as e:
                logger.error(f"Error fetching price for {symbol}: {e}")
        
        return prices
    
    def subscribe_realtime(
        self,
        symbols: List[str] = None,
        callback: callable = None,
    ) -> Dict[str, Any]:
        """
        訂閱即時數據
        
        Args:
            symbols: 商品列表
            callback: 價格更新回調函數，簽名: callback(symbol, price, timestamp)
            
        Returns:
            訂閱的 ticker 字典
        """
        if not self._connected:
            raise RuntimeError("Not connected to IB")
        
        if symbols is None:
            symbols = ["XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD"]
        
        tickers = {}
        
        for symbol in symbols:
            contract = self.get_spot_contract(symbol)
            if not contract:
                continue
            
            ticker = self.ib.reqMktData(contract, "", False, False)
            tickers[symbol] = ticker
            
            if callback:
                def on_update(t, symbol=symbol):
                    if t.midpoint():
                        asset_key = PRECIOUS_METAL_CONTRACTS[symbol].asset_key
                        callback(asset_key, t.midpoint(), datetime.now())
                
                ticker.updateEvent += on_update
        
        return tickers
    
    def unsubscribe_realtime(self, tickers: Dict[str, Any]) -> None:
        """取消訂閱"""
        for symbol, ticker in tickers.items():
            contract = self.get_spot_contract(symbol)
            if contract:
                self.ib.cancelMktData(contract)


def download_historical_data(
    output_path: str = "precious_metals_data.csv",
    duration: str = "5 Y",
    bar_size: str = "1 hour",
    host: str = "127.0.0.1",
    port: int = 7497,
) -> Optional[pd.DataFrame]:
    """
    下載歷史數據的便捷函數
    
    Args:
        output_path: 輸出文件路徑
        duration: 數據時長
        bar_size: K線週期
        host: IB 主機
        port: IB 端口
        
    Returns:
        DataFrame 或 None
    """
    if not HAS_IB:
        print("ib_insync not installed")
        return None
    
    fetcher = IBPreciousMetalsFetcher(host=host, port=port, client_id=99)
    
    try:
        if not fetcher.connect_sync():
            return None
        
        df = fetcher.fetch_historical_data(
            duration=duration,
            bar_size=bar_size,
        )
        
        if not df.empty:
            df.to_csv(output_path, index=False)
            print(f"Data saved to {output_path}")
            print(f"Shape: {df.shape}")
            print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        return df
        
    finally:
        fetcher.disconnect()


# ==================== 測試 ====================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 下載數據
    df = download_historical_data(
        output_path="precious_metals_1y.csv",
        duration="1 Y",
        bar_size="1 hour",
    )
    
    if df is not None:
        print("\nData sample:")
        print(df.head(10))
        print("\nData statistics:")
        print(df.describe())
