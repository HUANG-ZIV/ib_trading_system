"""
Tests 模組 - 測試套件

提供共用的測試工具和 fixtures
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, MagicMock
import random

# ============================================================
# Python 3.14+ 相容性修復
# ============================================================
# Python 3.14 移除了自動建立事件循環的功能
# 需要在導入 ib_insync 之前手動設定

try:
    asyncio.get_running_loop()
except RuntimeError:
    # 沒有運行中的事件循環，建立一個新的
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# ============================================================
# Mock ib_insync（在導入其他模組前）
# ============================================================

# 建立 Mock 類別
class MockIB:
    def __init__(self):
        self._connected = False
    def connect(self, *args, **kwargs):
        self._connected = True
        return self
    def disconnect(self):
        self._connected = False
    def isConnected(self):
        return self._connected
    def qualifyContracts(self, *contracts):
        return list(contracts)
    def reqMktData(self, *args, **kwargs):
        return MagicMock()
    def cancelMktData(self, *args):
        pass
    def placeOrder(self, contract, order):
        return MagicMock()
    def cancelOrder(self, *args):
        pass
    def positions(self):
        return []
    def accountSummary(self):
        return []
    def openOrders(self):
        return []


class MockContract:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# 建立 mock 模組
mock_ib_insync = MagicMock()
mock_ib_insync.IB = MockIB
mock_ib_insync.Contract = MockContract
mock_ib_insync.Stock = lambda *args, **kwargs: MockContract(secType='STK', **kwargs)
mock_ib_insync.Future = lambda *args, **kwargs: MockContract(secType='FUT', **kwargs)
mock_ib_insync.Option = lambda *args, **kwargs: MockContract(secType='OPT', **kwargs)
mock_ib_insync.Forex = lambda *args, **kwargs: MockContract(secType='CASH', **kwargs)
mock_ib_insync.Index = lambda *args, **kwargs: MockContract(secType='IND', **kwargs)
mock_ib_insync.Crypto = lambda *args, **kwargs: MockContract(secType='CRYPTO', **kwargs)
mock_ib_insync.Order = MagicMock
mock_ib_insync.LimitOrder = MagicMock
mock_ib_insync.MarketOrder = MagicMock
mock_ib_insync.StopOrder = MagicMock
mock_ib_insync.Trade = MagicMock
mock_ib_insync.util = MagicMock()

# 注入到 sys.modules
if 'ib_insync' not in sys.modules:
    sys.modules['ib_insync'] = mock_ib_insync

# 確保專案根目錄在 Python 路徑中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# 測試常數
# ============================================================

TEST_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
TEST_INITIAL_CAPITAL = 100000.0
TEST_COMMISSION_RATE = 0.001


# ============================================================
# Mock 工廠函數
# ============================================================

def create_mock_ib():
    """建立 Mock IB 連接"""
    mock_ib = MagicMock()
    mock_ib.isConnected.return_value = True
    mock_ib.client.getReqId.return_value = 1
    return mock_ib


def create_mock_contract(symbol: str = "AAPL"):
    """建立 Mock Contract"""
    mock_contract = MagicMock()
    mock_contract.symbol = symbol
    mock_contract.secType = "STK"
    mock_contract.exchange = "SMART"
    mock_contract.currency = "USD"
    return mock_contract


def create_mock_order(
    action: str = "BUY",
    quantity: int = 100,
    order_type: str = "MKT",
):
    """建立 Mock Order"""
    mock_order = MagicMock()
    mock_order.action = action
    mock_order.totalQuantity = quantity
    mock_order.orderType = order_type
    return mock_order


# ============================================================
# 測試數據生成
# ============================================================

def generate_bar_data(
    symbol: str = "AAPL",
    start_price: float = 150.0,
    num_bars: int = 100,
    volatility: float = 0.02,
    start_time: Optional[datetime] = None,
    interval_minutes: int = 1,
) -> List[Dict[str, Any]]:
    """
    生成模擬 K 線數據
    
    Args:
        symbol: 標的代碼
        start_price: 起始價格
        num_bars: K 線數量
        volatility: 波動率
        start_time: 開始時間
        interval_minutes: K 線間隔（分鐘）
        
    Returns:
        K 線數據列表
    """
    if start_time is None:
        start_time = datetime.now() - timedelta(minutes=num_bars * interval_minutes)
    
    bars = []
    price = start_price
    
    for i in range(num_bars):
        # 隨機價格變動
        change = random.gauss(0, volatility)
        price = price * (1 + change)
        
        # 生成 OHLC
        high = price * (1 + abs(random.gauss(0, volatility / 2)))
        low = price * (1 - abs(random.gauss(0, volatility / 2)))
        open_price = random.uniform(low, high)
        close_price = random.uniform(low, high)
        volume = random.randint(1000, 100000)
        
        bar = {
            "symbol": symbol,
            "timestamp": start_time + timedelta(minutes=i * interval_minutes),
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close_price, 2),
            "volume": volume,
        }
        bars.append(bar)
        
        price = close_price
    
    return bars


def generate_tick_data(
    symbol: str = "AAPL",
    base_price: float = 150.0,
    num_ticks: int = 1000,
    spread: float = 0.01,
) -> List[Dict[str, Any]]:
    """
    生成模擬 Tick 數據
    
    Args:
        symbol: 標的代碼
        base_price: 基礎價格
        num_ticks: Tick 數量
        spread: 買賣價差
        
    Returns:
        Tick 數據列表
    """
    ticks = []
    price = base_price
    timestamp = datetime.now()
    
    for i in range(num_ticks):
        # 隨機價格變動
        price = price * (1 + random.gauss(0, 0.001))
        
        bid = price - spread / 2
        ask = price + spread / 2
        
        tick = {
            "symbol": symbol,
            "timestamp": timestamp + timedelta(milliseconds=i * 100),
            "bid": round(bid, 2),
            "ask": round(ask, 2),
            "last": round(price, 2),
            "bid_size": random.randint(100, 1000),
            "ask_size": random.randint(100, 1000),
            "last_size": random.randint(1, 100),
        }
        ticks.append(tick)
    
    return ticks


def generate_trade_sequence(
    num_trades: int = 20,
    win_rate: float = 0.5,
    avg_win: float = 100.0,
    avg_loss: float = -80.0,
) -> List[Dict[str, Any]]:
    """
    生成模擬交易序列
    
    Args:
        num_trades: 交易數量
        win_rate: 勝率
        avg_win: 平均獲利
        avg_loss: 平均虧損
        
    Returns:
        交易記錄列表
    """
    trades = []
    
    for i in range(num_trades):
        is_win = random.random() < win_rate
        
        if is_win:
            pnl = abs(random.gauss(avg_win, avg_win * 0.3))
        else:
            pnl = -abs(random.gauss(abs(avg_loss), abs(avg_loss) * 0.3))
        
        trade = {
            "trade_id": i + 1,
            "symbol": random.choice(TEST_SYMBOLS),
            "pnl": round(pnl, 2),
            "is_win": is_win,
            "timestamp": datetime.now() - timedelta(days=num_trades - i),
        }
        trades.append(trade)
    
    return trades


# ============================================================
# 測試輔助類
# ============================================================

class MockEventBus:
    """Mock EventBus 用於測試"""
    
    def __init__(self):
        self.published_events = []
        self.subscribers = {}
    
    def publish(self, event):
        self.published_events.append(event)
    
    def subscribe(self, event_type, callback, priority=0):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type, callback):
        if event_type in self.subscribers:
            self.subscribers[event_type].remove(callback)
    
    def clear(self):
        self.published_events.clear()


class MockStrategy:
    """Mock Strategy 用於測試"""
    
    def __init__(self, symbols=None):
        self.symbols = symbols or ["AAPL"]
        self.bars_received = []
        self.signals_emitted = []
        self.initialized = False
        self.started = False
        self.stopped = False
    
    def initialize(self):
        self.initialized = True
    
    def start(self):
        self.started = True
    
    def stop(self):
        self.stopped = True
    
    def on_bar(self, bar):
        self.bars_received.append(bar)
    
    def emit_signal(self, **kwargs):
        self.signals_emitted.append(kwargs)


# ============================================================
# pytest fixtures（供 conftest.py 使用）
# ============================================================

def get_test_config():
    """取得測試配置"""
    return {
        "initial_capital": TEST_INITIAL_CAPITAL,
        "commission_rate": TEST_COMMISSION_RATE,
        "symbols": TEST_SYMBOLS,
    }


# ============================================================
# 匯出
# ============================================================

__all__ = [
    # 常數
    "TEST_SYMBOLS",
    "TEST_INITIAL_CAPITAL",
    "TEST_COMMISSION_RATE",
    # Mock 工廠
    "create_mock_ib",
    "create_mock_contract",
    "create_mock_order",
    # 數據生成
    "generate_bar_data",
    "generate_tick_data",
    "generate_trade_sequence",
    # 輔助類
    "MockEventBus",
    "MockStrategy",
    # 配置
    "get_test_config",
]