"""
conftest.py - Pytest 配置檔案

處理 Python 3.14+ 的 asyncio 相容性問題
必須在任何其他模組導入前執行
"""

import sys
import asyncio
import unittest.mock as mock

# ============================================================
# Python 3.14+ asyncio 相容性修復（最高優先級）
# ============================================================

def _ensure_event_loop():
    """確保有可用的 event loop"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

# 立即執行
_ensure_event_loop()


# ============================================================
# Mock ib_insync 模組（在導入前）
# ============================================================

class MockIB:
    """Mock IB 類"""
    def __init__(self):
        self._connected = False
        
    def connect(self, host='127.0.0.1', port=7497, clientId=1, timeout=30, readonly=False):
        self._connected = True
        return self
        
    def disconnect(self):
        self._connected = False
        
    def isConnected(self):
        return self._connected
    
    def qualifyContracts(self, *contracts):
        return list(contracts)
    
    def reqMktData(self, contract, genericTickList='', snapshot=False, regulatorySnapshot=False):
        return mock.MagicMock()
    
    def cancelMktData(self, contract):
        pass
    
    def reqHistoricalData(self, contract, endDateTime='', durationStr='1 D', 
                          barSizeSetting='1 hour', whatToShow='TRADES', 
                          useRTH=True, formatDate=1, keepUpToDate=False):
        return []
    
    def placeOrder(self, contract, order):
        trade = mock.MagicMock()
        trade.order = order
        trade.contract = contract
        return trade
    
    def cancelOrder(self, order):
        pass
    
    def positions(self):
        return []
    
    def accountSummary(self):
        return []
    
    def openOrders(self):
        return []
    
    def sleep(self, seconds):
        import time
        time.sleep(seconds)


class MockContract:
    """Mock Contract 類"""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockOrder:
    """Mock Order 類"""
    def __init__(self, **kwargs):
        self.orderId = 0
        self.action = kwargs.get('action', 'BUY')
        self.totalQuantity = kwargs.get('totalQuantity', 0)
        self.orderType = kwargs.get('orderType', 'MKT')
        self.lmtPrice = kwargs.get('lmtPrice', 0)
        self.auxPrice = kwargs.get('auxPrice', 0)


def MockStock(symbol, exchange='SMART', currency='USD'):
    return MockContract(symbol=symbol, secType='STK', exchange=exchange, currency=currency)

def MockFuture(symbol, lastTradeDateOrContractMonth='', exchange='', currency='USD', multiplier=''):
    return MockContract(symbol=symbol, secType='FUT', exchange=exchange, currency=currency,
                       lastTradeDateOrContractMonth=lastTradeDateOrContractMonth, multiplier=multiplier)

def MockOption(symbol, lastTradeDateOrContractMonth='', strike=0, right='', exchange='', currency='USD', multiplier='100'):
    return MockContract(symbol=symbol, secType='OPT', exchange=exchange, currency=currency,
                       lastTradeDateOrContractMonth=lastTradeDateOrContractMonth, strike=strike, 
                       right=right, multiplier=multiplier)

def MockForex(pair='', exchange='IDEALPRO', symbol='', currency=''):
    return MockContract(symbol=symbol or pair[:3], secType='CASH', exchange=exchange, 
                       currency=currency or pair[3:] if pair else 'USD')

def MockIndex(symbol, exchange='', currency='USD'):
    return MockContract(symbol=symbol, secType='IND', exchange=exchange, currency=currency)

def MockCrypto(symbol, exchange='PAXOS', currency='USD'):
    return MockContract(symbol=symbol, secType='CRYPTO', exchange=exchange, currency=currency)

def MockLimitOrder(action, totalQuantity, lmtPrice):
    return MockOrder(action=action, totalQuantity=totalQuantity, orderType='LMT', lmtPrice=lmtPrice)

def MockMarketOrder(action, totalQuantity):
    return MockOrder(action=action, totalQuantity=totalQuantity, orderType='MKT')

def MockStopOrder(action, totalQuantity, stopPrice):
    return MockOrder(action=action, totalQuantity=totalQuantity, orderType='STP', auxPrice=stopPrice)


class MockUtil:
    """Mock util 模組"""
    @staticmethod
    def startLoop():
        pass
    
    @staticmethod
    def run(*args, **kwargs):
        pass


# 建立 mock 模組
mock_ib_insync = mock.MagicMock()
mock_ib_insync.IB = MockIB
mock_ib_insync.Contract = MockContract
mock_ib_insync.Stock = MockStock
mock_ib_insync.Future = MockFuture
mock_ib_insync.Option = MockOption
mock_ib_insync.Forex = MockForex
mock_ib_insync.Index = MockIndex
mock_ib_insync.Crypto = MockCrypto
mock_ib_insync.Order = MockOrder
mock_ib_insync.LimitOrder = MockLimitOrder
mock_ib_insync.MarketOrder = MockMarketOrder
mock_ib_insync.StopOrder = MockStopOrder
mock_ib_insync.Trade = mock.MagicMock
mock_ib_insync.util = MockUtil

# 注入到 sys.modules（在其他模組導入前）
sys.modules['ib_insync'] = mock_ib_insync


# ============================================================
# Pytest Fixtures
# ============================================================

import pytest


@pytest.fixture(scope="session")
def event_loop():
    """建立 session-scoped event loop"""
    _ensure_event_loop()
    loop = asyncio.get_event_loop()
    yield loop


@pytest.fixture
def mock_ib():
    """提供 Mock IB 實例"""
    return MockIB()


@pytest.fixture
def mock_contract():
    """提供 Mock Contract"""
    return MockStock("AAPL")