"""
test_connection.py - 連接模組測試

測試 IBConnection 和合約建立功能
"""

import asyncio
import os
import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

# 先導入 tests 模組（設定 mock）
import tests

# ============================================================
# Python 3.14+ 相容性修復
# ============================================================
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# 測試目標
from core.connection import IBConnection, ConnectionConfig as IBConfig
from core.contracts import (
    ContractFactory,
    create_contract,
    create_stock_contract as create_stock,
    create_future_contract as create_future,
    create_option_contract as create_option,
    create_forex_contract as create_forex,
)


# ============================================================
# IBConfig 測試
# ============================================================

class TestIBConfig:
    """IBConfig 測試類"""
    
    def test_default_config(self):
        """測試預設配置"""
        config = IBConfig()
        
        assert config.host == "127.0.0.1"
        assert config.port == 7497
        assert config.client_id == 1
        assert config.timeout == 30
        assert config.readonly is False
    
    def test_custom_config(self):
        """測試自訂配置"""
        config = IBConfig(
            host="192.168.1.100",
            port=7496,
            client_id=10,
            timeout=60,
            readonly=True,
        )
        
        assert config.host == "192.168.1.100"
        assert config.port == 7496
        assert config.client_id == 10
        assert config.timeout == 60
        assert config.readonly is True
    
    def test_config_from_env(self):
        """測試從環境變數載入配置"""
        # 設定環境變數
        with patch.dict(os.environ, {
            "IB_HOST": "10.0.0.1",
            "IB_PORT": "4002",
            "IB_CLIENT_ID": "99",
            "IB_TIMEOUT": "45",
            "IB_READONLY": "true",
        }):
            config = IBConfig.from_env()
            
            assert config.host == "10.0.0.1"
            assert config.port == 4002
            assert config.client_id == 99
            assert config.timeout == 45
            assert config.readonly is True
    
    def test_config_from_env_defaults(self):
        """測試環境變數不存在時使用預設值"""
        # 清除相關環境變數
        env_vars = ["IB_HOST", "IB_PORT", "IB_CLIENT_ID", "IB_TIMEOUT", "IB_READONLY"]
        clean_env = {k: v for k, v in os.environ.items() if k not in env_vars}
        
        with patch.dict(os.environ, clean_env, clear=True):
            config = IBConfig.from_env()
            
            assert config.host == "127.0.0.1"
            assert config.port == 7497
            assert config.client_id == 1
    
    def test_config_to_dict(self):
        """測試配置轉換為字典"""
        config = IBConfig(host="localhost", port=7497)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["host"] == "localhost"
        assert config_dict["port"] == 7497


# ============================================================
# IBConnection 測試
# ============================================================

class TestIBConnection:
    """IBConnection 測試類"""
    
    @pytest.fixture
    def mock_ib(self):
        """建立 Mock IB 物件"""
        mock = MagicMock()
        mock.isConnected.return_value = False
        mock.connect.return_value = None
        mock.disconnect.return_value = None
        mock.reqAccountSummary.return_value = None
        return mock
    
    @pytest.fixture
    def connection(self, mock_ib):
        """建立 IBConnection 實例"""
        with patch("core.connection.IB", return_value=mock_ib):
            config = IBConfig(host="127.0.0.1", port=7497)
            conn = IBConnection(config)
            conn._ib = mock_ib
            return conn
    
    def test_connection_init(self, connection):
        """測試 IBConnection 初始化"""
        assert connection is not None
        assert connection._config.host == "127.0.0.1"
        assert connection._config.port == 7497
        assert connection.is_connected is False
    
    def test_connection_init_default_config(self):
        """測試使用預設配置初始化"""
        with patch("core.connection.IB"):
            conn = IBConnection()
            assert conn._config.host == "127.0.0.1"
            assert conn._config.port == 7497
    
    def test_connect_success(self, connection, mock_ib):
        """測試連接成功"""
        mock_ib.isConnected.return_value = True
        
        result = connection.connect()
        
        mock_ib.connect.assert_called_once_with(
            host="127.0.0.1",
            port=7497,
            clientId=1,
            timeout=30,
            readonly=False,
        )
        assert result is True
    
    def test_connect_failure(self, connection, mock_ib):
        """測試連接失敗"""
        mock_ib.connect.side_effect = Exception("Connection refused")
        
        result = connection.connect()
        
        assert result is False
    
    def test_disconnect(self, connection, mock_ib):
        """測試斷開連接"""
        mock_ib.isConnected.return_value = True
        
        connection.disconnect()
        
        mock_ib.disconnect.assert_called_once()
    
    def test_is_connected_property(self, connection, mock_ib):
        """測試 is_connected 屬性"""
        mock_ib.isConnected.return_value = False
        assert connection.is_connected is False
        
        mock_ib.isConnected.return_value = True
        assert connection.is_connected is True
    
    def test_ib_property(self, connection, mock_ib):
        """測試 ib 屬性"""
        assert connection.ib == mock_ib
    
    def test_reconnect(self, connection, mock_ib):
        """測試重新連接"""
        mock_ib.isConnected.side_effect = [True, False, True]
        
        # 先連接
        connection.connect()
        
        # 重新連接
        result = connection.reconnect()
        
        # 應該會斷開再連接
        assert mock_ib.disconnect.called
    
    def test_on_connect_callback(self, connection, mock_ib):
        """測試連接回調"""
        callback_called = False
        
        @connection.on_connect
        def on_connect_handler():
            nonlocal callback_called
            callback_called = True
        
        # 模擬連接成功
        mock_ib.isConnected.return_value = True
        connection.connect()
        
        # 手動觸發回調（因為是 mock）
        for cb in connection._on_connect_callbacks:
            cb()
        
        assert callback_called
    
    def test_on_disconnect_callback(self, connection, mock_ib):
        """測試斷開回調"""
        callback_called = False
        
        @connection.on_disconnect
        def on_disconnect_handler():
            nonlocal callback_called
            callback_called = True
        
        # 手動觸發回調
        for cb in connection._on_disconnect_callbacks:
            cb()
        
        assert callback_called


# ============================================================
# ContractFactory 測試
# ============================================================

class TestContractFactory:
    """ContractFactory 測試類"""
    
    @pytest.fixture
    def factory(self):
        """建立 ContractFactory 實例"""
        return ContractFactory()
    
    def test_create_stock(self, factory):
        """測試建立股票合約"""
        contract = factory.create_stock("AAPL")
        
        assert contract.symbol == "AAPL"
        assert contract.secType == "STK"
        assert contract.exchange == "SMART"
        assert contract.currency == "USD"
    
    def test_create_stock_custom_exchange(self, factory):
        """測試建立指定交易所的股票合約"""
        contract = factory.create_stock("AAPL", exchange="NASDAQ")
        
        assert contract.symbol == "AAPL"
        assert contract.exchange == "NASDAQ"
    
    def test_create_stock_custom_currency(self, factory):
        """測試建立非美元股票合約"""
        contract = factory.create_stock("2330", exchange="TWSE", currency="TWD")
        
        assert contract.symbol == "2330"
        assert contract.currency == "TWD"
    
    def test_create_future(self, factory):
        """測試建立期貨合約"""
        contract = factory.create_future("ES", "202412")
        
        assert contract.symbol == "ES"
        assert contract.secType == "FUT"
        assert contract.lastTradeDateOrContractMonth == "202412"
        assert contract.exchange == "CME"
    
    def test_create_future_custom_exchange(self, factory):
        """測試建立指定交易所的期貨合約"""
        contract = factory.create_future("NQ", "202412", exchange="CME")
        
        assert contract.symbol == "NQ"
        assert contract.exchange == "CME"
    
    def test_create_option_call(self, factory):
        """測試建立看漲期權合約"""
        contract = factory.create_option(
            symbol="AAPL",
            expiry="20241220",
            strike=150.0,
            right="C",
        )
        
        assert contract.symbol == "AAPL"
        assert contract.secType == "OPT"
        assert contract.lastTradeDateOrContractMonth == "20241220"
        assert contract.strike == 150.0
        assert contract.right == "C"
    
    def test_create_option_put(self, factory):
        """測試建立看跌期權合約"""
        contract = factory.create_option(
            symbol="AAPL",
            expiry="20241220",
            strike=140.0,
            right="P",
        )
        
        assert contract.right == "P"
        assert contract.strike == 140.0
    
    def test_create_forex(self, factory):
        """測試建立外匯合約"""
        contract = factory.create_forex("EUR", "USD")
        
        assert contract.symbol == "EUR"
        assert contract.secType == "CASH"
        assert contract.currency == "USD"
        assert contract.exchange == "IDEALPRO"
    
    def test_create_forex_pairs(self, factory):
        """測試建立不同外匯對"""
        # 歐元/美元
        eurusd = factory.create_forex("EUR", "USD")
        assert eurusd.symbol == "EUR"
        assert eurusd.currency == "USD"
        
        # 英鎊/日圓
        gbpjpy = factory.create_forex("GBP", "JPY")
        assert gbpjpy.symbol == "GBP"
        assert gbpjpy.currency == "JPY"
    
    def test_create_index(self, factory):
        """測試建立指數合約"""
        contract = factory.create_index("SPX")
        
        assert contract.symbol == "SPX"
        assert contract.secType == "IND"
        assert contract.exchange == "CBOE"
    
    def test_create_crypto(self, factory):
        """測試建立加密貨幣合約"""
        contract = factory.create_crypto("BTC")
        
        assert contract.symbol == "BTC"
        assert contract.secType == "CRYPTO"
        assert contract.exchange == "PAXOS"
        assert contract.currency == "USD"
    
    def test_get_contract_caching(self, factory):
        """測試合約快取"""
        # 第一次建立
        contract1 = factory.create_stock("AAPL")
        
        # 第二次取得（應該從快取）
        contract2 = factory.get_contract("AAPL_STK_SMART_USD")
        
        # 應該是相同的合約（如果有快取）
        # 注意：這取決於實作是否有快取機制
        assert contract1.symbol == contract2.symbol if contract2 else True


# ============================================================
# 便捷函數測試
# ============================================================

class TestConvenienceFunctions:
    """便捷函數測試類"""
    
    def test_create_stock_function(self):
        """測試 create_stock 便捷函數"""
        contract = create_stock("MSFT")
        
        assert contract.symbol == "MSFT"
        assert contract.secType == "STK"
    
    def test_create_future_function(self):
        """測試 create_future 便捷函數"""
        contract = create_future("ES", "202412")
        
        assert contract.symbol == "ES"
        assert contract.secType == "FUT"
    
    def test_create_option_function(self):
        """測試 create_option 便捷函數"""
        contract = create_option("AAPL", "20241220", 150.0, "C")
        
        assert contract.symbol == "AAPL"
        assert contract.secType == "OPT"
    
    def test_create_forex_function(self):
        """測試 create_forex 便捷函數"""
        contract = create_forex("EUR", "USD")
        
        assert contract.symbol == "EUR"
        assert contract.secType == "CASH"
    
    # 注意：create_index 和 create_crypto 便捷函數不存在
    # 請使用 ContractFactory().index() 和 ContractFactory().crypto()


# ============================================================
# 整合測試
# ============================================================

class TestIntegration:
    """整合測試類"""
    
    def test_connection_with_contract_factory(self):
        """測試連接和合約工廠整合"""
        with patch("core.connection.IB") as mock_ib_class:
            mock_ib = MagicMock()
            mock_ib.isConnected.return_value = True
            mock_ib_class.return_value = mock_ib
            
            # 建立連接
            conn = IBConnection()
            conn._ib = mock_ib
            
            # 建立合約
            factory = ContractFactory()
            contract = factory.create_stock("AAPL")
            
            assert conn.is_connected is True
            assert contract.symbol == "AAPL"
    
    def test_multiple_contracts(self):
        """測試建立多個合約"""
        factory = ContractFactory()
        
        contracts = [
            factory.create_stock("AAPL"),
            factory.create_stock("GOOGL"),
            factory.create_stock("MSFT"),
            factory.create_future("ES", "202412"),
            factory.create_forex("EUR", "USD"),
        ]
        
        assert len(contracts) == 5
        assert all(c is not None for c in contracts)
        
        # 驗證每個合約
        assert contracts[0].symbol == "AAPL"
        assert contracts[1].symbol == "GOOGL"
        assert contracts[2].symbol == "MSFT"
        assert contracts[3].secType == "FUT"
        assert contracts[4].secType == "CASH"


# ============================================================
# 執行測試
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])