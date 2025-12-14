"""
Contracts 模組 - 合約工廠

提供建立和驗證 IB 合約的工具
"""

import logging
from typing import Optional, List, Union
from datetime import datetime

from ib_insync import (
    Contract,
    Stock,
    Future,
    Option,
    Forex,
    Index,
    CFD,
    ContFuture,
    IB,
)

from config.symbols import SymbolConfig, SecurityType, OptionRight


# 設定 logger
logger = logging.getLogger(__name__)


class ContractFactory:
    """
    合約工廠
    
    從 SymbolConfig 建立 IB 合約物件
    
    使用方式:
        factory = ContractFactory()
        
        # 從 SymbolConfig 建立
        config = create_stock("AAPL")
        contract = factory.create(config)
        
        # 使用便捷方法
        contract = factory.stock("AAPL")
        contract = factory.future("ES", "202412")
    """
    
    def __init__(self):
        """初始化合約工廠"""
        logger.debug("ContractFactory 初始化完成")
    
    # ========== 主要建立方法 ==========
    
    def create(self, config: SymbolConfig) -> Contract:
        """
        從 SymbolConfig 建立合約
        
        Args:
            config: 交易標的配置
            
        Returns:
            IB Contract 物件
            
        Raises:
            ValueError: 不支援的證券類型
        """
        builders = {
            SecurityType.STOCK: self._create_stock,
            SecurityType.FUTURE: self._create_future,
            SecurityType.OPTION: self._create_option,
            SecurityType.FOREX: self._create_forex,
            SecurityType.INDEX: self._create_index,
            SecurityType.CFD: self._create_cfd,
        }
        
        builder = builders.get(config.security_type)
        if builder is None:
            raise ValueError(f"不支援的證券類型: {config.security_type}")
        
        contract = builder(config)
        logger.debug(f"建立合約: {config.symbol} ({config.security_type.name})")
        
        return contract
    
    def create_many(self, configs: List[SymbolConfig]) -> List[Contract]:
        """
        批量建立合約
        
        Args:
            configs: 交易標的配置列表
            
        Returns:
            IB Contract 物件列表
        """
        return [self.create(config) for config in configs]
    
    # ========== 私有建立方法 ==========
    
    def _create_stock(self, config: SymbolConfig) -> Stock:
        """建立股票合約"""
        contract = Stock(
            symbol=config.symbol,
            exchange=config.exchange,
            currency=config.currency,
        )
        
        if config.primary_exchange:
            contract.primaryExchange = config.primary_exchange
        
        return contract
    
    def _create_future(self, config: SymbolConfig) -> Union[Future, ContFuture]:
        """建立期貨合約"""
        # 如果沒有指定到期日，使用連續合約
        if not config.expiry:
            contract = ContFuture(
                symbol=config.symbol,
                exchange=config.exchange,
                currency=config.currency,
            )
        else:
            contract = Future(
                symbol=config.symbol,
                exchange=config.exchange,
                currency=config.currency,
                lastTradeDateOrContractMonth=config.expiry,
            )
        
        if config.multiplier:
            contract.multiplier = config.multiplier
        
        if config.local_symbol:
            contract.localSymbol = config.local_symbol
        
        if config.trading_class:
            contract.tradingClass = config.trading_class
        
        return contract
    
    def _create_option(self, config: SymbolConfig) -> Option:
        """建立期權合約"""
        if not all([config.expiry, config.strike, config.right]):
            raise ValueError("期權合約需要 expiry, strike, right")
        
        # 轉換期權類型
        right = "C" if config.right == OptionRight.CALL else "P"
        
        contract = Option(
            symbol=config.symbol,
            exchange=config.exchange,
            currency=config.currency,
            lastTradeDateOrContractMonth=config.expiry,
            strike=config.strike,
            right=right,
        )
        
        if config.multiplier:
            contract.multiplier = config.multiplier
        
        if config.trading_class:
            contract.tradingClass = config.trading_class
        
        return contract
    
    def _create_forex(self, config: SymbolConfig) -> Forex:
        """建立外匯合約"""
        contract = Forex(
            pair=f"{config.symbol}{config.currency}",
            exchange=config.exchange,
        )
        
        return contract
    
    def _create_index(self, config: SymbolConfig) -> Index:
        """建立指數合約"""
        contract = Index(
            symbol=config.symbol,
            exchange=config.exchange,
            currency=config.currency,
        )
        
        return contract
    
    def _create_cfd(self, config: SymbolConfig) -> CFD:
        """建立差價合約"""
        contract = CFD(
            symbol=config.symbol,
            exchange=config.exchange,
            currency=config.currency,
        )
        
        return contract
    
    # ========== 便捷方法 ==========
    
    def stock(
        self,
        symbol: str,
        exchange: str = "SMART",
        currency: str = "USD",
        primary_exchange: Optional[str] = None,
    ) -> Stock:
        """
        建立股票合約的便捷方法
        
        Args:
            symbol: 股票代碼
            exchange: 交易所 (預設 SMART)
            currency: 貨幣 (預設 USD)
            primary_exchange: 主要交易所
            
        Returns:
            Stock 合約
            
        Example:
            >>> factory = ContractFactory()
            >>> aapl = factory.stock("AAPL")
            >>> tsm = factory.stock("TSM", primary_exchange="NYSE")
        """
        contract = Stock(
            symbol=symbol,
            exchange=exchange,
            currency=currency,
        )
        
        if primary_exchange:
            contract.primaryExchange = primary_exchange
        
        logger.debug(f"建立股票合約: {symbol}")
        return contract
    
    def future(
        self,
        symbol: str,
        expiry: Optional[str] = None,
        exchange: str = "CME",
        currency: str = "USD",
        multiplier: Optional[str] = None,
        local_symbol: Optional[str] = None,
    ) -> Union[Future, ContFuture]:
        """
        建立期貨合約的便捷方法
        
        Args:
            symbol: 期貨代碼
            expiry: 到期月份 (YYYYMM) 或日期 (YYYYMMDD)，None 為連續合約
            exchange: 交易所
            currency: 貨幣
            multiplier: 合約乘數
            local_symbol: 本地代碼
            
        Returns:
            Future 或 ContFuture 合約
            
        Example:
            >>> factory = ContractFactory()
            >>> es = factory.future("ES", "202412")
            >>> es_cont = factory.future("ES")  # 連續合約
        """
        if not expiry:
            contract = ContFuture(
                symbol=symbol,
                exchange=exchange,
                currency=currency,
            )
        else:
            contract = Future(
                symbol=symbol,
                exchange=exchange,
                currency=currency,
                lastTradeDateOrContractMonth=expiry,
            )
        
        if multiplier:
            contract.multiplier = multiplier
        
        if local_symbol:
            contract.localSymbol = local_symbol
        
        logger.debug(f"建立期貨合約: {symbol} (expiry={expiry})")
        return contract
    
    def option(
        self,
        symbol: str,
        expiry: str,
        strike: float,
        right: str,
        exchange: str = "SMART",
        currency: str = "USD",
        multiplier: str = "100",
    ) -> Option:
        """
        建立期權合約的便捷方法
        
        Args:
            symbol: 標的代碼
            expiry: 到期日 (YYYYMMDD)
            strike: 行權價
            right: "C" (Call) 或 "P" (Put)
            exchange: 交易所
            currency: 貨幣
            multiplier: 合約乘數
            
        Returns:
            Option 合約
            
        Example:
            >>> factory = ContractFactory()
            >>> call = factory.option("AAPL", "20241220", 200.0, "C")
            >>> put = factory.option("SPY", "20241220", 450.0, "P")
        """
        contract = Option(
            symbol=symbol,
            exchange=exchange,
            currency=currency,
            lastTradeDateOrContractMonth=expiry,
            strike=strike,
            right=right.upper(),
            multiplier=multiplier,
        )
        
        logger.debug(f"建立期權合約: {symbol} {strike} {right} @ {expiry}")
        return contract
    
    def forex(
        self,
        base: str,
        quote: str = "USD",
        exchange: str = "IDEALPRO",
    ) -> Forex:
        """
        建立外匯合約的便捷方法
        
        Args:
            base: 基準貨幣 (如 EUR)
            quote: 報價貨幣 (預設 USD)
            exchange: 交易所 (預設 IDEALPRO)
            
        Returns:
            Forex 合約
            
        Example:
            >>> factory = ContractFactory()
            >>> eurusd = factory.forex("EUR")
            >>> gbpusd = factory.forex("GBP", "USD")
        """
        contract = Forex(
            pair=f"{base}{quote}",
            exchange=exchange,
        )
        
        logger.debug(f"建立外匯合約: {base}/{quote}")
        return contract
    
    def index(
        self,
        symbol: str,
        exchange: str = "CBOE",
        currency: str = "USD",
    ) -> Index:
        """
        建立指數合約的便捷方法
        
        Args:
            symbol: 指數代碼
            exchange: 交易所
            currency: 貨幣
            
        Returns:
            Index 合約
            
        Example:
            >>> factory = ContractFactory()
            >>> spx = factory.index("SPX")
            >>> vix = factory.index("VIX")
        """
        contract = Index(
            symbol=symbol,
            exchange=exchange,
            currency=currency,
        )
        
        logger.debug(f"建立指數合約: {symbol}")
        return contract
    
    def cfd(
        self,
        symbol: str,
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> CFD:
        """
        建立差價合約的便捷方法
        
        Args:
            symbol: 代碼
            exchange: 交易所
            currency: 貨幣
            
        Returns:
            CFD 合約
        """
        contract = CFD(
            symbol=symbol,
            exchange=exchange,
            currency=currency,
        )
        
        logger.debug(f"建立 CFD 合約: {symbol}")
        return contract


# ============================================================
# 全局工廠實例
# ============================================================

_factory: Optional[ContractFactory] = None


def get_contract_factory() -> ContractFactory:
    """
    取得全局 ContractFactory 實例（單例模式）
    
    Returns:
        ContractFactory 實例
    """
    global _factory
    
    if _factory is None:
        _factory = ContractFactory()
    
    return _factory


# ============================================================
# 驗證函數
# ============================================================

def qualify_contract(
    ib: IB,
    contract: Contract,
) -> Optional[Contract]:
    """
    驗證合約（同步）
    
    向 IB 查詢合約詳情，填充完整資訊
    
    Args:
        ib: IB 連接實例
        contract: 要驗證的合約
        
    Returns:
        驗證後的合約，失敗返回 None
    """
    try:
        qualified = ib.qualifyContracts(contract)
        if qualified:
            logger.debug(f"合約驗證成功: {contract.symbol}")
            return qualified[0]
        else:
            logger.warning(f"合約驗證失敗: {contract.symbol}")
            return None
    except Exception as e:
        logger.error(f"合約驗證錯誤 ({contract.symbol}): {e}")
        return None


async def qualify_contract_async(
    ib: IB,
    contract: Contract,
) -> Optional[Contract]:
    """
    驗證合約（異步）
    
    向 IB 查詢合約詳情，填充完整資訊
    
    Args:
        ib: IB 連接實例
        contract: 要驗證的合約
        
    Returns:
        驗證後的合約，失敗返回 None
    """
    try:
        qualified = await ib.qualifyContractsAsync(contract)
        if qualified:
            logger.debug(f"合約驗證成功: {contract.symbol}")
            return qualified[0]
        else:
            logger.warning(f"合約驗證失敗: {contract.symbol}")
            return None
    except Exception as e:
        logger.error(f"合約驗證錯誤 ({contract.symbol}): {e}")
        return None


def qualify_contracts(
    ib: IB,
    contracts: List[Contract],
) -> List[Contract]:
    """
    批量驗證合約（同步）
    
    Args:
        ib: IB 連接實例
        contracts: 要驗證的合約列表
        
    Returns:
        驗證成功的合約列表
    """
    try:
        qualified = ib.qualifyContracts(*contracts)
        logger.debug(f"批量驗證合約: {len(qualified)}/{len(contracts)} 成功")
        return qualified
    except Exception as e:
        logger.error(f"批量合約驗證錯誤: {e}")
        return []


async def qualify_contracts_async(
    ib: IB,
    contracts: List[Contract],
) -> List[Contract]:
    """
    批量驗證合約（異步）
    
    Args:
        ib: IB 連接實例
        contracts: 要驗證的合約列表
        
    Returns:
        驗證成功的合約列表
    """
    try:
        qualified = await ib.qualifyContractsAsync(*contracts)
        logger.debug(f"批量驗證合約: {len(qualified)}/{len(contracts)} 成功")
        return qualified
    except Exception as e:
        logger.error(f"批量合約驗證錯誤: {e}")
        return []


# ============================================================
# 便捷函數
# ============================================================

def create_contract(config: SymbolConfig) -> Contract:
    """
    從 SymbolConfig 建立合約的便捷函數
    
    Args:
        config: 交易標的配置
        
    Returns:
        IB Contract 物件
    """
    factory = get_contract_factory()
    return factory.create(config)


def create_stock_contract(
    symbol: str,
    exchange: str = "SMART",
    currency: str = "USD",
    primary_exchange: Optional[str] = None,
) -> Stock:
    """建立股票合約的便捷函數"""
    factory = get_contract_factory()
    return factory.stock(symbol, exchange, currency, primary_exchange)


def create_future_contract(
    symbol: str,
    expiry: Optional[str] = None,
    exchange: str = "CME",
    currency: str = "USD",
    multiplier: Optional[str] = None,
) -> Union[Future, ContFuture]:
    """建立期貨合約的便捷函數"""
    factory = get_contract_factory()
    return factory.future(symbol, expiry, exchange, currency, multiplier)


def create_option_contract(
    symbol: str,
    expiry: str,
    strike: float,
    right: str,
    exchange: str = "SMART",
    currency: str = "USD",
) -> Option:
    """建立期權合約的便捷函數"""
    factory = get_contract_factory()
    return factory.option(symbol, expiry, strike, right, exchange, currency)


def create_forex_contract(
    base: str,
    quote: str = "USD",
    exchange: str = "IDEALPRO",
) -> Forex:
    """建立外匯合約的便捷函數"""
    factory = get_contract_factory()
    return factory.forex(base, quote, exchange)