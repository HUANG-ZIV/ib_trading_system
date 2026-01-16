"""Event Contract 策略基類"""
import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from ib_insync import IB, Contract, Option, Trade, LimitOrder
from core.contracts import get_contract_factory

logger = logging.getLogger(__name__)

@dataclass
class EventPosition:
    contract: Contract
    quantity: int
    avg_price: float
    side: str
    entry_time: datetime = field(default_factory=datetime.now)
    
    @property
    def max_profit(self) -> float:
        return self.quantity * (1.0 - self.avg_price)
    
    @property
    def max_loss(self) -> float:
        return self.quantity * self.avg_price

@dataclass
class EventSignal:
    symbol: str
    expiry: str
    strike: float
    side: str
    action: str
    quantity: int
    limit_price: Optional[float] = None
    reason: str = ""
    edge: float = 0.0

class EventContractStrategy:
    def __init__(self, ib: IB, symbol: str, max_position: int = 100,
                 max_risk_per_trade: float = 50.0):
        self._ib = ib
        self._symbol = symbol
        self._max_position = max_position
        self._max_risk_per_trade = max_risk_per_trade
        self._factory = get_contract_factory()
        self._positions: Dict[str, EventPosition] = {}
        self._is_running = False
        logger.info(f"EventContractStrategy 初始化: {symbol}")
    
    @property
    def symbol(self) -> str:
        return self._symbol
    
    @property
    def total_position(self) -> int:
        return sum(p.quantity for p in self._positions.values())
    
    def create_contract(self, expiry: str, strike: float, is_yes: bool = True) -> Option:
        return self._factory.forecastex(self._symbol, expiry, strike, is_yes)
    
    def submit_order(self, contract: Contract, quantity: int, limit_price: float) -> Optional[Trade]:
        if quantity <= 0 or limit_price <= 0 or limit_price >= 1:
            return None
        if self.total_position + quantity > self._max_position:
            logger.warning("超過持倉限制")
            return None
        order = LimitOrder(action="BUY", totalQuantity=quantity, lmtPrice=limit_price)
        try:
            trade = self._ib.placeOrder(contract, order)
            logger.info(f"訂單: {contract.symbol} {contract.strike} x{quantity} @ {limit_price}")
            return trade
        except Exception as e:
            logger.error(f"訂單失敗: {e}")
            return None
    
    @abstractmethod
    def generate_signals(self) -> List[EventSignal]:
        raise NotImplementedError
    
    def calculate_edge(self, market_price: float, estimated_prob: float) -> float:
        return estimated_prob - market_price
    
    def start(self):
        self._is_running = True
        signals = self.generate_signals()
        for signal in signals:
            contract = self.create_contract(signal.expiry, signal.strike, signal.side == "YES")
            self.submit_order(contract, signal.quantity, signal.limit_price)
    
    def stop(self):
        self._is_running = False
