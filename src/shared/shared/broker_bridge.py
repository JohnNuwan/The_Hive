import logging
from abc import ABC, abstractmethod
from decimal import Decimal
from shared.models import TradeOrder

logger = logging.getLogger(__name__)

class BrokerInterface(ABC):
    """Interface abstraite pour tous les courtiers (Forex, Crypto, Actions)."""
    @abstractmethod
    async def connect(self) -> bool: pass
    
    @abstractmethod
    async def execute_order(self, order: TradeOrder) -> bool: pass

class BinanceBridge(BrokerInterface):
    """Pont vers Binance (MOCK pour Phase 8)."""
    async def connect(self) -> bool:
        logger.info("BinanceBridge: Liaison API établie.")
        return True
    
    async def execute_order(self, order: TradeOrder) -> bool:
        logger.info(f"BinanceBridge: Ordre Crypto exécuté: {order.symbol} {order.volume}")
        return True

class MT5Bridge(BrokerInterface):
    """Pont vers MetaTrader 5."""
    def __init__(self, mt5_service):
        self.mt5 = mt5_service
        
    async def connect(self) -> bool:
        return await self.mt5.connect()
    
    async def execute_order(self, order: TradeOrder) -> bool:
        return await self.mt5.execute_order(order)

class BrokerBridge:
    """Le répartiteur Multi-Broker."""
    def __init__(self, mt5_service):
        self.mt5 = MT5Bridge(mt5_service)
        self.binance = BinanceBridge()
        
    async def route_order(self, order: TradeOrder):
        # Routage intelligent selon le symbole
        if "/" in order.symbol or any(c in order.symbol for c in ["BTC", "ETH", "USDT"]):
            return await self.binance.execute_order(order)
        return await self.mt5.execute_order(order)
