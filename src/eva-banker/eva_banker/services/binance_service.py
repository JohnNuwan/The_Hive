import logging
import asyncio
from decimal import Decimal
from typing import Dict, Any, Optional
import ccxt.async_support as ccxt

logger = logging.getLogger(__name__)

class BinanceService:
    """
    Service d'intégration Crypto (Binance/Kraken) via CCXT (Un-Mocked).
    Gère la connexion Spot et potentiellement Futures.
    """
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = False, exchange_id: str = "binance"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.exchange_id = exchange_id
        
        # Initialisation via la factory CCXT async
        exchange_class = getattr(ccxt, self.exchange_id)
        config = {
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
        }
        self.exchange = exchange_class(config)
        if self.testnet:
            self.exchange.set_sandbox_mode(True)

    async def initialize(self):
        """Charge les marchés (indispensable pour CCXT)"""
        try:
            await self.exchange.load_markets()
            logger.info(f"✅ Crypto Exchange '{self.exchange_id}' initialisé. (Testnet: {self.testnet})")
        except Exception as e:
            logger.error(f"Erreur d'initialisation de l'exchange {self.exchange_id}: {e}")

    async def close(self):
        """Ferme la session aiohttp sous-jacente."""
        await self.exchange.close()

    async def get_account_balances(self) -> Dict[str, Decimal]:
        """Récupère les soldes du compte Spot."""
        if not self.api_key:
            # Fallback mock pour tests si l'api key n'est pas encore fournie
            return {
                "BTC": Decimal("0.00"),
                "ETH": Decimal("0.00"),
                "USDT": Decimal("0.00") # Live empty
            }

        try:
            balance = await self.exchange.fetch_balance()
            free_balances = balance.get('free', {})
            # Ne retourne que les soldes positifs
            return {asset: Decimal(str(amount)) for asset, amount in free_balances.items() if amount > 0}
        except Exception as e:
            logger.error(f"Erreur fetch_balance ({self.exchange_id}): {e}")
            return {}

    async def place_order(self, symbol: str, action: str, volume: float) -> Dict[str, Any]:
        """Place un ordre Market d'achat ou de vente au comptant."""
        if not self.api_key:
            logger.warning(f"Crypto: Simulation d'ordre {action} sur {symbol} car pas d'API_KEY.")
            return {
                "order_id": "SIM-CRYPTO-123",
                "status": "FILLED",
                "symbol": symbol,
                "executed_qty": volume,
                "price": 0.0 # Will be mapped by risk
            }

        logger.info(f"Crypto Exchange: Placement ordre {action} {volume} sur {symbol}")
        side = 'buy' if action.upper() == 'BUY' else 'sell'
        
        # CCXT utilise habituellement la nomenclature 'BTC/USDT'
        ccxt_symbol = symbol
        if "/" not in symbol and len(symbol) > 4:
             # Heuristique basique, e.g., BTCUSDT -> BTC/USDT
             if symbol.endswith("USDT"):
                 ccxt_symbol = f"{symbol[:-4]}/USDT"
        
        try:
            order = await self.exchange.create_market_order(ccxt_symbol, side, volume)
            return {
                "order_id": order.get('id'),
                "status": order.get('status', 'OPEN').upper(),
                "symbol": symbol,
                "executed_qty": order.get('filled', volume),
                "price": order.get('average', order.get('price'))
            }
        except Exception as e:
            logger.error(f"Erreur create_order ({self.exchange_id}): {e}")
            return {"success": False, "message": str(e)}

    async def get_ticker(self, symbol: str) -> Optional[Dict[str, float]]:
        """Récupère le ticker temps réel (Bid/Ask)"""
        ccxt_symbol = symbol
        if "/" not in symbol and symbol.endswith("USDT"):
            ccxt_symbol = f"{symbol[:-4]}/USDT"

        try:
            ticker = await self.exchange.fetch_ticker(ccxt_symbol)
            return {
                "bid": ticker.get('bid'),
                "ask": ticker.get('ask'),
                "last": ticker.get('last')
            }
        except Exception as e:
            logger.error(f"Erreur fetch_ticker ({self.exchange_id}): {e}")
            return None
