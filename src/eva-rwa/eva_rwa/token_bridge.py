import json
import os
import logging

logger = logging.getLogger(__name__)

RWA_FILE = "rwa_portfolio.json"

class TokenBridge:
    """Pont vers les actifs réels tokenisés (RealT, Centrifuge)"""
    
    def __init__(self):
        self.portfolio = []
        self._load_portfolio()

    def _save_portfolio(self):
        """Sauvegarde le portfolio sur disque"""
        try:
            with open(RWA_FILE, "w") as f:
                json.dump(self.portfolio, f, indent=4)
            logger.info("💾 Portfolio RWA sauvegardé")
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde RWA: {e}")

    def _load_portfolio(self):
        """Charge le portfolio depuis le disque"""
        if os.path.exists(RWA_FILE):
            try:
                with open(RWA_FILE, "r") as f:
                    self.portfolio = json.load(f)
                logger.info("📂 Portfolio RWA chargé")
            except Exception as e:
                logger.error(f"❌ Erreur chargement RWA: {e}")
        else:
            # Portfolio par défaut si premier lancement
            self.portfolio = [
                {"id": "REALT-FLORIDA-102", "type": "Real Estate", "valuation": 1500.0, "yield": 0.091},
                {"id": "CENTRIFUGE-CFG-1", "type": "DeFi Credit", "valuation": 500.0, "yield": 0.12}
            ]
            self._save_portfolio()

    def get_portfolio(self):
        return {
            "assets": self.portfolio,
            "total_valuation": sum(a["valuation"] for a in self.portfolio),
            "currency": "USD"
        }
