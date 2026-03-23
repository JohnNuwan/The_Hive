"""
Token Bridge — Gestionnaire de Portefeuille d'Actifs Réels.

Responsable du CRUD sur le portefeuille RWA :
- Ajout / Suppression / Mise à jour d'actifs.
- Persistance sur disque (JSON).
- Calculs de valorisation et rendement.
"""

import json
import os
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RWA_FILE = os.path.join(DATA_DIR, "rwa_portfolio.json")


class TokenBridge:
    """Pont vers les actifs réels tokenisés (RealT, Centrifuge, Foncier)."""

    def __init__(self):
        self.portfolio: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []
        os.makedirs(DATA_DIR, exist_ok=True)
        self._load_portfolio()

    # ─── PERSISTANCE ──────────────────────────────────────────────────────────

    def _save_portfolio(self):
        """Sauvegarde le portfolio sur disque."""
        try:
            with open(RWA_FILE, "w") as f:
                json.dump(
                    {"assets": self.portfolio, "updated_at": datetime.now().isoformat()},
                    f, indent=4, default=str,
                )
            logger.info(f"💾 Portfolio RWA sauvegardé ({len(self.portfolio)} actifs)")
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde RWA: {e}")

    def _load_portfolio(self):
        """Charge le portfolio depuis le disque ou crée le défaut."""
        if os.path.exists(RWA_FILE):
            try:
                with open(RWA_FILE, "r") as f:
                    data = json.load(f)
                self.portfolio = data.get("assets", data) if isinstance(data, dict) else data
                logger.info(f"📂 Portfolio RWA chargé ({len(self.portfolio)} actifs)")
                return
            except Exception as e:
                logger.error(f"❌ Erreur chargement RWA: {e}")

        # Portfolio par défaut si premier lancement
        self.portfolio = [
            {
                "id": "RWA-REALT-FL01",
                "name": "RealT — Appartement Miami Beach",
                "category": "real_estate",
                "valuation": 1500.0,
                "annual_yield": 0.091,
                "acquisition_date": "2026-01-15T00:00:00",
                "location": "Miami Beach, FL, USA",
                "tokenized": True,
                "metadata": {"platform": "RealT", "tokens": 15},
            },
            {
                "id": "RWA-CFG-001",
                "name": "Centrifuge — Pool Crédit PME",
                "category": "defi",
                "valuation": 500.0,
                "annual_yield": 0.12,
                "acquisition_date": "2026-02-01T00:00:00",
                "location": None,
                "tokenized": True,
                "metadata": {"platform": "Centrifuge", "pool": "CFG-NS2"},
            },
        ]
        self._save_portfolio()

    # ─── CRUD ─────────────────────────────────────────────────────────────────

    def add_asset(self, asset: dict[str, Any]) -> dict[str, Any]:
        """Ajoute un actif au portefeuille et sauvegarde."""
        self.portfolio.append(asset)
        self.history.append({
            "action": "ACQUIRE",
            "asset_id": asset.get("id"),
            "valuation": asset.get("valuation"),
            "timestamp": datetime.now().isoformat(),
        })
        self._save_portfolio()
        return asset

    def remove_asset(self, asset_id: str) -> bool:
        """Retire un actif par ID. Retourne True si trouvé et supprimé."""
        for i, asset in enumerate(self.portfolio):
            if asset.get("id") == asset_id:
                removed = self.portfolio.pop(i)
                self.history.append({
                    "action": "SELL",
                    "asset_id": asset_id,
                    "valuation": removed.get("valuation"),
                    "timestamp": datetime.now().isoformat(),
                })
                self._save_portfolio()
                return True
        return False

    def update_valuation(self, asset_id: str, new_valuation: float) -> bool:
        """Met à jour la valorisation d'un actif. Retourne True si trouvé."""
        for asset in self.portfolio:
            if asset.get("id") == asset_id:
                old_val = asset.get("valuation", 0)
                asset["valuation"] = new_valuation
                self.history.append({
                    "action": "REVALUE",
                    "asset_id": asset_id,
                    "old_valuation": old_val,
                    "new_valuation": new_valuation,
                    "timestamp": datetime.now().isoformat(),
                })
                self._save_portfolio()
                return True
        return False

    # ─── LECTURE ───────────────────────────────────────────────────────────────

    def get_portfolio(self) -> dict[str, Any]:
        """Retourne le portefeuille avec les totaux calculés."""
        total = sum(a.get("valuation", 0) for a in self.portfolio)
        weighted_yield = 0.0
        if total > 0:
            weighted_yield = sum(
                a.get("valuation", 0) * a.get("annual_yield", 0)
                for a in self.portfolio
            ) / total

        return {
            "assets": self.portfolio,
            "total_valuation": round(total, 2),
            "weighted_yield": round(weighted_yield, 4),
            "asset_count": len(self.portfolio),
            "currency": "EUR",
        }

    def get_history(self) -> list[dict[str, Any]]:
        """Retourne l'historique des transactions."""
        return self.history
