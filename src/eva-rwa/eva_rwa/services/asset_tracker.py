"""
Asset Tracker — Suivi et Analyse du Portefeuille RWA.

Fournit des analyses consolidées du portefeuille :
- Répartition par catégorie d'actif.
- Calcul de rendement pondéré.
- Identification des actifs sous-performants.
- Rapport complet pour le dashboard.
"""

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

CATEGORY_LABELS = {
    "real_estate": "🏠 Immobilier",
    "energy": "⚡ Énergie",
    "defi": "🔗 DeFi",
    "land": "🌍 Foncier",
    "industrial": "🏭 Industriel",
    "other": "📦 Autre",
}


class AssetTracker:
    """Analyseur et tracker du portefeuille d'actifs réels."""

    def __init__(self, bridge):
        self.bridge = bridge
        logger.info("📊 Asset Tracker initialisé")

    def get_category_breakdown(self) -> dict[str, Any]:
        """Répartition du portefeuille par catégorie d'actif."""
        portfolio = self.bridge.get_portfolio()
        categories: dict[str, dict[str, Any]] = {}

        for asset in portfolio["assets"]:
            cat = asset.get("category", "other")
            if cat not in categories:
                categories[cat] = {
                    "label": CATEGORY_LABELS.get(cat, cat),
                    "count": 0,
                    "total_valuation": 0.0,
                    "avg_yield": 0.0,
                    "assets": [],
                }
            categories[cat]["count"] += 1
            categories[cat]["total_valuation"] += asset.get("valuation", 0)
            categories[cat]["assets"].append(asset.get("id"))

        # Calcul des rendements moyens par catégorie
        for cat_key, cat_data in categories.items():
            cat_assets = [
                a for a in portfolio["assets"]
                if a.get("category") == cat_key
            ]
            if cat_assets:
                yields = [a.get("annual_yield", 0) for a in cat_assets]
                cat_data["avg_yield"] = round(sum(yields) / len(yields), 4)
            cat_data["total_valuation"] = round(cat_data["total_valuation"], 2)

            # Part du portefeuille
            total = portfolio["total_valuation"]
            if total > 0:
                cat_data["share"] = round(cat_data["total_valuation"] / total, 4)
            else:
                cat_data["share"] = 0.0

        return {
            "categories": categories,
            "total_categories": len(categories),
            "portfolio_total": portfolio["total_valuation"],
        }

    def get_underperformers(self, threshold: float = 0.05) -> list[dict[str, Any]]:
        """Identifie les actifs avec un rendement inférieur au seuil."""
        portfolio = self.bridge.get_portfolio()
        underperformers = []

        for asset in portfolio["assets"]:
            annual_yield = asset.get("annual_yield", 0)
            if annual_yield < threshold:
                underperformers.append({
                    "id": asset.get("id"),
                    "name": asset.get("name"),
                    "yield": annual_yield,
                    "threshold": threshold,
                    "recommendation": "SELL" if annual_yield < threshold / 2 else "MONITOR",
                })

        return underperformers

    def get_full_report(self) -> dict[str, Any]:
        """Rapport consolidé pour le dashboard et l'API /portfolio."""
        portfolio = self.bridge.get_portfolio()
        breakdown = self.get_category_breakdown()

        # Déterminer la phase depuis le Sovereign Fund (approximation)
        total = portfolio["total_valuation"]
        if total < 50_000:
            phase = "Phase 1 — Indépendance Énergétique"
            phase_target = 50_000
        elif total < 500_000:
            phase = "Phase 2 — Infrastructure Industrielle"
            phase_target = 500_000
        else:
            phase = "Phase 3 — Diplomatie Financière"
            phase_target = 5_000_000

        phase_progress = min(round(total / phase_target, 4), 1.0) if phase_target > 0 else 1.0

        return {
            "total_assets": portfolio["asset_count"],
            "total_valuation": portfolio["total_valuation"],
            "weighted_yield": portfolio["weighted_yield"],
            "by_category": breakdown["categories"],
            "phase": phase,
            "phase_progress": phase_progress,
            "assets": portfolio["assets"],
            "currency": "EUR",
            "generated_at": datetime.now().isoformat(),
        }
