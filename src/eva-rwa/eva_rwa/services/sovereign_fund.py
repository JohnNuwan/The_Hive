"""
Sovereign Fund — Stratégie d'Investissement Long Terme.

Gère la vision stratégique du Sovereign Fund de THE HIVE :
- Phase 1 : Indépendance Énergétique (solaire, batterie, terrain).
- Phase 2 : Infrastructure Industrielle (usines, robotisation).
- Phase 3 : Diplomatie Financière (dette souveraine, influence).

Fournit des recommandations d'acquisition alignées avec la phase courante.
"""

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# PHASES STRATÉGIQUES
# ═══════════════════════════════════════════════════════════════════════════════

PHASES = [
    {
        "id": 1,
        "name": "Indépendance Énergétique",
        "emoji": "⚡",
        "description": "Acquérir des terrains et installations de production d'énergie renouvelable.",
        "target_valuation": 50_000,
        "priority_categories": ["energy", "land"],
        "milestones": [
            "Premier terrain acquis",
            "Installation solaire opérationnelle",
            "Autonomie énergétique partielle",
            "Excédent revendu au réseau",
        ],
    },
    {
        "id": 2,
        "name": "Infrastructure Industrielle",
        "emoji": "🏭",
        "description": "Rachat et robotisation d'usines en difficulté.",
        "target_valuation": 500_000,
        "priority_categories": ["industrial", "real_estate"],
        "milestones": [
            "Première usine acquise",
            "Robotisation partielle",
            "Production autonome",
            "Bénéfice net positif",
        ],
    },
    {
        "id": 3,
        "name": "Diplomatie Financière",
        "emoji": "🌍",
        "description": "Acquisition de dette gouvernementale pour influence géopolitique.",
        "target_valuation": 5_000_000,
        "priority_categories": ["defi", "real_estate", "industrial"],
        "milestones": [
            "Première obligation souveraine",
            "Portefeuille diversifié internationalement",
            "Influence au conseil d'administration",
            "Souveraineté financière atteinte",
        ],
    },
]


class SovereignFund:
    """Gestionnaire de la stratégie Sovereign Fund."""

    def __init__(self, bridge):
        self.bridge = bridge
        logger.info("👑 Sovereign Fund initialisé")

    def _get_current_phase(self) -> dict[str, Any]:
        """Détermine la phase stratégique actuelle basée sur la valorisation."""
        portfolio = self.bridge.get_portfolio()
        total = portfolio["total_valuation"]

        for phase in PHASES:
            if total < phase["target_valuation"]:
                return phase

        # Si au-delà de la Phase 3
        return PHASES[-1]

    def _get_phase_progress(self) -> float:
        """Calcule la progression vers l'objectif de la phase actuelle (0.0 à 1.0)."""
        portfolio = self.bridge.get_portfolio()
        total = portfolio["total_valuation"]
        phase = self._get_current_phase()
        target = phase["target_valuation"]
        return min(round(total / target, 4), 1.0) if target > 0 else 1.0

    def check_alignment(self, category: str) -> dict[str, Any]:
        """
        Vérifie si une acquisition est alignée avec la stratégie actuelle.

        Returns:
            dict avec 'aligned' (bool) et 'reason' (str).
        """
        phase = self._get_current_phase()
        priority = phase["priority_categories"]

        if category in priority:
            return {
                "aligned": True,
                "phase": phase["name"],
                "reason": f"✅ '{category}' est une priorité de la phase {phase['id']} ({phase['name']})",
            }
        return {
            "aligned": False,
            "phase": phase["name"],
            "reason": f"⚠️ '{category}' n'est pas prioritaire en phase {phase['id']}. Priorités : {priority}",
        }

    def get_strategy_report(self) -> dict[str, Any]:
        """Rapport complet de la stratégie Sovereign Fund."""
        phase = self._get_current_phase()
        progress = self._get_phase_progress()
        portfolio = self.bridge.get_portfolio()

        return {
            "current_phase": {
                "id": phase["id"],
                "name": phase["name"],
                "emoji": phase["emoji"],
                "description": phase["description"],
                "target_valuation": phase["target_valuation"],
                "priority_categories": phase["priority_categories"],
                "milestones": phase["milestones"],
            },
            "progress": progress,
            "portfolio_valuation": portfolio["total_valuation"],
            "remaining_to_target": max(0, phase["target_valuation"] - portfolio["total_valuation"]),
            "all_phases": [
                {"id": p["id"], "name": p["name"], "emoji": p["emoji"], "target": p["target_valuation"]}
                for p in PHASES
            ],
            "generated_at": datetime.now().isoformat(),
        }

    def get_recommendations(self) -> dict[str, Any]:
        """Génère des recommandations d'acquisition pour la phase actuelle."""
        phase = self._get_current_phase()
        portfolio = self.bridge.get_portfolio()
        categories = {}
        for asset in portfolio["assets"]:
            cat = asset.get("category", "other")
            categories[cat] = categories.get(cat, 0) + asset.get("valuation", 0)

        recommendations = []
        for priority_cat in phase["priority_categories"]:
            current_val = categories.get(priority_cat, 0)
            target_share = phase["target_valuation"] / len(phase["priority_categories"])

            if current_val < target_share:
                recommendations.append({
                    "priority": "HIGH" if current_val == 0 else "MEDIUM",
                    "category": priority_cat,
                    "reason": f"Sous-représenté : {current_val:.0f}€ vs objectif ~{target_share:.0f}€",
                    "target_amount": round(target_share - current_val, 2),
                    "expected_yield": 0.08 if priority_cat in ("energy", "land") else 0.10,
                })

        if not recommendations:
            recommendations.append({
                "priority": "LOW",
                "category": "diversification",
                "reason": "Portefeuille bien équilibré. Envisager une diversification géographique.",
                "target_amount": 0,
                "expected_yield": 0.07,
            })

        return {
            "phase": phase["name"],
            "recommendations": recommendations,
            "generated_at": datetime.now().isoformat(),
        }
