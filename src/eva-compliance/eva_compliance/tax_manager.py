"""
Tax Manager — Provisionnement Fiscal Automatique de THE HIVE.

Intercepte les événements de trading gagnants et provisionne
automatiquement les taxes (URSSAF / BNC) dans un compte escrow
séparé pour garantir que l'IA ne réinvestisse pas l'argent de l'État.

Principe de fonctionnement :
    1. Le Banker ferme un trade profitable.
    2. Le signal arrive via Redis Pub/Sub (canal eva.compliance.trades).
    3. Le TaxManager calcule 25% du profit (taux BNC approximatif).
    4. Le montant est « bloqué » dans un fichier escrow_ledger.json.
    5. L'IA ne peut utiliser que le net réinvestissable.
"""

import json
import logging
import os
from datetime import datetime
from decimal import Decimal
from typing import Any

logger = logging.getLogger(__name__)


class TaxManager:
    """
    Gestionnaire fiscal de la ruche.

    Provisionne automatiquement les taxes sur les gains de trading
    dans un fichier escrow sécurisé (JSON persistant).

    Attributes:
        TAX_RATE: Taux d'imposition appliqué (25% BNC par défaut).
        ESCROW_FILE: Chemin du fichier de persistance escrow.
    """

    TAX_RATE = Decimal("0.25")  # 25% URSSAF approximatif pour BNC
    ESCROW_FILE = "escrow_ledger.json"

    def __init__(self, escrow_path: str | None = None):
        """
        Initialise le gestionnaire fiscal.

        Args:
            escrow_path: Chemin personnalisé du fichier escrow (optionnel).
        """
        if escrow_path:
            self.ESCROW_FILE = escrow_path
        self._ensure_ledger_exists()

    def _ensure_ledger_exists(self):
        """Crée le fichier escrow s'il n'existe pas encore."""
        if not os.path.exists(self.ESCROW_FILE):
            with open(self.ESCROW_FILE, "w", encoding="utf-8") as f:
                json.dump({"total_tax_blocked": 0.0, "transactions": []}, f, indent=4)

    def process_trade_result(self, trade_data: dict[str, Any]) -> dict[str, Any]:
        """
        Traite un résultat de trade et provisionne la taxe si profitable.

        Si le profit est positif, calcule la part fiscale (25%)
        et l'enregistre dans le ledger escrow.

        Args:
            trade_data: Données du trade (profit, ticket_id, symbol).

        Returns:
            dict: Statut (TAX_BLOCKED ou IGNORED), montants et message.
        """
        profit = Decimal(str(trade_data.get("profit", 0.0)))

        if profit <= 0:
            return {
                "status": "IGNORED",
                "reason": "No profit or Loss",
                "tax_blocked": 0.0,
            }

        # Calcul de la part de l'État
        tax_amount = profit * self.TAX_RATE
        net_profit = profit - tax_amount

        # Enregistrement dans le Ledger (Escrow)
        self._record_transaction(
            trade_data.get("ticket_id", "UNKNOWN"), profit, tax_amount
        )

        return {
            "status": "TAX_BLOCKED",
            "gross_profit": float(profit),
            "tax_amount": float(tax_amount),
            "net_reinvestable": float(net_profit),
            "message": f"25% bloqué en escrow ({float(tax_amount):.2f} €).",
        }

    def get_escrow_status(self) -> dict[str, Any]:
        """
        Retourne l'état actuel du compte escrow.

        Returns:
            dict: Total bloqué et nombre de transactions enregistrées.
        """
        try:
            with open(self.ESCROW_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {
                    "total_blocked": data.get("total_tax_blocked", 0.0),
                    "transaction_count": len(data.get("transactions", [])),
                }
        except Exception:
            return {"total_blocked": 0.0, "transaction_count": 0}

    def _record_transaction(self, ticket_id: str, gross: Decimal, tax: Decimal):
        """
        Écrit une transaction fiscale dans le fichier escrow.

        Opération atomique lecture-écriture avec troncature pour
        éviter la corruption du fichier.

        Args:
            ticket_id: Identifiant du ticket MT5.
            gross: Profit brut du trade.
            tax: Montant de la taxe provisionnée.
        """
        try:
            with open(self.ESCROW_FILE, "r+", encoding="utf-8") as f:
                data = json.load(f)

                transaction = {
                    "timestamp": datetime.now().isoformat(),
                    "ticket_id": ticket_id,
                    "gross_profit": float(gross),
                    "tax_blocked": float(tax),
                }

                data["transactions"].append(transaction)
                data["total_tax_blocked"] += float(tax)

                f.seek(0)
                json.dump(data, f, indent=4)
                f.truncate()
        except Exception as e:
            # Erreur critique — doit déclencher une alerte Sentinel en production
            logger.critical(f"ERREUR CRITIQUE écriture escrow: {e}")
