import json
import os
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any

class TaxManager:
    """
    THE CONTROLLER (Le Comptable)
    -----------------------------
    Cet agent passif intercept les événements de trading gagnants et 
    provisionne automatiquement les taxes (URSSAF) pour éviter
    que l'IA ne réinvestisse l'argent de l'État.
    """

    TAX_RATE = Decimal("0.25") # 25% URSSAF approx pour BNC
    ESCROW_FILE = "escrow_ledger.json"

    def __init__(self, escrow_path: str = None):
        if escrow_path:
            self.ESCROW_FILE = escrow_path
        self._ensure_ledger_exists()

    def _ensure_ledger_exists(self):
        if not os.path.exists(self.ESCROW_FILE):
            with open(self.ESCROW_FILE, 'w') as f:
                json.dump({"total_tax_blocked": 0.0, "transactions": []}, f, indent=4)

    def process_trade_result(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Traite un résultat de trade. Si gain > 0, bloque la taxe.
        """
        profit = Decimal(str(trade_data.get("profit", 0.0)))
        
        if profit <= 0:
            return {
                "status": "IGNORED",
                "reason": "No profit or Loss",
                "tax_blocked": 0.0
            }

        # Calcul de la part de l'État
        tax_amount = profit * self.TAX_RATE
        net_profit = profit - tax_amount

        # Enregistrement dans le Ledger (Escrow)
        self._record_transaction(trade_data.get("ticket_id", "UNKNOWN"), profit, tax_amount)

        return {
            "status": "TAX_BLOCKED",
            "gross_profit": float(profit),
            "tax_amount": float(tax_amount),
            "net_reinvestable": float(net_profit),
            "message": "Security Warning: 25% of gains locked in Escrow."
        }

    def _record_transaction(self, ticket_id: str, gross: Decimal, tax: Decimal):
        """Écrit la transaction dans le fichier JSON sécurisé"""
        try:
            with open(self.ESCROW_FILE, 'r+') as f:
                data = json.load(f)
                
                transaction = {
                    "timestamp": datetime.now().isoformat(),
                    "ticket_id": ticket_id,
                    "gross_profit": float(gross),
                    "tax_blocked": float(tax)
                }
                
                data["transactions"].append(transaction)
                data["total_tax_blocked"] += float(tax)
                
                f.seek(0)
                json.dump(data, f, indent=4)
                f.truncate()
        except Exception as e:
            print(f"CRITICAL ERROR WRITING TAX LEDGER: {e}")
            # En prod, cela devrait déclencher une alerte Sentinel
