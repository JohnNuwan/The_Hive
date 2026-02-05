import os
import json
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any

class DebtManager:
    """
    Gère la 'Dette de Naissance' de la Ruche (-2 500 €).
    S'assure que les premiers profits sont alloués au remboursement.
    """
    
    def __init__(self, data_path: str = "data/financial/debt_tracker.json"):
        self.data_path = data_path
        self.initial_debt = Decimal("2500.00")
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if os.path.exists(self.data_path):
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["current_debt"] = Decimal(str(data["current_debt"]))
                return data
        return {
            "initial_debt": float(self.initial_debt),
            "current_debt": self.initial_debt,
            "last_update": datetime.now().isoformat(),
            "repayments": []
        }

    def _save_state(self):
        save_data = self.state.copy()
        save_data["current_debt"] = float(self.state["current_debt"])
        save_data["last_update"] = datetime.now().isoformat()
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=4)

    def register_repayment(self, amount: Decimal, source: str = "Trading"):
        """Enregistre un remboursement et réduit la dette."""
        if amount <= 0:
            return
        
        self.state["current_debt"] -= amount
        if self.state["current_debt"] < 0:
            self.state["current_debt"] = Decimal("0.00")
            
        self.state["repayments"].append({
            "amount": float(amount),
            "date": datetime.now().isoformat(),
            "source": source
        })
        self._save_state()
        
    def get_status(self) -> Dict[str, Any]:
        """Affiche l'état actuel de la dette."""
        is_cleared = self.state["current_debt"] <= 0
        return {
            "is_cleared": is_cleared,
            "remaining_debt": float(self.state["current_debt"]),
            "progress_percent": float(((self.initial_debt - self.state["current_debt"]) / self.initial_debt) * 100),
            "total_repaid": float(self.initial_debt - self.state["current_debt"])
        }

if __name__ == "__main__":
    # Test simple
    manager = DebtManager()
    status = manager.get_status()
    print("--- ETAT DE LA DETTE INITIALE ---")
    print(f"Dette restante : {status['remaining_debt']} EUR")
    print(f"Progression : {status['progress_percent']:.2f} %")
    if status['is_cleared']:
        print("[OK] DETTE REMBOURSEE. ACTIVATION LOI 5 : ABONDANCE.")
    else:
        print("[ATTENTE] LOI 5 EN ATTENTE DE REMBOURSEMENT INTEGRAL.")
