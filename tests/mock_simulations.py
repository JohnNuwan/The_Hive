import time
from datetime import datetime
import sys
import os

# Ajout des chemins pour les imports
sys.path.append(os.path.abspath("src/eva-compliance"))
sys.path.append(os.path.abspath("src/eva-substrate"))

try:
    from eva_compliance.tax_manager import TaxManager
    from eva_substrate.scheduler import Scheduler
except ImportError:
    print("⚠️ Attention: Modules métier non trouvés. Assurez-vous que les fichiers existent.")

class HiveMockSimulator:
    """
    SIMULATEUR DE VIE (Genesis Mock)
    --------------------------------
    Valide les logiques métier critiques sans infrastructure réelle.
    """

    def __init__(self):
        self.tax_man = TaxManager()
        self.scheduler = Scheduler()

    def simulate_day_night_cycle(self):
        print("\n--- SIMULATION CYCLE CIRCADIEN ---")
        # On force l'heure pour tester (Mocking datetime)
        # Simulation JOUR (14h)
        print("Test 1: Heure forcée à 14:00 (Jour)")
        status = self.scheduler.heartbeat() # Le scheduler actuel utilise datetime.now()
        # Note: Pour un vrai test unitaire on mockerait datetime.
        print(f"Mode Actuel: {self.scheduler.current_state}")

    def simulate_financial_flow(self, transactions):
        print("\n--- SIMULATION FLUX FISCAL ---")
        for amount in transactions:
            print(f"Nouveau gain détecté: {amount}€")
            # Le TaxManager attend un dictionnaire avec la clé 'profit'
            trade_data = {"profit": amount, "ticket_id": f"MOCK_{int(time.time())}"}
            provision = self.tax_man.process_trade_result(trade_data)
            print(f"Action TaxManager: {provision['status']} | Bloqué: {provision['tax_amount']}€")
        
        status = self.tax_man.get_escrow_status()
        print(f"SOLDE TOTAL ESCROW FISCAL: {status['total_blocked']}€")

if __name__ == "__main__":
    print("LANCEMENT DE LA SIMULATION (LOGIC ONLY)")
    sim = HiveMockSimulator()
    
    # 1. Simulation Finance (1000€ de gains cumulés)
    sim.simulate_financial_flow([100.0, 500.0, 250.0, 150.0])
    
    # 2. Simulation Rythme
    sim.simulate_day_night_cycle()
    
    print("\nVALIDATION LOGIQUE TERMINEE.")
