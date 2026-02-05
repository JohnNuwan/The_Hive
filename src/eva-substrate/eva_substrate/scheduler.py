from datetime import datetime
from .circadian_rhythm import CircadianRhythm
from .resource_allocator import ResourceAllocator

class Scheduler:
    """
    THE KEEPER (Le Gardien)
    -----------------------
    Orchestrateur central des ressources basées sur le temps.
    Il impose le Rythme Circadien à toute la Ruche.
    """

    def __init__(self):
        self.rhythm = CircadianRhythm()
        self.allocator = ResourceAllocator()
        self.current_state = "UNKNOWN"

    def heartbeat(self):
        """
        Appelé toutes les minutes par un cron ou une boucle.
        Vérifie l'heure et ajuste les ressources.
        """
        # 1. Lire le mode temporel (Jour/Nuit)
        mode_info = self.rhythm.get_current_mode()
        is_night = mode_info["is_night"]
        
        # 2. Déterminer le profil Hardware cible
        target_profile = "DEEP_LEARNING" if is_night else "STANDARD"
        
        # 3. Appliquer les changements si nécessaire
        if target_profile != self.current_state:
            print(f"🔄 SWITCHING MODE: {self.current_state} -> {target_profile}")
            
            # Allocation TPU (Vision vs Sécurité)
            # La nuit, on réduit la vision (il fait noir) et on maximise le calcul
            tpu_mode = "WAR_MODE" if is_night else "SIGHT_MODE"
            self.allocator.set_profile(tpu_mode)
            
            self.current_state = target_profile
            
            return {
                "action": "MODE_SWITCH",
                "new_mode": target_profile,
                "details": mode_info
            }
            
        return {"action": "NO_CHANGE", "mode": self.current_state}

    def get_system_status(self):
        return {
            "rhythm": self.rhythm.get_current_mode(),
            "resources": self.allocator.allocations,
            "keeper_state": self.current_state
        }
