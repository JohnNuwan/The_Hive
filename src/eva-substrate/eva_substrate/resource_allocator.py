class ResourceAllocator:
    """Gestionnaire d'allocation des Google Coral TPUs"""
    
    def __init__(self):
        self.tpu_count = 8 # 4x Dual Edge TPU
        self.allocations = {
            "sentinel": 4, # Sécurité par défaut
            "wraith": 4    # Vision par défaut
        }

    def set_profile(self, profile: str):
        if profile == "WAR_MODE":
            self.allocations = {"sentinel": 8, "wraith": 0}
        elif profile == "SIGHT_MODE":
            self.allocations = {"sentinel": 2, "wraith": 6}
        else:
            self.allocations = {"sentinel": 4, "wraith": 4}
            
        return {
            "profile": profile,
            "allocations": self.allocations,
            "status": "TPU_RECONFIGURED"
        }
