import psutil
import random

class EnergyMonitor:
    """Surveillance de la consommation énergétique (Mock IPMI)"""
    
    def __init__(self):
        self.base_idle_watts = 120 # EPYC idle
        self.max_load_watts = 650  # EPYC + 3090 Full Load

    def get_current_consumption(self):
        # Simulation basée sur l'usage CPU réel si possible
        cpu_usage = psutil.cpu_percent()
        estimated_watts = self.base_idle_watts + (self.max_load_watts - self.base_idle_watts) * (cpu_usage / 100)
        
        return {
            "watts": round(estimated_watts, 2),
            "cpu_usage": cpu_usage,
            "unit": "W",
            "source": "IPMI_MOCK"
        }
