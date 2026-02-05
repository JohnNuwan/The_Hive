import random

class IotController:
    """Contrôleur IoT via protocole MQTT pour capteurs/production solaire"""
    
    def __init__(self):
        self.broker = "localhost" # Passer en config .env plus tard

    def get_latest_data(self):
        # Simulation de télémétrie IoT (ex: Panneaux solaires)
        return {
            "solar_production_w": random.randint(0, 3500),
            "battery_level": 82,
            "external_temp": 24.5,
            "status": "CONNECTED",
            "protocol": "MQTT"
        }
