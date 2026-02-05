from datetime import datetime

class CircadianRhythm:
    """Gestion du cycle Jour/Nuit pour l'IA"""
    
    def __init__(self):
        # Heures creuses (Nuit) : 22h - 06h
        self.night_hours = range(22, 6) or range(22, 24) # Correction simple

    def get_current_mode(self):
        current_hour = datetime.now().hour
        # Mode Nuit si heure entre 22h et 06h
        is_night = current_hour >= 22 or current_hour < 6
        
        return {
            "mode": "NIGHT (Full Power)" if is_night else "DAY (Eco Mode)",
            "is_night": is_night,
            "hour": current_hour,
            "strategy": "DEEP_LEARNING" if is_night else "TRADING_PASSIVE"
        }
