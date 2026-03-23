"""
IoT Controller — Contrôleur de capteurs physiques (énergie, température).

Gère la télémétrie des actifs physiques :
- Production solaire (panneaux photovoltaïques).
- Niveau batterie (stockage énergie).
- Température extérieure.
- Historique de production quotidienne.

En mode simulation, génère des données réalistes basées sur l'heure.
En production, se connectera via MQTT (Mosquitto).
"""

import logging
import random
from collections import deque
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# Heures d'ensoleillement moyennes (France métropolitaine)
SOLAR_CURVE = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0,
    6: 50, 7: 200, 8: 600, 9: 1200, 10: 2000, 11: 2800,
    12: 3200, 13: 3400, 14: 3200, 15: 2800, 16: 2000, 17: 1200,
    18: 600, 19: 200, 20: 50, 21: 0, 22: 0, 23: 0,
}


class IotController:
    """Contrôleur IoT pour le monitoring des actifs physiques."""

    def __init__(self, broker: str = "localhost"):
        self.broker = broker
        self.connected = False
        self.battery_level = 82.0
        self.daily_history: deque[dict[str, Any]] = deque(maxlen=90)
        self._seed_history()
        logger.info(f"🔌 IoT Controller initialisé (broker: {broker})")

    def _seed_history(self):
        """Génère un historique simulé sur 30 jours."""
        now = datetime.now()
        for i in range(30, 0, -1):
            day = now - timedelta(days=i)
            # Production aléatoire réaliste (3-25 kWh/jour selon saison)
            month = day.month
            seasonal_factor = 1.0 if month in (5, 6, 7, 8) else 0.6 if month in (3, 4, 9, 10) else 0.3
            production = round(random.uniform(5, 25) * seasonal_factor, 2)
            self.daily_history.append({
                "date": day.strftime("%Y-%m-%d"),
                "production_kwh": production,
                "savings_eur": round(production * 0.18, 2),  # ~0.18€/kWh
                "peak_w": round(production * 180 + random.randint(-200, 200)),
            })

    def _get_solar_production(self) -> float:
        """Calcule la production solaire simulée basée sur l'heure actuelle."""
        hour = datetime.now().hour
        base = SOLAR_CURVE.get(hour, 0)
        # Ajout variation aléatoire (nuages, etc.)
        variation = random.uniform(0.7, 1.1)
        return round(base * variation)

    def get_telemetry(self) -> dict[str, Any]:
        """
        Retourne les données de télémétrie instantanées.

        En production, ces données viendraient des capteurs via MQTT.
        En mode simulation, elles sont calculées en temps réel.
        """
        solar_w = self._get_solar_production()

        # Simulation batterie (charge quand solaire > 1kW, décharge la nuit)
        if solar_w > 1000:
            self.battery_level = min(100.0, self.battery_level + 0.1)
        elif solar_w == 0:
            self.battery_level = max(10.0, self.battery_level - 0.05)

        # Production journalière estimée
        daily_kwh = sum(
            SOLAR_CURVE.get(h, 0) for h in range(24)
        ) / 1000.0  # Wh → kWh

        return {
            "solar_production_w": solar_w,
            "battery_level": round(self.battery_level, 1),
            "external_temp": round(random.uniform(5, 30), 1),
            "daily_production_kwh": round(daily_kwh, 2),
            "monthly_savings_eur": round(daily_kwh * 30 * 0.18, 2),
            "status": "CONNECTED" if solar_w > 0 else "STANDBY",
            "mode": "SIMULATION",
            "timestamp": datetime.now().isoformat(),
        }

    def get_history(self, days: int = 7) -> dict[str, Any]:
        """Retourne l'historique de production sur N jours."""
        history = list(self.daily_history)[-days:]
        total_kwh = sum(d["production_kwh"] for d in history)
        total_savings = sum(d["savings_eur"] for d in history)

        return {
            "period_days": days,
            "history": history,
            "total_production_kwh": round(total_kwh, 2),
            "total_savings_eur": round(total_savings, 2),
            "avg_daily_kwh": round(total_kwh / max(len(history), 1), 2),
        }
