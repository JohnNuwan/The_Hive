"""
News Filter Service — Filtre de calendrier économique
═══════════════════════════════════════════════════

Bloque automatiquement le trading pendant les événements macro à fort impact.
Conforme à la Constitution ROE: news_filter_minutes = 30 (avant/après).

Événements surveillés (High Impact):
  - NFP (Non-Farm Payrolls)
  - FOMC (Federal Reserve)
  - CPI / PPI (Inflation)
  - BCE (ECB) décisions de taux
  - PMI Manufacturing / Services
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class NewsFilterService:
    """
    Filtre de nouvelles économiques pour The Banker.

    En production, se connecte à une API de calendrier économique
    (Forex Factory, Investing.com, MQL5 Calendar).
    """

    def __init__(self, filter_minutes: int = 30):
        self.filter_minutes = filter_minutes  # Minutes avant/après événement
        self.is_active = False
        self.blocked_until: Optional[datetime] = None
        self.current_blocking_event: Optional[str] = None
        self.high_impact_events: List[Dict[str, Any]] = []
        self._running = True

    async def start_monitoring(self) -> None:
        """Démarre la surveillance du calendrier en tâche de fond."""
        logger.info(
            f"📰 News Filter démarré (buffer: ±{self.filter_minutes}min)"
        )
        while self._running:
            try:
                await self._check_calendar()
            except Exception as e:
                logger.error(f"Erreur News Filter: {e}")
            await asyncio.sleep(60)  # Vérification chaque minute

    async def _check_calendar(self) -> None:
        """Vérifie le calendrier et active/désactive le filtre."""
        self.high_impact_events = await self._fetch_economic_calendar()
        now = datetime.now()

        for event in self.high_impact_events:
            if event["impact"] != "HIGH":
                continue

            event_time = event["time"]
            window_start = event_time - timedelta(minutes=self.filter_minutes)
            window_end = event_time + timedelta(minutes=self.filter_minutes)

            if window_start <= now <= window_end:
                if not self.is_active:
                    self.is_active = True
                    self.blocked_until = window_end
                    self.current_blocking_event = event["name"]
                    logger.warning(
                        f"🚨 NEWS FILTER ACTIVÉ: '{event['name']}' "
                        f"— Trading bloqué jusqu'à {window_end.strftime('%H:%M')}"
                    )
                return

        # Aucun événement en cours
        if self.is_active and (not self.blocked_until or now > self.blocked_until):
            self.is_active = False
            self.blocked_until = None
            self.current_blocking_event = None
            logger.info("✅ NEWS FILTER DÉSACTIVÉ — Trading autorisé")

    async def _fetch_economic_calendar(self) -> List[Dict[str, Any]]:
        """
        Récupère le calendrier économique.

        Stub: retourne des événements simulés.
        En production: connecter à ForexFactory API ou MQL5 Calendar.
        """
        now = datetime.now()
        return [
            {
                "name": "NFP Report",
                "impact": "HIGH",
                "currency": "USD",
                "time": now.replace(hour=14, minute=30, second=0),
            },
            {
                "name": "FOMC Rate Decision",
                "impact": "HIGH",
                "currency": "USD",
                "time": now.replace(hour=20, minute=0, second=0),
            },
            {
                "name": "CPI Data (YoY)",
                "impact": "HIGH",
                "currency": "USD",
                "time": now.replace(hour=14, minute=30, second=0) + timedelta(days=1),
            },
            {
                "name": "PMI Manufacturing",
                "impact": "MEDIUM",
                "currency": "EUR",
                "time": now.replace(hour=10, minute=0, second=0),
            },
        ]

    def should_block_trading(self) -> bool:
        """Retourne True si le trading doit être bloqué."""
        return self.is_active

    def get_status(self) -> Dict[str, Any]:
        """Retourne l'état complet du filtre."""
        now = datetime.now()
        upcoming = [
            {
                "name": e["name"],
                "impact": e["impact"],
                "currency": e["currency"],
                "time": e["time"].isoformat(),
                "minutes_until": max(0, int((e["time"] - now).total_seconds() / 60)),
            }
            for e in self.high_impact_events
            if e["time"] > now
        ]
        return {
            "is_active": self.is_active,
            "blocked_until": self.blocked_until.isoformat() if self.blocked_until else None,
            "blocking_event": self.current_blocking_event,
            "filter_minutes": self.filter_minutes,
            "upcoming_events": upcoming[:5],
        }

    def stop(self) -> None:
        self._running = False
