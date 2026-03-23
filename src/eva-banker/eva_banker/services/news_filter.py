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

import httpx
from dateutil import parser

from shared.telegram_client import TelegramClient
from shared.config import get_settings

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
        self.current_blocking_currency: str = "ALL"
        self.high_impact_events: List[Dict[str, Any]] = []
        self._running = True
        self.last_fetch_time: Optional[datetime] = None

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
            # On considère "High" (et parfois Holiday comme critique selon les setups)
            if event.get("impact", "").upper() not in ["HIGH", "HOLIDAY"]:
                continue

            event_time = event["time"]
            window_start = event_time - timedelta(minutes=self.filter_minutes)
            window_end = event_time + timedelta(minutes=self.filter_minutes)

            if window_start <= now <= window_end:
                if not self.is_active:
                    self.is_active = True
                    self.blocked_until = window_end
                    self.current_blocking_event = event["name"]
                    self.current_blocking_currency = event.get("currency", "ALL").upper()
                    
                    # Compute Affected vs Active Symbols
                    all_symbols = get_settings().banker_symbols
                    affected = [s for s in all_symbols if self.current_blocking_currency in s] if self.current_blocking_currency != "ALL" else all_symbols
                    active = [s for s in all_symbols if s not in affected]
                    
                    # Formatting
                    aff_str = ", ".join(affected) if affected else "Aucun"
                    act_str = ", ".join(active) if active else "Aucun"
                    
                    msg = (
                        f"🚨 *NEWS FILTER ACTIVÉ*\n\n"
                        f"📰 *Événement*: `{event['name']}` ({self.current_blocking_currency})\n"
                        f"💥 *Impact*: {event.get('impact', 'HIGH')}\n"
                        f"⏳ *Durée*: Jusqu'à {window_end.strftime('%H:%M')}\n\n"
                        f"🛑 *Trading Suspendu*:\n{aff_str}\n\n"
                        f"✅ *Trading Actif*:\n{act_str}"
                    )
                    logger.warning(f"News Filter activated for {self.current_blocking_currency} until {window_end.strftime('%H:%M')}")
                    asyncio.create_task(TelegramClient().send_message(msg))
                return

        # Aucun événement en cours
        if self.is_active and (not self.blocked_until or now > self.blocked_until):
            self.is_active = False
            self.blocked_until = None
            msg = f"✅ *NEWS FILTER DÉSACTIVÉ*\nL'événement `{self.current_blocking_event}` est terminé.\nReprise du Trading."
            self.current_blocking_event = None
            self.current_blocking_currency = "ALL"
            logger.info(msg.replace("\n", " "))
            asyncio.create_task(TelegramClient().send_message(msg))

    def should_block_trading(self, symbol: str = "") -> bool:
        """Retourne True si le trading doit être bloqué pour ce symbole."""
        if not self.is_active: return False
        
        curr = self.current_blocking_currency
        if curr == "ALL" or not curr:
            return True
            
        # Bloque uniquement si la devise impactée fait partie du symbole
        if curr.upper() in symbol.upper():
            return True
            
        return False

    async def _fetch_economic_calendar(self) -> List[Dict[str, Any]]:
        """
        Récupère le calendrier économique temps réel via ForexFactory avec un cache de 4 heures.
        """
        now = datetime.now()
        # Anti 429: Fetch uniquement toutes les 4 heures
        if self.high_impact_events and self.last_fetch_time:
            if (now - self.last_fetch_time).total_seconds() < 14400:
                return self.high_impact_events

        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                
                if response.status_code == 429:
                    logger.warning("⚠️ News Filter: ForexFactory Rate Limit (429). Backing off 15 min.")
                    self.last_fetch_time = now - timedelta(seconds=14400 - 900) # retry in 15 min
                    return self.high_impact_events
                    
                response.raise_for_status()
                data = response.json()
                
                events = []
                for item in data:
                    try:
                        # Parsing des dates ISO 8601 renvoyées par ForexFactory
                        event_date = parser.isoparse(item["date"]).replace(tzinfo=None)
                        events.append({
                            "name": item.get("title", "Unknown"),
                            "impact": item.get("impact", "").upper(),
                            "currency": item.get("country", ""),
                            "time": event_date,
                        })
                    except Exception as e:
                        logger.warning(f"Erreur parsing date calendrier: {e} sur {item}")
                        
                # On trie par date histoire d'être propre
                events.sort(key=lambda x: x["time"])
                self.last_fetch_time = now
                self.high_impact_events = events
                return events
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                 logger.warning("⚠️ News Filter: 429 Rate Limit hit. Backing off.")
                 self.last_fetch_time = now - timedelta(seconds=14400 - 900)
            else:
                 logger.error(f"News Filter API error: {e}")
            return self.high_impact_events
        except Exception as e:
            logger.error(f"Impossible de joindre ForexFactory: {e}")
            # Renvoie le cache précédent si échec au lieu de vider la mémoire
            return self.high_impact_events

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
