"""Filtre de calendrier economique pour THE HIVE."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

import httpx
from dateutil import parser

from shared.config import get_settings
from shared.telegram_client import TelegramClient

logger = logging.getLogger(__name__)


class NewsFilterService:
    """Bloque temporairement le trading autour des annonces macro critiques.

    Le filtre observe le calendrier Forex Factory et suspend le trading
    pendant une fenetre configurable avant et apres les evenements a fort
    impact. Les notifications Telegram distinguent les symboles bloques des
    symboles encore autorises.
    """

    def __init__(self, filter_minutes: int = 30) -> None:
        """Initialise le filtre de nouvelles.

        Args:
            filter_minutes (int): Nombre de minutes a bloquer avant et apres
                un evenement critique.
        """
        self.filter_minutes = filter_minutes
        self.is_active = False
        self.blocked_until: datetime | None = None
        self.current_blocking_event: str | None = None
        self.current_blocking_currency = "ALL"
        self.high_impact_events: list[dict[str, Any]] = []
        self._running = True
        self.last_fetch_time: datetime | None = None

    async def start_monitoring(self) -> None:
        """Demarre la surveillance continue du calendrier economique."""
        logger.info(
            "News Filter demarre (buffer: ±%s min).",
            self.filter_minutes,
        )
        while self._running:
            try:
                await self._check_calendar()
            except Exception as exc:
                logger.error("Erreur News Filter: %s", exc)
            await asyncio.sleep(60)

    async def _check_calendar(self) -> None:
        """Met a jour l'etat du filtre selon les annonces en cours."""
        self.high_impact_events = await self._fetch_economic_calendar()
        now = datetime.now()

        for event in self.high_impact_events:
            if event.get("impact", "").upper() not in {"HIGH", "HOLIDAY"}:
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

                    all_symbols = get_settings().banker_symbols
                    if self.current_blocking_currency != "ALL":
                        affected = [
                            symbol
                            for symbol in all_symbols
                            if self.current_blocking_currency in symbol
                        ]
                    else:
                        affected = list(all_symbols)
                    active = [symbol for symbol in all_symbols if symbol not in affected]

                    affected_text = ", ".join(affected) if affected else "Aucun"
                    active_text = ", ".join(active) if active else "Aucun"
                    message = (
                        f"\U0001F6A8 *NEWS FILTER ACTIV\u00c9*\n\n"
                        f"\U0001F4F0 *\u00c9v\u00e9nement*: `{event['name']}` "
                        f"({self.current_blocking_currency})\n"
                        f"\U0001F4A5 *Impact*: {event.get('impact', 'HIGH')}\n"
                        f"\u23F3 *Dur\u00e9e*: Jusqu'\u00e0 {window_end.strftime('%H:%M')}\n\n"
                        f"\U0001F6D1 *Trading Suspendu*:\n{affected_text}\n\n"
                        f"\u2705 *Trading Actif*:\n{active_text}"
                    )
                    logger.warning(
                        "News Filter active pour %s jusqu'a %s.",
                        self.current_blocking_currency,
                        window_end.strftime("%H:%M"),
                    )
                    asyncio.create_task(TelegramClient().send_message(message))
                return

        if self.is_active and (not self.blocked_until or now > self.blocked_until):
            message = (
                f"\u2705 *NEWS FILTER D\u00c9SACTIV\u00c9*\n"
                f"L'\u00e9v\u00e9nement `{self.current_blocking_event}` est termin\u00e9.\n"
                "Reprise du Trading."
            )
            self.is_active = False
            self.blocked_until = None
            self.current_blocking_event = None
            self.current_blocking_currency = "ALL"
            logger.info(message.replace("\n", " "))
            asyncio.create_task(TelegramClient().send_message(message))

    def should_block_trading(self, symbol: str = "") -> bool:
        """Indique si le symbole doit etre bloque par le filtre.

        Args:
            symbol (str): Symbole a verifier.

        Returns:
            bool: ``True`` si le trading doit etre bloque.
        """
        if not self.is_active:
            return False

        currency = self.current_blocking_currency
        if currency == "ALL" or not currency:
            return True
        return currency.upper() in symbol.upper()

    async def _fetch_economic_calendar(self) -> list[dict[str, Any]]:
        """Recupere le calendrier economique via Forex Factory.

        Un cache de 4 heures est applique pour limiter les erreurs ``429``.

        Returns:
            list[dict[str, Any]]: Liste triee des evenements connus.
        """
        now = datetime.now()
        if self.high_impact_events and self.last_fetch_time:
            if (now - self.last_fetch_time).total_seconds() < 14_400:
                return self.high_impact_events

        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)

                if response.status_code == 429:
                    logger.warning(
                        "News Filter: limite Forex Factory atteinte (429), nouveau test dans 15 min."
                    )
                    self.last_fetch_time = now - timedelta(seconds=14_400 - 900)
                    return self.high_impact_events

                response.raise_for_status()
                payload = response.json()

        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                logger.warning(
                    "News Filter: limite Forex Factory atteinte (429), nouveau test differe."
                )
                self.last_fetch_time = now - timedelta(seconds=14_400 - 900)
            else:
                logger.error("News Filter API error: %s", exc)
            return self.high_impact_events
        except Exception as exc:
            logger.error("Impossible de joindre Forex Factory: %s", exc)
            return self.high_impact_events

        events: list[dict[str, Any]] = []
        for item in payload:
            try:
                event_date = parser.isoparse(item["date"]).replace(tzinfo=None)
            except Exception as exc:
                logger.warning("Erreur de parsing calendrier: %s sur %s", exc, item)
                continue

            events.append(
                {
                    "name": item.get("title", "Unknown"),
                    "impact": item.get("impact", "").upper(),
                    "currency": item.get("country", ""),
                    "time": event_date,
                }
            )

        events.sort(key=lambda event: event["time"])
        self.last_fetch_time = now
        self.high_impact_events = events
        return events

    def get_status(self) -> dict[str, Any]:
        """Retourne l'etat detaille du filtre.

        Returns:
            dict[str, Any]: Etat courant et prochains evenements.
        """
        now = datetime.now()
        upcoming = [
            {
                "name": event["name"],
                "impact": event["impact"],
                "currency": event["currency"],
                "time": event["time"].isoformat(),
                "minutes_until": max(0, int((event["time"] - now).total_seconds() / 60)),
            }
            for event in self.high_impact_events
            if event["time"] > now
        ]
        return {
            "is_active": self.is_active,
            "blocked_until": self.blocked_until.isoformat() if self.blocked_until else None,
            "blocking_event": self.current_blocking_event,
            "filter_minutes": self.filter_minutes,
            "upcoming_events": upcoming[:5],
        }

    def stop(self) -> None:
        """Arrete proprement la surveillance du filtre."""
        self._running = False
