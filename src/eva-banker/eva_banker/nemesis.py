"""
Nemesis System - Apprentissage adaptatif des defaites.

Inspire du Nemesis System de "Shadow of Mordor".
Chaque perte en trading est analysee, classifiee et memorisee.
Quand un type d'ennemi bat EVA plusieurs fois dans une fenetre recente,
une phase de meditation est declenchee pour suspendre temporairement le
trading et lancer un apprentissage correctif.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class NemesisSystem:
    """
    Gere la memoire des defaites et le blocage anti-tilt adaptatif.

    Le systeme opere sur une fenetre glissante recente plutot qu'un compteur a
    vie. Cela evite de retrigger une meditation sur des pertes anciennes deja
    traitees.
    """

    def __init__(self) -> None:
        """
        Initialise les seuils et les structures de suivi Nemesis.
        """
        self.block_duration_seconds = max(
            300,
            int(os.getenv("BANKER_NEMESIS_BLOCK_SECONDS", "3600")),
        )
        self.trigger_threshold = max(
            2,
            int(os.getenv("BANKER_NEMESIS_TRIGGER_THRESHOLD", "3")),
        )
        self.lookback_hours = max(
            1,
            int(os.getenv("BANKER_NEMESIS_LOOKBACK_HOURS", "12")),
        )
        self.max_ledger_entries = max(
            50,
            int(os.getenv("BANKER_NEMESIS_MAX_LEDGER_ENTRIES", "200")),
        )
        self.symbol_quarantine_hours = max(
            1,
            int(os.getenv("BANKER_NEMESIS_SYMBOL_QUARANTINE_HOURS", "4")),
        )
        self.symbol_loss_streak_threshold = max(
            2,
            int(os.getenv("BANKER_NEMESIS_SYMBOL_LOSS_STREAK_THRESHOLD", "2")),
        )
        self.symbol_event_threshold = max(
            2,
            int(os.getenv("BANKER_NEMESIS_SYMBOL_EVENT_THRESHOLD", "3")),
        )
        self.symbol_loss_lookback_hours = max(
            1,
            int(os.getenv("BANKER_NEMESIS_SYMBOL_LOSS_LOOKBACK_HOURS", "4")),
        )
        self.symbol_event_lookback_hours = max(
            1,
            int(os.getenv("BANKER_NEMESIS_SYMBOL_EVENT_LOOKBACK_HOURS", "12")),
        )
        self.global_loss_threshold = max(
            2,
            int(os.getenv("BANKER_NEMESIS_GLOBAL_LOSS_THRESHOLD", "4")),
        )
        self.global_loss_lookback_hours = max(
            1,
            int(os.getenv("BANKER_NEMESIS_GLOBAL_LOSS_LOOKBACK_HOURS", "6")),
        )
        self.global_quarantine_symbol_threshold = max(
            2,
            int(os.getenv("BANKER_NEMESIS_GLOBAL_QUARANTINE_SYMBOL_THRESHOLD", "2")),
        )
        self.symbol_loss_day_threshold_percent = max(
            0.10,
            float(os.getenv("BANKER_NEMESIS_SYMBOL_LOSS_DAY_THRESHOLD_PERCENT", "0.60")),
        )
        self.defeat_ledger: List[Dict[str, Any]] = []
        self.known_nemeses: Dict[str, int] = {}
        self.lifetime_nemeses: Dict[str, int] = {}
        self.last_meditation_by_type: Dict[str, str] = {}
        self.quarantine_expires_at_by_symbol: Dict[str, str] = {}
        self.recent_losses_by_symbol: Dict[str, Dict[str, Any]] = {}
        self.escalation_state = "normal"
        self.trading_blocked_until: Optional[datetime] = None
        self._meditation_in_progress = False

    async def report_loss(
        self,
        trade_id: str,
        loss_amount: float,
        market_context: Dict[str, Any],
    ) -> None:
        """
        Enregistre une defaite et met a jour l'etat Nemesis.

        Args:
            trade_id (str): Identifiant unique du trade cloture.
            loss_amount (float): Montant de la perte constatee.
            market_context (Dict[str, Any]): Contexte de marche associe.
        """
        if any(entry.get("trade_id") == trade_id for entry in self.defeat_ledger):
            logger.info("Nemesis ignore un doublon de defaite pour le trade %s.", trade_id)
            return

        now = datetime.now()
        nemesis_type = self._classify_nemesis(market_context)
        defeat_entry = {
            "timestamp": now.isoformat(),
            "trade_id": trade_id,
            "loss": loss_amount,
            "context": market_context,
            "nemesis_type": nemesis_type,
            "symbol": str(market_context.get("symbol") or "").strip().upper() or None,
        }
        self.defeat_ledger.append(defeat_entry)
        self._trim_ledger()

        self.lifetime_nemeses[nemesis_type] = self.lifetime_nemeses.get(nemesis_type, 0) + 1
        self._refresh_recent_nemeses(now=now)
        self._refresh_symbol_controls(now=now)
        recent_count = self.known_nemeses.get(nemesis_type, 0)

        logger.warning(
            "Nemesis '%s' detecte (trade %s). Defaites recentes: %s/%s.",
            nemesis_type,
            trade_id,
            recent_count,
            self.trigger_threshold,
        )

        if recent_count >= self.trigger_threshold:
            await self._trigger_meditation(nemesis_type, now=now)

        symbol = str(defeat_entry.get("symbol") or "").strip().upper()
        if symbol:
            self._apply_symbol_quarantine_rules(symbol=symbol, now=now)

        if self._should_escalate_globally(now=now, context=market_context):
            self.escalation_state = "global_blocked"
            await self._trigger_meditation("GLOBAL_ESCALATION", now=now)
        elif self.quarantine_expires_at_by_symbol:
            self.escalation_state = "symbol_quarantine"
        else:
            self.escalation_state = "normal"

        await self._save_state()

    def preview_nemesis_type(self, market_context: Dict[str, Any]) -> str:
        """
        Retourne le type Nemesis probable sans modifier l'etat interne.

        Args:
            market_context (Dict[str, Any]): Contexte capture lors de la perte.

        Returns:
            str: Type Nemesis estime a partir du contexte courant.
        """
        return self._classify_nemesis(market_context)

    def _classify_nemesis(self, context: Dict[str, Any]) -> str:
        """
        Classifie la cause probable de la perte.

        Args:
            context (Dict[str, Any]): Contexte capture lors de la perte.

        Returns:
            str: Type de Nemesis detecte.
        """
        volatility = context.get("volatility", 0)
        news_event = context.get("news_event", False)
        trend_reversal = context.get("trend_reversal", False)

        if news_event:
            return "BLACK_SWAN_NEMESIS"
        if trend_reversal:
            return "TREND_REVERSAL"
        if volatility > 0.03:
            return "WHIPLASH_VOLATILITY"
        return "LIQUIDITY_TRAP"

    async def _trigger_meditation(
        self,
        nemesis_type: str,
        now: Optional[datetime] = None,
    ) -> None:
        """
        Declenche une meditation si aucun blocage n'est deja actif.

        Args:
            nemesis_type (str): Type de pattern responsable du blocage.
            now (Optional[datetime]): Horodatage de reference si deja calcule.
        """
        if self.should_block_trading():
            logger.info(
                "Meditation deja active. Blocage conserve pour le type %s.",
                nemesis_type,
            )
            return

        now = now or datetime.now()
        self._meditation_in_progress = True
        self.trading_blocked_until = now + timedelta(seconds=self.block_duration_seconds)
        self.last_meditation_by_type[nemesis_type] = now.isoformat()
        self._refresh_recent_nemeses(now=now)
        logger.info(
            "Meditation declenchee pour Nemesis '%s'. Trading bloque jusqu'a %s.",
            nemesis_type,
            self.trading_blocked_until.strftime("%H:%M"),
        )

        try:
            from shared.redis_client import get_redis_client

            redis = get_redis_client()
            await redis.broadcast_to_swarm(
                source="banker",
                action="NEMESIS_MEDITATION",
                payload={
                    "nemesis_type": nemesis_type,
                    "defeats": self.known_nemeses.get(nemesis_type, 0),
                    "blocked_until": self.trading_blocked_until.isoformat(),
                },
            )
        except Exception as exc:
            logger.error("Erreur notification meditation: %s", exc)

    def should_block_trading(self) -> bool:
        """
        Indique si le trading doit rester bloque.

        Returns:
            bool: True si une meditation est active.
        """
        if self.trading_blocked_until and datetime.now() < self.trading_blocked_until:
            return True
        if self.trading_blocked_until and datetime.now() >= self.trading_blocked_until:
            self._meditation_in_progress = False
            self.trading_blocked_until = None
            self._refresh_recent_nemeses()
            self._refresh_symbol_controls()
            self.escalation_state = (
                "symbol_quarantine" if self.quarantine_expires_at_by_symbol else "normal"
            )
        return False

    def is_symbol_quarantined(self, symbol: str) -> bool:
        """
        Indique si un symbole est temporairement exclu des nouvelles entrees.

        Args:
            symbol (str): Symbole a verifier.

        Returns:
            bool: ``True`` si le symbole est en quarantaine.
        """
        self._refresh_symbol_controls()
        return str(symbol or "").strip().upper() in self.quarantine_expires_at_by_symbol

    async def _save_state(self) -> None:
        """
        Persiste l'etat courant dans Redis si disponible.
        """
        try:
            from shared.redis_client import get_redis_client

            redis = get_redis_client()
            await redis.cache_set(
                "nemesis:state",
                {
                    "defeat_ledger": self.defeat_ledger[-self.max_ledger_entries :],
                    "known_nemeses": self.known_nemeses,
                    "lifetime_nemeses": self.lifetime_nemeses,
                    "last_meditation_by_type": self.last_meditation_by_type,
                    "quarantine_expires_at_by_symbol": self.quarantine_expires_at_by_symbol,
                    "recent_losses_by_symbol": self.recent_losses_by_symbol,
                    "escalation_state": self.escalation_state,
                    "trading_blocked_until": (
                        self.trading_blocked_until.isoformat()
                        if self.trading_blocked_until
                        else None
                    ),
                },
                ttl_seconds=86400,
            )
        except Exception:
            pass

    async def load_state(self) -> None:
        """
        Recharge l'etat depuis Redis au demarrage.
        """
        try:
            from shared.redis_client import get_redis_client

            redis = get_redis_client()
            state = await redis.cache_get("nemesis:state")
            if not state:
                return

            self.defeat_ledger = state.get("defeat_ledger", [])
            self.known_nemeses = state.get("known_nemeses", {})
            self.lifetime_nemeses = state.get(
                "lifetime_nemeses",
                state.get("known_nemeses", {}),
            )
            self.last_meditation_by_type = state.get("last_meditation_by_type", {})
            self.quarantine_expires_at_by_symbol = dict(
                state.get("quarantine_expires_at_by_symbol") or {}
            )
            self.recent_losses_by_symbol = dict(state.get("recent_losses_by_symbol") or {})
            self.escalation_state = str(state.get("escalation_state") or "normal")
            if state.get("trading_blocked_until"):
                self.trading_blocked_until = datetime.fromisoformat(
                    state["trading_blocked_until"]
                )
                self._meditation_in_progress = datetime.now() < self.trading_blocked_until

            if not self.last_meditation_by_type and self.known_nemeses:
                now_iso = datetime.now().isoformat()
                self.last_meditation_by_type = {
                    nemesis_type: now_iso
                    for nemesis_type, defeats in self.known_nemeses.items()
                    if int(defeats) >= self.trigger_threshold
                }

            self._trim_ledger()
            self._refresh_recent_nemeses()
            self._refresh_symbol_controls()
            logger.info("Etat Nemesis charge depuis Redis.")
        except Exception:
            pass

    def get_status(self) -> Dict[str, Any]:
        """
        Retourne l'etat courant du systeme Nemesis.

        Returns:
            Dict[str, Any]: Blocage, compteurs recents et historique recent.
        """
        self._refresh_recent_nemeses()
        self._refresh_symbol_controls()
        return {
            "total_defeats": len(self.defeat_ledger),
            "known_nemeses": self.known_nemeses,
            "lifetime_nemeses": self.lifetime_nemeses,
            "trading_blocked": self.should_block_trading(),
            "blocked_until": (
                self.trading_blocked_until.isoformat()
                if self.trading_blocked_until
                else None
            ),
            "meditation_active": self._meditation_in_progress,
            "trigger_threshold": self.trigger_threshold,
            "lookback_hours": self.lookback_hours,
            "quarantined_symbols": sorted(self.quarantine_expires_at_by_symbol.keys()),
            "quarantine_expires_at_by_symbol": self.quarantine_expires_at_by_symbol,
            "recent_losses_by_symbol": self.recent_losses_by_symbol,
            "escalation_state": self.escalation_state,
            "last_meditation_by_type": self.last_meditation_by_type,
            "recent_defeats": self.defeat_ledger[-5:],
        }

    def _trim_ledger(self) -> None:
        """
        Tronque le journal des defaites pour garder une taille bornee.
        """
        if len(self.defeat_ledger) > self.max_ledger_entries:
            self.defeat_ledger = self.defeat_ledger[-self.max_ledger_entries :]

    def _refresh_recent_nemeses(self, now: Optional[datetime] = None) -> None:
        """
        Recalcule les compteurs operationnels sur une fenetre glissante.

        Args:
            now (Optional[datetime]): Horodatage de reference optionnel.
        """
        now = now or datetime.now()
        lookback_start = now - timedelta(hours=self.lookback_hours)
        recent_counts: Dict[str, int] = {}

        for entry in self.defeat_ledger:
            entry_time = self._parse_timestamp(entry.get("timestamp"))
            if entry_time is None or entry_time < lookback_start:
                continue

            nemesis_type = str(entry.get("nemesis_type") or "UNKNOWN")
            meditation_start = self._parse_timestamp(
                self.last_meditation_by_type.get(nemesis_type)
            )
            if meditation_start and entry_time <= meditation_start:
                continue
            recent_counts[nemesis_type] = recent_counts.get(nemesis_type, 0) + 1

        self.known_nemeses = recent_counts

    def _refresh_symbol_controls(self, now: Optional[datetime] = None) -> None:
        """
        Recalcule les compteurs recents par symbole et purge les quarantaines expirees.

        Args:
            now (Optional[datetime]): Horodatage de reference optionnel.
        """
        now = now or datetime.now()
        cleaned_quarantine: Dict[str, str] = {}
        for symbol, expires_at in dict(self.quarantine_expires_at_by_symbol).items():
            expiry_dt = self._parse_timestamp(expires_at)
            if expiry_dt is not None and expiry_dt > now:
                cleaned_quarantine[symbol] = expires_at
        self.quarantine_expires_at_by_symbol = cleaned_quarantine

        recent_summary: Dict[str, Dict[str, Any]] = {}
        global_window_start = now - timedelta(
            hours=max(
                self.symbol_loss_lookback_hours,
                self.symbol_event_lookback_hours,
                self.global_loss_lookback_hours,
            )
        )
        today = now.date()
        for entry in self.defeat_ledger:
            entry_time = self._parse_timestamp(entry.get("timestamp"))
            if entry_time is None or entry_time < global_window_start:
                continue
            symbol = str(entry.get("symbol") or ((entry.get("context") or {}).get("symbol")) or "").strip().upper()
            if not symbol:
                continue
            item = recent_summary.setdefault(
                symbol,
                {
                    "recent_losses_4h": 0,
                    "recent_events_12h": 0,
                    "recent_losses_6h": 0,
                    "day_loss_amount": 0.0,
                    "day_loss_percent": 0.0,
                    "latest_nemesis_type": str(entry.get("nemesis_type") or "UNKNOWN"),
                },
            )
            if entry_time >= now - timedelta(hours=self.symbol_loss_lookback_hours):
                item["recent_losses_4h"] += 1
            if entry_time >= now - timedelta(hours=self.symbol_event_lookback_hours):
                item["recent_events_12h"] += 1
            if entry_time >= now - timedelta(hours=self.global_loss_lookback_hours):
                item["recent_losses_6h"] += 1
            if entry_time.date() == today:
                item["day_loss_amount"] += float(entry.get("loss") or 0.0)
                day_open_balance = float(
                    ((entry.get("context") or {}).get("day_open_balance") or 0.0)
                )
                if day_open_balance > 0:
                    item["day_loss_percent"] = round(
                        item["day_loss_amount"] / day_open_balance * 100.0,
                        4,
                    )
        self.recent_losses_by_symbol = recent_summary

    def _apply_symbol_quarantine_rules(self, symbol: str, now: Optional[datetime] = None) -> None:
        """
        Place un symbole en quarantaine si ses pertes recentes deviennent anormales.

        Args:
            symbol (str): Symbole a evaluer.
            now (Optional[datetime]): Horodatage de reference.
        """
        now = now or datetime.now()
        self._refresh_symbol_controls(now=now)
        summary = dict(self.recent_losses_by_symbol.get(symbol) or {})
        should_quarantine = (
            int(summary.get("recent_losses_4h", 0)) >= self.symbol_loss_streak_threshold
            or int(summary.get("recent_events_12h", 0)) >= self.symbol_event_threshold
            or float(summary.get("day_loss_percent", 0.0)) >= self.symbol_loss_day_threshold_percent
        )
        if not should_quarantine:
            return

        expires_at = now + timedelta(hours=self.symbol_quarantine_hours)
        previous_expiry = self._parse_timestamp(self.quarantine_expires_at_by_symbol.get(symbol))
        if previous_expiry is None or expires_at > previous_expiry:
            self.quarantine_expires_at_by_symbol[symbol] = expires_at.isoformat()
            logger.warning(
                "Nemesis place %s en quarantaine jusqu'a %s.",
                symbol,
                expires_at.isoformat(),
            )

    def _should_escalate_globally(
        self,
        *,
        now: Optional[datetime] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Decide si les pertes recentes imposent un blocage global.

        Args:
            now (Optional[datetime]): Horodatage de reference.
            context (Optional[Dict[str, Any]]): Contexte de perte courant.

        Returns:
            bool: ``True`` si un blocage global doit etre applique.
        """
        now = now or datetime.now()
        self._refresh_symbol_controls(now=now)
        if bool((context or {}).get("risk_governor_triggered")):
            return True
        if len(self.quarantine_expires_at_by_symbol) >= self.global_quarantine_symbol_threshold:
            return True

        recent_losses = 0
        window_start = now - timedelta(hours=self.global_loss_lookback_hours)
        for entry in self.defeat_ledger:
            entry_time = self._parse_timestamp(entry.get("timestamp"))
            if entry_time is not None and entry_time >= window_start:
                recent_losses += 1
        return recent_losses >= self.global_loss_threshold

    @staticmethod
    def _parse_timestamp(raw_value: Optional[str]) -> Optional[datetime]:
        """
        Convertit un timestamp ISO en datetime.

        Args:
            raw_value (Optional[str]): Horodatage au format ISO.

        Returns:
            Optional[datetime]: Horodatage parse ou None si invalide.
        """
        if not raw_value:
            return None
        try:
            return datetime.fromisoformat(raw_value)
        except ValueError:
            logger.warning("Timestamp Nemesis invalide ignore: %s", raw_value)
            return None


_nemesis_instance: Optional[NemesisSystem] = None


def get_nemesis_system() -> NemesisSystem:
    """
    Retourne l'instance singleton du systeme Nemesis.

    Returns:
        NemesisSystem: Instance partagee du systeme Nemesis.
    """
    global _nemesis_instance
    if _nemesis_instance is None:
        _nemesis_instance = NemesisSystem()
    return _nemesis_instance
