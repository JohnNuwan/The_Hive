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
            int(os.getenv("BANKER_SYMBOL_QUARANTINE_HOURS", "12")),
        )
        self.symbol_quarantine_lookback_hours = max(
            1,
            int(os.getenv("BANKER_SYMBOL_QUARANTINE_LOOKBACK_HOURS", "24")),
        )
        self.symbol_quarantine_same_type_threshold = max(
            2,
            int(os.getenv("BANKER_SYMBOL_QUARANTINE_SAME_TYPE_THRESHOLD", "3")),
        )
        self.symbol_quarantine_total_threshold = max(
            2,
            int(os.getenv("BANKER_SYMBOL_QUARANTINE_TOTAL_THRESHOLD", "4")),
        )
        self.defeat_ledger: List[Dict[str, Any]] = []
        self.known_nemeses: Dict[str, int] = {}
        self.lifetime_nemeses: Dict[str, int] = {}
        self.last_meditation_by_type: Dict[str, str] = {}
        self.symbol_quarantine_until_by_symbol: Dict[str, str] = {}
        self.quarantine_reason_by_symbol: Dict[str, str] = {}
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
        }
        self.defeat_ledger.append(defeat_entry)
        self._trim_ledger()

        self.lifetime_nemeses[nemesis_type] = self.lifetime_nemeses.get(nemesis_type, 0) + 1
        self._refresh_recent_nemeses(now=now)
        self._refresh_symbol_quarantines(now=now)
        self._update_symbol_quarantine(
            symbol=str(market_context.get("symbol") or "").strip().upper(),
            nemesis_type=nemesis_type,
            now=now,
        )
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

        await self._save_state()

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
        return False

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
                    "symbol_quarantine_until_by_symbol": self.symbol_quarantine_until_by_symbol,
                    "quarantine_reason_by_symbol": self.quarantine_reason_by_symbol,
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
            self.symbol_quarantine_until_by_symbol = dict(
                state.get("symbol_quarantine_until_by_symbol", {})
            )
            self.quarantine_reason_by_symbol = dict(
                state.get("quarantine_reason_by_symbol", {})
            )
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
            self._refresh_symbol_quarantines()
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
        self._refresh_symbol_quarantines()
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
            "quarantined_symbols": sorted(self.symbol_quarantine_until_by_symbol.keys()),
            "quarantine_reason_by_symbol": dict(self.quarantine_reason_by_symbol),
            "quarantine_until_by_symbol": dict(self.symbol_quarantine_until_by_symbol),
            "recent_defeats": self.defeat_ledger[-5:],
        }

    def is_symbol_quarantined(self, symbol: str) -> bool:
        """Indique si un symbole est temporairement retire du live universe.

        Args:
            symbol (str): Symbole a verifier.

        Returns:
            bool: ``True`` si le symbole reste en quarantaine.
        """
        normalized_symbol = str(symbol or "").strip().upper()
        if not normalized_symbol:
            return False
        self._refresh_symbol_quarantines()
        until_raw = self.symbol_quarantine_until_by_symbol.get(normalized_symbol)
        until_dt = self._parse_timestamp(until_raw)
        return bool(until_dt and datetime.now() < until_dt)

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

    def _refresh_symbol_quarantines(self, now: Optional[datetime] = None) -> None:
        """Nettoie les quarantaines symbole expirees.

        Args:
            now (Optional[datetime]): Horodatage de reference.
        """
        now = now or datetime.now()
        expired_symbols = [
            symbol
            for symbol, until_raw in self.symbol_quarantine_until_by_symbol.items()
            if (self._parse_timestamp(until_raw) or now) <= now
        ]
        for symbol in expired_symbols:
            self.symbol_quarantine_until_by_symbol.pop(symbol, None)
            self.quarantine_reason_by_symbol.pop(symbol, None)

    def _update_symbol_quarantine(
        self,
        *,
        symbol: str,
        nemesis_type: str,
        now: Optional[datetime] = None,
    ) -> None:
        """Met a jour la quarantaine locale d'un symbole.

        Args:
            symbol (str): Symbole concerne par la perte.
            nemesis_type (str): Type de defaite classe.
            now (Optional[datetime]): Horodatage de reference.
        """
        normalized_symbol = str(symbol or "").strip().upper()
        if not normalized_symbol:
            return
        now = now or datetime.now()
        lookback_start = now - timedelta(hours=self.symbol_quarantine_lookback_hours)
        same_type_losses = 0
        total_losses = 0
        for entry in self.defeat_ledger:
            entry_time = self._parse_timestamp(entry.get("timestamp"))
            if entry_time is None or entry_time < lookback_start:
                continue
            entry_context = dict(entry.get("context") or {})
            entry_symbol = str(entry_context.get("symbol") or "").strip().upper()
            if entry_symbol != normalized_symbol:
                continue
            total_losses += 1
            if str(entry.get("nemesis_type") or "").strip().upper() == nemesis_type:
                same_type_losses += 1

        if (
            same_type_losses < self.symbol_quarantine_same_type_threshold
            and total_losses < self.symbol_quarantine_total_threshold
        ):
            return

        quarantine_until = now + timedelta(hours=self.symbol_quarantine_hours)
        self.symbol_quarantine_until_by_symbol[normalized_symbol] = quarantine_until.isoformat()
        if same_type_losses >= self.symbol_quarantine_same_type_threshold:
            reason = f"nemesis_repeat::{nemesis_type}"
        else:
            reason = "defeats_repeat::all"
        self.quarantine_reason_by_symbol[normalized_symbol] = reason
        logger.warning(
            "Quarantaine live activee pour %s jusqu'a %s (%s).",
            normalized_symbol,
            quarantine_until.isoformat(),
            reason,
        )

    def predict_trap(self, symbol: str, action: str = "BUY") -> dict:
        """Evalue le risque de LIQUIDITY_TRAP avant l'ouverture d'un trade.

        Methode appelee PRE-TRADE pour bloquer preventif les setups risques
        avant d'envoyer l'ordre MT5.

        Args:
            symbol (str): Symbole a evaluer.
            action (str): Direction du trade (BUY ou SELL).

        Returns:
            dict: block (bool), risk_score (float 0-1), reason (str).
        """
        normalized_symbol = str(symbol or "").strip().upper()
        now = datetime.now()
        lookback = now - timedelta(hours=self.lookback_hours)

        recent_symbol_losses = []
        for entry in self.defeat_ledger:
            entry_time = self._parse_timestamp(entry.get("timestamp"))
            if not entry_time or entry_time < lookback:
                continue
            ctx = dict(entry.get("context") or {})
            entry_symbol = str(ctx.get("symbol") or "").strip().upper()
            if entry_symbol == normalized_symbol:
                recent_symbol_losses.append(entry)

        n_losses = len(recent_symbol_losses)
        liquidity_trap_losses = sum(
            1 for e in recent_symbol_losses
            if str(e.get("nemesis_type") or "").upper() == "LIQUIDITY_TRAP"
        )

        total_nemesis_defeats = max(sum(self.known_nemeses.values()), 1)
        global_trap_rate = self.known_nemeses.get("LIQUIDITY_TRAP", 0) / total_nemesis_defeats

        symbol_score = min(1.0, n_losses / max(self.symbol_quarantine_same_type_threshold, 1))
        global_score = min(1.0, global_trap_rate)
        risk_score = round(0.6 * symbol_score + 0.4 * global_score, 3)

        threshold = float(os.getenv("BANKER_NEMESIS_PRETRADE_BLOCK_THRESHOLD", "0.70"))
        should_block = risk_score >= threshold

        if should_block:
            reason = (
                f"Nemesis pre-trade BLOCK {action} {normalized_symbol}: "
                f"{n_losses} pertes recentes ({liquidity_trap_losses} LIQUIDITY_TRAP) "
                f"+ taux_global={global_trap_rate:.0%}. Score={risk_score:.2f}>={threshold:.2f}"
            )
            logger.warning("Nemesis pre-trade block: %s", reason)
        else:
            reason = (
                f"Nemesis pre-trade OK: score={risk_score:.2f}<{threshold:.2f} "
                f"({n_losses} pertes recentes sur {normalized_symbol})"
            )

        return {
            "block": should_block,
            "risk_score": risk_score,
            "reason": reason,
            "recent_losses": n_losses,
            "liquidity_trap_losses": liquidity_trap_losses,
            "global_trap_rate": global_trap_rate,
        }

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
