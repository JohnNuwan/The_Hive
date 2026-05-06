"""Analyse les trades live clotures pour alimenter MuZero et le runtime."""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LiveTradeReviewRecord:
    """Represente une revue structuree d'un trade live ferme.

    Args:
        recorded_at (str): Horodatage de l'enregistrement de la revue.
        closed_at (str): Horodatage de cloture du trade.
        symbol (str): Symbole du trade.
        action (str): Direction initiale du trade.
        pnl (float): Profit ou perte realise.
        directional_return_pct (float): Rendement directionnel du trade en pourcentage.
        close_reason (str): Raison normalisee de cloture.
        quality_label (str): Etiquette metier principale du trade.
        trade_bucket (str): Famille de supervision (positif, negatif, neutre).
        nemesis_type (str): Type Nemesis associe si disponible.
        context_label (str): Regime de marche capture a l'ouverture.
        strategy_family (str): Famille strategique ou skill source.
        live_engine (str): Moteur live ayant execute le trade.
        live_champion_id (str): Identifiant du champion actif.
        ticket (int | None): Ticket MT5 si disponible.
        duration_minutes (float): Duree du trade en minutes.
        metadata (dict[str, Any]): Metadonnees completes de cloture.
    """

    recorded_at: str
    closed_at: str
    symbol: str
    action: str
    pnl: float
    directional_return_pct: float
    close_reason: str
    quality_label: str
    trade_bucket: str
    nemesis_type: str
    context_label: str
    strategy_family: str
    live_engine: str
    live_champion_id: str
    ticket: int | None
    duration_minutes: float
    metadata: dict[str, Any]


class LiveTradeReviewService:
    """Stocke et synthestise les clotures de trades live.

    Ce service est volontairement additif: il ne modifie pas les decisions live
    ni le training en cours. Il collecte des signaux plus riches afin de nourrir
    ensuite le replay MuZero, la selection de seeds et des hints de mutation
    runtime.

    Args:
        data_dir (str): Repertoire de persistence des revues live.
        rolling_days (int | None): Fenetre glissante d'analyse.
        min_trades_for_hints (int | None): Minimum de trades pour emettre des hints.
    """

    def __init__(
        self,
        data_dir: str = "data/live_trade_reviews",
        rolling_days: int | None = None,
        min_trades_for_hints: int | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.rolling_days = max(
            1,
            int(rolling_days or os.getenv("LIVE_TRADE_REVIEW_ROLLING_DAYS", "5")),
        )
        self.min_trades_for_hints = max(
            5,
            int(min_trades_for_hints or os.getenv("LIVE_TRADE_REVIEW_MIN_TRADES", "20")),
        )
        self.latest_summary_path = self.data_dir / "latest_summary.json"
        self._records: list[LiveTradeReviewRecord] = self._load_recent_records()

    def record_closed_trade(
        self,
        *,
        symbol: str,
        action: str,
        price: float,
        volume: float,
        pnl: float,
        indicators: dict[str, Any] | None = None,
        observation: dict[str, Any] | None = None,
        next_observation: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        """Enregistre et score un trade ferme.

        Args:
            symbol (str): Symbole du trade.
            action (str): Direction du trade.
            price (float): Prix de cloture connu.
            volume (float): Volume cloture.
            pnl (float): Profit ou perte realise.
            indicators (dict[str, Any] | None): Indicateurs associes.
            observation (dict[str, Any] | None): Observation d'ouverture ou de cloture.
            next_observation (dict[str, Any] | None): Observation suivante.
            metadata (dict[str, Any] | None): Metadonnees enrichies du Banker.
            timestamp (str | None): Horodatage de cloture.

        Returns:
            dict[str, Any]: Etiquette retenue et resume agreges.
        """

        normalized_metadata = self._sanitize_mapping(metadata or {})
        record = self._build_record(
            symbol=symbol,
            action=action,
            price=price,
            volume=volume,
            pnl=pnl,
            indicators=indicators or {},
            observation=observation or {},
            next_observation=next_observation or {},
            metadata=normalized_metadata,
            timestamp=timestamp,
        )
        self._records.append(record)
        self._trim_records()
        self._append_record(record)
        summary = self.get_summary()
        return {
            "quality_label": record.quality_label,
            "trade_bucket": record.trade_bucket,
            "symbol": record.symbol,
            "ga_runtime_hints": summary.get("ga_runtime_hints", {}),
            "summary": {
                "total_closed_trades": summary.get("total_closed_trades", 0),
                "net_pnl": summary.get("net_pnl", 0.0),
                "hard_negative_mix": summary.get("hard_negative_mix", {}),
            },
        }

    def get_summary(self) -> dict[str, Any]:
        """Construit le resume glissant des trades fermes.

        Returns:
            dict[str, Any]: Resume complet et hints runtime.
        """

        window_start = datetime.now() - timedelta(days=self.rolling_days)
        records = [
            record
            for record in self._records
            if (self._parse_iso(record.closed_at) or self._parse_iso(record.recorded_at) or datetime.now())
            >= window_start
        ]
        total_closed = len(records)
        label_counter: Counter[str] = Counter()
        nemesis_counter: Counter[str] = Counter()
        symbol_payload: dict[str, dict[str, Any]] = {}
        net_pnl = 0.0
        gross_profit = 0.0
        gross_loss = 0.0
        positive_anchors = 0
        hard_negatives = 0

        for record in records:
            label_counter[record.quality_label] += 1
            if record.nemesis_type:
                nemesis_counter[record.nemesis_type] += 1

            net_pnl += record.pnl
            if record.pnl > 0.0:
                gross_profit += record.pnl
            elif record.pnl < 0.0:
                gross_loss += abs(record.pnl)

            if record.trade_bucket == "positive_anchor":
                positive_anchors += 1
            elif record.trade_bucket == "hard_negative":
                hard_negatives += 1

            symbol_state = symbol_payload.setdefault(
                record.symbol,
                {
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "net_pnl": 0.0,
                    "positive_anchors": 0,
                    "hard_negatives": 0,
                    "quality_labels": Counter(),
                },
            )
            symbol_state["trades"] += 1
            symbol_state["net_pnl"] += record.pnl
            symbol_state["quality_labels"][record.quality_label] += 1
            if record.pnl > 0.0:
                symbol_state["wins"] += 1
            elif record.pnl < 0.0:
                symbol_state["losses"] += 1
            if record.trade_bucket == "positive_anchor":
                symbol_state["positive_anchors"] += 1
            elif record.trade_bucket == "hard_negative":
                symbol_state["hard_negatives"] += 1

        by_symbol: dict[str, dict[str, Any]] = {}
        for symbol, state in symbol_payload.items():
            trades = int(state.get("trades", 0) or 0)
            wins = int(state.get("wins", 0) or 0)
            by_symbol[symbol] = {
                "trades": trades,
                "wins": wins,
                "losses": int(state.get("losses", 0) or 0),
                "net_pnl": round(float(state.get("net_pnl", 0.0) or 0.0), 2),
                "win_rate": round((wins / trades) * 100.0, 2) if trades > 0 else 0.0,
                "positive_anchors": int(state.get("positive_anchors", 0) or 0),
                "hard_negatives": int(state.get("hard_negatives", 0) or 0),
                "quality_labels": dict(state.get("quality_labels", Counter())),
            }

        summary = {
            "status": "ok",
            "window_days": self.rolling_days,
            "generated_at": datetime.now().isoformat(),
            "total_closed_trades": total_closed,
            "net_pnl": round(net_pnl, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "positive_anchors": positive_anchors,
            "hard_negatives": hard_negatives,
            "quality_labels": dict(label_counter),
            "nemesis_mix": dict(nemesis_counter),
            "hard_negative_mix": self._build_hard_negative_mix(label_counter, total_closed),
            "by_symbol": by_symbol,
            "ga_runtime_hints": self._build_runtime_hints(by_symbol=by_symbol, total_closed_trades=total_closed),
        }
        self.latest_summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary

    def get_stats(self) -> dict[str, Any]:
        """Retourne un etat compact exploitable par l'API.

        Returns:
            dict[str, Any]: Statistiques de stockage et de fenetre.
        """

        summary = self.get_summary()
        return {
            "data_dir": str(self.data_dir),
            "window_days": self.rolling_days,
            "records_loaded": len(self._records),
            "latest_summary_path": str(self.latest_summary_path),
            "summary": summary,
        }

    def _build_record(
        self,
        *,
        symbol: str,
        action: str,
        price: float,
        volume: float,
        pnl: float,
        indicators: dict[str, Any],
        observation: dict[str, Any],
        next_observation: dict[str, Any],
        metadata: dict[str, Any],
        timestamp: str | None,
    ) -> LiveTradeReviewRecord:
        """Normalise un trade ferme en enregistrement stable.

        Args:
            symbol (str): Symbole du trade.
            action (str): Direction du trade.
            price (float): Prix de cloture.
            volume (float): Volume du trade.
            pnl (float): Profit ou perte realise.
            indicators (dict[str, Any]): Indicateurs fournis.
            observation (dict[str, Any]): Observation principale.
            next_observation (dict[str, Any]): Observation suivante.
            metadata (dict[str, Any]): Metadonnees du trade.
            timestamp (str | None): Horodatage de cloture.

        Returns:
            LiveTradeReviewRecord: Enregistrement complet et score.
        """

        closed_at = str(
            metadata.get("exit_time")
            or timestamp
            or datetime.now().isoformat()
        )
        entry_price = self._safe_float(metadata.get("entry_price"))
        exit_price = self._safe_float(metadata.get("exit_price"), fallback=price)
        directional_return_pct = self._safe_float(
            metadata.get("directional_return_pct"),
            fallback=self._compute_directional_return_pct(
                action=action,
                entry_price=entry_price,
                exit_price=exit_price,
            ),
        )
        duration_minutes = self._safe_float(
            metadata.get("duration_minutes"),
            fallback=self._compute_duration_minutes(
                entry_time=metadata.get("entry_time"),
                exit_time=closed_at,
            ),
        )
        close_reason = self._safe_text(metadata.get("close_reason") or "unknown")
        quality_label = self._classify_trade_quality(
            pnl=pnl,
            close_reason=close_reason,
            metadata=metadata,
        )
        if quality_label == "good_trade":
            trade_bucket = "positive_anchor"
        elif quality_label in {
            "bad_trade",
            "liquidity_trap",
            "range_entry_loss",
            "bad_runner_exit",
            "bad_pyramid_exit",
            "hard_stop_exit",
        }:
            trade_bucket = "hard_negative"
        else:
            trade_bucket = "neutral"

        metadata_payload = {
            **metadata,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "volume": self._safe_float(metadata.get("volume"), fallback=volume),
            "indicators": self._sanitize_mapping(indicators),
            "observation": self._sanitize_mapping(observation),
            "next_observation": self._sanitize_mapping(next_observation),
        }
        return LiveTradeReviewRecord(
            recorded_at=datetime.now().isoformat(),
            closed_at=closed_at,
            symbol=self._safe_text(symbol).upper() or "UNKNOWN",
            action=self._safe_text(action).upper() or "UNKNOWN",
            pnl=round(self._safe_float(pnl), 4),
            directional_return_pct=round(directional_return_pct, 6),
            close_reason=close_reason,
            quality_label=quality_label,
            trade_bucket=trade_bucket,
            nemesis_type=self._safe_text(metadata.get("nemesis_type")).upper(),
            context_label=self._safe_text(metadata.get("context_label")).upper(),
            strategy_family=self._safe_text(metadata.get("strategy_family")).lower(),
            live_engine=self._safe_text(metadata.get("live_engine")).lower(),
            live_champion_id=self._safe_text(metadata.get("live_champion_id")),
            ticket=self._safe_int(metadata.get("ticket")),
            duration_minutes=round(duration_minutes, 3),
            metadata=self._sanitize_mapping(metadata_payload),
        )

    def _classify_trade_quality(
        self,
        *,
        pnl: float,
        close_reason: str,
        metadata: dict[str, Any],
    ) -> str:
        """Attribue une etiquette metier a un trade ferme.

        Args:
            pnl (float): Profit ou perte realise.
            close_reason (str): Raison normalisee de cloture.
            metadata (dict[str, Any]): Metadonnees enrichies du trade.

        Returns:
            str: Etiquette finale retenue.
        """

        nemesis_type = self._safe_text(metadata.get("nemesis_type")).upper()
        if self._safe_bool(metadata.get("hard_stop_exit")) or close_reason == "stop_loss":
            return "hard_stop_exit"
        if nemesis_type == "LIQUIDITY_TRAP":
            return "liquidity_trap"
        if self._safe_bool(metadata.get("bad_runner_exit")):
            return "bad_runner_exit"
        if self._safe_bool(metadata.get("bad_pyramid_exit")):
            return "bad_pyramid_exit"
        if pnl < 0.0 and (
            self._safe_bool(metadata.get("range_context"))
            or self._safe_text(metadata.get("context_label")).upper() in {"RANGING", "RANGE", "NEUTRAL"}
        ):
            return "range_entry_loss"
        if pnl > 0.0:
            return "good_trade"
        if pnl < 0.0:
            return "bad_trade"
        return "flat_trade"

    def _build_hard_negative_mix(
        self,
        label_counter: Counter[str],
        total_closed: int,
    ) -> dict[str, Any]:
        """Construit le mix de trades negatifs pour le replay nocturne.

        Args:
            label_counter (Counter[str]): Repartition des etiquettes.
            total_closed (int): Nombre total de trades fermes.

        Returns:
            dict[str, Any]: Mix brut et parts associees.
        """

        hard_negative_labels = [
            "liquidity_trap",
            "range_entry_loss",
            "bad_runner_exit",
            "bad_pyramid_exit",
            "hard_stop_exit",
            "bad_trade",
        ]
        mix = {
            label: int(label_counter.get(label, 0) or 0)
            for label in hard_negative_labels
            if int(label_counter.get(label, 0) or 0) > 0
        }
        shares = {
            f"{label}_share": round((count / total_closed), 4) if total_closed > 0 else 0.0
            for label, count in mix.items()
        }
        return {"counts": mix, **shares}

    def _build_runtime_hints(
        self,
        *,
        by_symbol: dict[str, dict[str, Any]],
        total_closed_trades: int,
    ) -> dict[str, Any]:
        """Produit des hints de mutation runtime a partir des trades fermes.

        Args:
            by_symbol (dict[str, dict[str, Any]]): Resume des performances par symbole.
            total_closed_trades (int): Taille totale de la fenetre.

        Returns:
            dict[str, Any]: Hints de promotion, de demotion et de maintien.
        """

        promote_symbols: list[str] = []
        demote_symbols: list[str] = []
        hold_symbols: list[str] = []

        for symbol, payload in by_symbol.items():
            trades = int(payload.get("trades", 0) or 0)
            net_pnl = self._safe_float(payload.get("net_pnl"))
            win_rate = self._safe_float(payload.get("win_rate"))
            hard_negatives = int(payload.get("hard_negatives", 0) or 0)
            positive_anchors = int(payload.get("positive_anchors", 0) or 0)

            if trades < 2:
                hold_symbols.append(symbol)
                continue
            if net_pnl > 0.0 and positive_anchors >= 1 and win_rate >= 50.0:
                promote_symbols.append(symbol)
            elif net_pnl < 0.0 and (hard_negatives >= 2 or win_rate < 35.0):
                demote_symbols.append(symbol)
            else:
                hold_symbols.append(symbol)

        return {
            "eligible": total_closed_trades >= self.min_trades_for_hints,
            "min_trades_for_hints": self.min_trades_for_hints,
            "promote_symbols": sorted(promote_symbols),
            "demote_symbols": sorted(demote_symbols),
            "hold_symbols": sorted(hold_symbols),
        }

    def _load_recent_records(self) -> list[LiveTradeReviewRecord]:
        """Recharge les revues recentes depuis le disque.

        Returns:
            list[LiveTradeReviewRecord]: Historique glissant local.
        """

        cutoff = datetime.now() - timedelta(days=self.rolling_days + 1)
        records: list[LiveTradeReviewRecord] = []
        for path in sorted(self.data_dir.glob("live_trade_review_*.jsonl")):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for raw_line in handle:
                        line = raw_line.strip()
                        if not line:
                            continue
                        payload = json.loads(line)
                        closed_at = self._parse_iso(payload.get("closed_at"))
                        if closed_at is not None and closed_at < cutoff:
                            continue
                        records.append(LiveTradeReviewRecord(**payload))
            except Exception as exc:
                logger.warning("Lecture d'une revue live ignoree (%s): %s", path, exc)
        return records

    def _append_record(self, record: LiveTradeReviewRecord) -> None:
        """Persiste une revue dans le journal quotidien.

        Args:
            record (LiveTradeReviewRecord): Revue a ajouter.
        """

        file_token = datetime.now().strftime("%Y%m%d")
        output_path = self.data_dir / f"live_trade_review_{file_token}.jsonl"
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def _trim_records(self) -> None:
        """Garde uniquement la fenetre glissante utile en memoire."""

        cutoff = datetime.now() - timedelta(days=self.rolling_days + 1)
        self._records = [
            record
            for record in self._records
            if (self._parse_iso(record.closed_at) or self._parse_iso(record.recorded_at) or datetime.now())
            >= cutoff
        ]

    @staticmethod
    def _safe_float(raw_value: Any, fallback: float = 0.0) -> float:
        """Normalise une valeur numerique flottante.

        Args:
            raw_value (Any): Valeur brute.
            fallback (float): Valeur de repli.

        Returns:
            float: Valeur convertie.
        """

        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return float(fallback)

    @staticmethod
    def _safe_int(raw_value: Any) -> int | None:
        """Normalise un entier facultatif.

        Args:
            raw_value (Any): Valeur brute.

        Returns:
            int | None: Valeur convertie ou ``None``.
        """

        try:
            return int(raw_value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_text(raw_value: Any) -> str:
        """Normalise un texte libre.

        Args:
            raw_value (Any): Valeur brute.

        Returns:
            str: Texte nettoye.
        """

        return str(raw_value or "").strip()

    @staticmethod
    def _safe_bool(raw_value: Any) -> bool:
        """Normalise un booleen permissif.

        Args:
            raw_value (Any): Valeur brute.

        Returns:
            bool: Valeur booleenne normalisee.
        """

        if isinstance(raw_value, bool):
            return raw_value
        return str(raw_value or "").strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _parse_iso(raw_value: Any) -> datetime | None:
        """Convertit une date ISO en objet ``datetime``.

        Args:
            raw_value (Any): Horodatage brut.

        Returns:
            datetime | None: Date convertie ou ``None``.
        """

        text = str(raw_value or "").strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None

    @classmethod
    def _compute_duration_minutes(cls, entry_time: Any, exit_time: Any) -> float:
        """Calcule une duree de trade en minutes.

        Args:
            entry_time (Any): Horodatage d'ouverture.
            exit_time (Any): Horodatage de cloture.

        Returns:
            float: Duree en minutes, ou ``0.0`` si inconnue.
        """

        entry_dt = cls._parse_iso(entry_time)
        exit_dt = cls._parse_iso(exit_time)
        if entry_dt is None or exit_dt is None:
            return 0.0
        return max(0.0, (exit_dt - entry_dt).total_seconds() / 60.0)

    @staticmethod
    def _compute_directional_return_pct(
        *,
        action: str,
        entry_price: float,
        exit_price: float,
    ) -> float:
        """Calcule le rendement signe d'un trade ferme.

        Args:
            action (str): Direction initiale du trade.
            entry_price (float): Prix d'ouverture.
            exit_price (float): Prix de cloture.

        Returns:
            float: Rendement en pourcentage.
        """

        if entry_price <= 0.0 or exit_price <= 0.0:
            return 0.0
        raw_return = ((exit_price - entry_price) / entry_price) * 100.0
        if str(action or "").strip().upper() == "SELL":
            raw_return *= -1.0
        return raw_return

    @classmethod
    def _sanitize_mapping(cls, payload: dict[str, Any]) -> dict[str, Any]:
        """Rend une charge utile sure pour la persistence JSON.

        Args:
            payload (dict[str, Any]): Charge brute.

        Returns:
            dict[str, Any]: Charge nettoyee.
        """

        sanitized: dict[str, Any] = {}
        for key, value in dict(payload or {}).items():
            sanitized[str(key)] = cls._sanitize_value(value)
        return sanitized

    @classmethod
    def _sanitize_value(cls, value: Any) -> Any:
        """Normalise une valeur libre pour la persistence JSON.

        Args:
            value (Any): Valeur brute.

        Returns:
            Any: Valeur serialisable.
        """

        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return cls._sanitize_mapping(value)
        if isinstance(value, (list, tuple, set)):
            return [cls._sanitize_value(item) for item in value]
        if hasattr(value, "isoformat") and callable(value.isoformat):
            try:
                return value.isoformat()
            except Exception:
                return str(value)
        return str(value)
