"""Gere la promotion des challengers vers les champions live."""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from eva_lab.training_utils import (
    get_horizon_timeframe,
    infer_family_from_symbols,
    resolve_feature_profile,
    resolve_symbol_overrides,
    resolve_training_symbols,
)
from eva_lab.training_status import load_latest_terminal_summary

logger = logging.getLogger(__name__)


class ChampionPromoter:
    """Centralise la promotion et la resolution des champions live.

    Cette classe evite que le moteur live charge un checkpoint intermediaire
    tant qu'aucune promotion n'a ete validee par l'Arena.
    """

    def __init__(
        self,
        weights_dir: str = "data/muzero/weights",
        results_dir: str = "data/muzero/results",
    ) -> None:
        """Initialise les dossiers de poids et de manifestes.

        Args:
            weights_dir (str): Dossier contenant les checkpoints MuZero.
            results_dir (str): Dossier contenant les rapports et manifestes live.
        """
        self.weights_dir = Path(weights_dir)
        self.results_dir = Path(results_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def normalize_live_selection_policy(
        raw_policy: str | None,
        default: str = "champion_only",
    ) -> str:
        """Normalise une politique de selection live.

        Args:
            raw_policy (str | None): Valeur brute a normaliser.
            default (str): Politique de repli si la valeur est absente.

        Returns:
            str: Politique de selection live autorisee.
        """
        raw_value = str(raw_policy or default).strip().lower()
        allowed = {"champion_only", "champion_then_latest", "checkpoint_preview"}
        if raw_value not in allowed:
            logger.warning(
                "Politique live inconnue '%s'. Repli sur 'champion_only'.",
                raw_policy,
            )
            return "champion_only"
        return raw_value

    @classmethod
    def get_live_selection_policy(cls) -> str:
        """Retourne la politique live active.

        Returns:
            str: Politique de selection live.
        """
        return cls.normalize_live_selection_policy(
            os.getenv("MUZERO_LIVE_SELECTION_POLICY", "champion_only"),
            default="champion_only",
        )

    @staticmethod
    def normalize_engine_name(engine: str | None) -> str:
        """Normalise le nom d'un moteur de modeles.

        Args:
            engine (str | None): Nom brut du moteur.

        Returns:
            str: Nom stable du moteur (`muzero` ou `dreamer`).
        """
        normalized = str(engine or "muzero").strip().lower()
        if normalized in {"dreamer", "dreamerv3", "dreamer_v3"}:
            return "dreamer"
        return "muzero"

    def get_manifest_path(self, horizon: str, engine: str = "muzero") -> Path:
        """Construit le chemin du manifeste d'un horizon.

        Args:
            horizon (str): Horizon concerne.
            engine (str): Moteur concerne.

        Returns:
            Path: Chemin du manifeste JSON.
        """
        normalized_engine = self.normalize_engine_name(engine)
        if normalized_engine == "muzero":
            return self.results_dir / f"champion_{horizon.lower()}.json"
        return self.results_dir / f"champion_{normalized_engine}_{horizon.lower()}.json"

    def get_arena_report_path(self, horizon: str, engine: str = "muzero") -> Path:
        """Construit le chemin du rapport Arena d'un horizon.

        Args:
            horizon (str): Horizon concerne.
            engine (str): Moteur concerne.

        Returns:
            Path: Chemin du rapport Arena.
        """
        normalized_engine = self.normalize_engine_name(engine)
        if normalized_engine == "muzero":
            return self.results_dir / f"arena_{horizon.lower()}_latest.json"
        return self.results_dir / f"arena_{normalized_engine}_{horizon.lower()}_latest.json"

    def get_champion_path(self, horizon: str, engine: str = "muzero") -> Path:
        """Retourne le chemin du champion live pour un moteur.

        Args:
            horizon (str): Horizon strategique.
            engine (str): Moteur cible.

        Returns:
            Path: Chemin du checkpoint champion.
        """
        normalized_engine = self.normalize_engine_name(engine)
        return self.weights_dir / f"{normalized_engine}_champion_{horizon.lower()}.pkl"

    def get_latest_model_path(self, horizon: str, engine: str = "muzero") -> Path:
        """Retourne le chemin du dernier modele latest d'un moteur.

        Args:
            horizon (str): Horizon strategique.
            engine (str): Moteur cible.

        Returns:
            Path: Chemin du modele latest.
        """
        normalized_engine = self.normalize_engine_name(engine)
        return self.weights_dir / f"{normalized_engine}_{horizon.lower()}_latest.pkl"

    def get_checkpoint_glob(self, horizon: str, engine: str = "muzero") -> str:
        """Construit le motif de recherche des checkpoints intermediaires.

        Args:
            horizon (str): Horizon strategique.
            engine (str): Moteur cible.

        Returns:
            str: Motif `glob` compatible `Path.glob`.
        """
        normalized_engine = self.normalize_engine_name(engine)
        return f"{normalized_engine}_{horizon.lower()}_ckpt_*.pkl"

    def get_engine_label(self, engine: str, variant: str = "champion") -> str:
        """Retourne un libelle lisible pour un moteur live.

        Args:
            engine (str): Moteur cible.
            variant (str): Variante de rendu (`champion`, `latest`, `preview`, `blocked`).

        Returns:
            str: Libelle stable pour l'UI et les logs.
        """
        normalized_engine = self.normalize_engine_name(engine)
        if normalized_engine == "dreamer":
            base_label = "DreamerV3"
        else:
            base_label = "MuZero JAX"
        suffix_map = {
            "champion": "Champion",
            "latest": "Latest",
            "preview": "Preview",
            "blocked": "Bloque",
        }
        suffix = suffix_map.get(str(variant or "champion").lower(), "Champion")
        return f"{base_label} {suffix}".strip()

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        """Convertit une valeur quelconque en flottant robuste.

        Args:
            value (Any): Valeur brute a convertir.
            default (float): Valeur de repli si la conversion echoue.

        Returns:
            float: Valeur numerique exploitable.
        """
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _read_boolean_env(name: str, default: bool) -> bool:
        """Interprete une variable d'environnement booleenne.

        Args:
            name (str): Nom de la variable d'environnement.
            default (bool): Valeur de repli si la variable est absente.

        Returns:
            bool: Valeur booleenne normalisee.
        """
        raw_value = os.getenv(name)
        if raw_value is None:
            return default
        return raw_value.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def normalize_gate_profile(raw_profile: str | None, default: str = "standard") -> str:
        """Normalise un profil de gate de promotion.

        Args:
            raw_profile (str | None): Profil brut a normaliser.
            default (str): Profil de repli si absent.

        Returns:
            str: Profil de gate supporte.
        """
        normalized = str(raw_profile or default).strip().lower()
        if normalized in {"gold", "gold_only", "gold-demo"}:
            return "gold_demo"
        if normalized not in {"standard", "gold_demo"}:
            return default
        return normalized

    def _resolve_gate_profile(self, *payloads: dict[str, Any] | None) -> str:
        """Determine le profil de gate a utiliser.

        Args:
            *payloads (dict[str, Any] | None): Artefacts potentiellement
                porteurs du profil de gate.

        Returns:
            str: Profil de gate normalise.
        """
        for payload in payloads:
            current = dict(payload or {})
            direct_profile = str(current.get("gate_profile") or "").strip()
            if direct_profile:
                return self.normalize_gate_profile(direct_profile)
            promotion_gate = dict(current.get("promotion_gate") or {})
            embedded_profile = str(promotion_gate.get("gate_profile") or "").strip()
            if embedded_profile:
                return self.normalize_gate_profile(embedded_profile)
        env_profile = (
            str(os.getenv("TRAINING_GATE_PROFILE", "")).strip()
            or str(os.getenv("MUZERO_PROMOTION_GATE_PROFILE", "")).strip()
        )
        return self.normalize_gate_profile(env_profile or "standard")

    def get_promotion_thresholds(self, gate_profile: str | None = None) -> dict[str, Any]:
        """Retourne les seuils minimums de promotion live.

        Returns:
            dict[str, Any]: Configuration active de filtrage des champions.
        """
        normalized_gate_profile = self.normalize_gate_profile(gate_profile or "standard")
        if normalized_gate_profile == "gold_demo":
            return {
                "gate_profile": normalized_gate_profile,
                "require_positive_metrics": True,
                "min_win_rate": 55.0,
                "min_return_pct": 0.0,
                "min_profit_factor": 1.10,
                "min_total_trades": 12.0,
                "min_eval_games": 12.0,
                "min_eval_symbols": 1.0,
                "min_expectancy_pct": 0.0,
                "max_drawdown_pct": 3.0,
                "min_positive_episode_rate": 12.0,
                "min_net_realized_pct": 0.0,
                "min_long_entry_share": 0.15,
                "min_short_entry_share": 0.15,
                "max_directional_imbalance": 0.70,
                "min_split_efficiency": 0.40,
                "min_pyramid_efficiency": 0.40,
                "min_slbe_capture_rate": 0.35,
                "max_hold_drag_score": 0.55,
                "min_close_quality_score": 0.40,
            }

        return {
            "gate_profile": normalized_gate_profile,
            "require_positive_metrics": self._read_boolean_env(
                "MUZERO_PROMOTION_REQUIRE_POSITIVE_METRICS",
                True,
            ),
            "min_win_rate": self._to_float(
                os.getenv("MUZERO_PROMOTION_MIN_WIN_RATE", "55.0"),
                default=55.0,
            ),
            "min_return_pct": self._to_float(
                os.getenv("MUZERO_PROMOTION_MIN_RETURN_PCT", "0.0"),
                default=0.0,
            ),
            "min_profit_factor": self._to_float(
                os.getenv("MUZERO_PROMOTION_MIN_PROFIT_FACTOR", "1.15"),
                default=1.15,
            ),
            "min_total_trades": self._to_float(
                os.getenv("MUZERO_PROMOTION_MIN_TOTAL_TRADES", "24"),
                default=24.0,
            ),
            "min_eval_games": self._to_float(
                os.getenv("MUZERO_PROMOTION_MIN_EVAL_GAMES", "12"),
                default=12.0,
            ),
            "min_eval_symbols": self._to_float(
                os.getenv("MUZERO_PROMOTION_MIN_EVAL_SYMBOLS", "3"),
                default=3.0,
            ),
            "min_expectancy_pct": self._to_float(
                os.getenv("MUZERO_PROMOTION_MIN_EXPECTANCY_PCT", "0.0"),
                default=0.0,
            ),
            "max_drawdown_pct": self._to_float(
                os.getenv("MUZERO_PROMOTION_MAX_DRAWDOWN_PCT", "3.5"),
                default=3.5,
            ),
            "min_positive_episode_rate": self._to_float(
                os.getenv("MUZERO_PROMOTION_MIN_POSITIVE_EPISODE_RATE", "15.0"),
                default=15.0,
            ),
            "min_net_realized_pct": self._to_float(
                os.getenv("MUZERO_PROMOTION_MIN_NET_REALIZED_PCT", "0.0"),
                default=0.0,
            ),
            "min_long_entry_share": self._to_float(
                os.getenv("MUZERO_PROMOTION_MIN_LONG_ENTRY_SHARE", "0.20"),
                default=0.20,
            ),
            "min_short_entry_share": self._to_float(
                os.getenv("MUZERO_PROMOTION_MIN_SHORT_ENTRY_SHARE", "0.20"),
                default=0.20,
            ),
            "max_directional_imbalance": self._to_float(
                os.getenv("MUZERO_PROMOTION_MAX_DIRECTIONAL_IMBALANCE", "0.60"),
                default=0.60,
            ),
            "min_split_efficiency": self._to_float(
                os.getenv("MUZERO_PROMOTION_MIN_SPLIT_EFFICIENCY", "0.45"),
                default=0.45,
            ),
            "min_pyramid_efficiency": self._to_float(
                os.getenv("MUZERO_PROMOTION_MIN_PYRAMID_EFFICIENCY", "0.45"),
                default=0.45,
            ),
            "min_slbe_capture_rate": self._to_float(
                os.getenv("MUZERO_PROMOTION_MIN_SLBE_CAPTURE_RATE", "0.40"),
                default=0.40,
            ),
            "max_hold_drag_score": self._to_float(
                os.getenv("MUZERO_PROMOTION_MAX_HOLD_DRAG_SCORE", "0.40"),
                default=0.40,
            ),
            "min_close_quality_score": self._to_float(
                os.getenv("MUZERO_PROMOTION_MIN_CLOSE_QUALITY_SCORE", "0.45"),
                default=0.45,
            ),
        }

    def _derive_failure_mode(
        self,
        *,
        reason: str,
        metrics: dict[str, Any],
        checks: dict[str, bool],
    ) -> str:
        """Derive un mode d'echec stable pour l'observabilite V3.

        Args:
            reason (str): Raison brute retournee par la gate.
            metrics (dict[str, Any]): Metriques consolidees du challenger.
            checks (dict[str, bool]): Resultat de chaque controle unitaire.

        Returns:
            str: Code d'echec stable et exploitable dans Nexus/Core.
        """
        directional_bias = str(metrics.get("directional_bias") or "").lower()
        if not checks.get("directional_entries", True):
            return "inactive"
        if directional_bias == "sell_heavy":
            return "sell_heavy"
        if directional_bias == "buy_heavy":
            return "buy_heavy"
        if not checks.get("total_trades", True) or not checks.get("evaluation_games", True):
            return "insufficient_sample"
        if not checks.get("return_pct", True) or not checks.get("net_realized_pct", True) or not checks.get("profit_factor", True):
            return "unprofitable"
        if not checks.get("close_quality_score", True) or not checks.get("split_efficiency", True):
            return "bad_exit"
        if reason in {"directional_balance", "directional_entries"}:
            return "inactive" if reason == "directional_entries" else directional_bias or "inactive"
        return str(reason or "blocked")

    def evaluate_promotion_gate(
        self,
        battle_report: dict[str, Any] | None,
        gate_profile: str | None = None,
    ) -> dict[str, Any]:
        """Valide l'eligibilite d'un challenger avant de l'exposer au live.

        Args:
            battle_report (dict[str, Any] | None): Rapport Arena du challenger.
            gate_profile (str | None): Profil de gate explicite a appliquer.

        Returns:
            dict[str, Any]: Verdict detaille avec metriques, seuils et raison.
        """
        normalized_gate_profile = self.normalize_gate_profile(
            gate_profile or self._resolve_gate_profile(battle_report),
        )
        thresholds = self.get_promotion_thresholds(normalized_gate_profile)
        if not battle_report:
            return {
                "allowed": False,
                "status": "blocked",
                "reason": "missing_battle_report",
                "gate_profile": normalized_gate_profile,
                "failure_mode": "insufficient_sample",
                "checks": {"arena_victory": False},
                "thresholds": thresholds,
                "metrics": {
                    "win_rate": 0.0,
                    "return_pct": 0.0,
                    "net_realized_pct": 0.0,
                    "profit_factor": 0.0,
                    "total_trades": 0.0,
                    "evaluation_games": 0.0,
                    "evaluation_symbols": 0.0,
                    "expectancy_pct": 0.0,
                    "max_drawdown_pct": 100.0,
                    "positive_episode_rate": 0.0,
                    "long_entries": 0.0,
                    "short_entries": 0.0,
                    "long_entry_share": 0.0,
                    "short_entry_share": 0.0,
                    "directional_imbalance": 1.0,
                },
            }

        challenger_metrics = (battle_report.get("challenger", {}) or {}).get("metrics", {}) or {}
        mechanics_metrics = dict(challenger_metrics.get("metrics_by_position_mechanics") or {})
        validation = battle_report.get("validation", {}) or {}
        win_rate = self._to_float(challenger_metrics.get("win_rate"))
        return_pct = self._to_float(challenger_metrics.get("return_pct"))
        net_realized_pct = self._to_float(challenger_metrics.get("net_realized_pct"))
        profit_factor = self._to_float(challenger_metrics.get("profit_factor"))
        total_trades = self._to_float(challenger_metrics.get("total_trades"))
        evaluation_games = self._to_float(challenger_metrics.get("evaluation_games"))
        evaluation_symbols = self._to_float(challenger_metrics.get("evaluation_symbols"))
        expectancy_pct = self._to_float(challenger_metrics.get("expectancy_pct"))
        max_drawdown_pct = self._to_float(challenger_metrics.get("max_drawdown_pct"), default=100.0)
        positive_episode_rate = self._to_float(challenger_metrics.get("positive_episode_rate"))
        long_entries = self._to_float(challenger_metrics.get("long_entries"))
        short_entries = self._to_float(challenger_metrics.get("short_entries"))
        long_entry_share = self._to_float(challenger_metrics.get("long_entry_share"))
        short_entry_share = self._to_float(challenger_metrics.get("short_entry_share"))
        directional_imbalance = self._to_float(
            challenger_metrics.get("directional_imbalance"),
            default=1.0,
        )
        split_efficiency = self._to_float(mechanics_metrics.get("split_efficiency"))
        pyramid_efficiency = self._to_float(mechanics_metrics.get("pyramid_efficiency"))
        slbe_capture_rate = self._to_float(mechanics_metrics.get("slbe_capture_rate"))
        hold_drag_score = self._to_float(mechanics_metrics.get("hold_drag_score"))
        close_quality_score = self._to_float(mechanics_metrics.get("close_quality_score"))
        split_executed = self._to_float(mechanics_metrics.get("split_executed"))
        pyramids_opened = self._to_float(mechanics_metrics.get("pyramids_opened"))
        slbe_triggered = self._to_float(mechanics_metrics.get("slbe_triggered"))
        close_events = self._to_float(mechanics_metrics.get("close_winner_count")) + self._to_float(
            mechanics_metrics.get("close_loser_count")
        )
        checks = {
            "arena_victory": str(battle_report.get("outcome", "")).upper() == "VICTORY",
            "validation_sample": bool(validation.get("sample_size_ok", False)),
            "evaluation_games": evaluation_games >= thresholds["min_eval_games"],
            "evaluation_symbols": evaluation_symbols >= thresholds["min_eval_symbols"],
            "total_trades": total_trades >= thresholds["min_total_trades"],
            "max_drawdown_pct": max_drawdown_pct <= thresholds["max_drawdown_pct"],
            "directional_entries": (long_entries + short_entries) > 0,
        }

        if thresholds["require_positive_metrics"]:
            checks["win_rate"] = win_rate >= thresholds["min_win_rate"]
            checks["return_pct"] = return_pct > thresholds["min_return_pct"]
            checks["net_realized_pct"] = net_realized_pct >= thresholds["min_net_realized_pct"]
            checks["profit_factor"] = profit_factor > thresholds["min_profit_factor"]
            checks["expectancy_pct"] = expectancy_pct >= thresholds["min_expectancy_pct"]
            checks["positive_episode_rate"] = (
                positive_episode_rate >= thresholds["min_positive_episode_rate"]
            )
            checks["directional_balance"] = (
                long_entry_share >= thresholds["min_long_entry_share"]
                and short_entry_share >= thresholds["min_short_entry_share"]
                and directional_imbalance <= thresholds["max_directional_imbalance"]
            )

        checks["hold_drag_score"] = hold_drag_score <= thresholds["max_hold_drag_score"]
        checks["split_efficiency"] = (
            split_executed <= 0 or split_efficiency >= thresholds["min_split_efficiency"]
        )
        checks["pyramid_efficiency"] = (
            pyramids_opened <= 0 or pyramid_efficiency >= thresholds["min_pyramid_efficiency"]
        )
        checks["slbe_capture_rate"] = (
            slbe_triggered <= 0 or slbe_capture_rate >= thresholds["min_slbe_capture_rate"]
        )
        checks["close_quality_score"] = (
            close_events <= 0 or close_quality_score >= thresholds["min_close_quality_score"]
        )

        failure_reason = next((name for name, passed in checks.items() if not passed), "eligible")
        metrics_payload = {
            "win_rate": win_rate,
            "return_pct": return_pct,
            "net_realized_pct": net_realized_pct,
            "profit_factor": profit_factor,
            "total_trades": total_trades,
            "evaluation_games": evaluation_games,
            "evaluation_symbols": evaluation_symbols,
            "expectancy_pct": expectancy_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "positive_episode_rate": positive_episode_rate,
            "long_entries": long_entries,
            "short_entries": short_entries,
            "long_entry_share": long_entry_share,
            "short_entry_share": short_entry_share,
            "directional_imbalance": directional_imbalance,
            "directional_bias": str(challenger_metrics.get("directional_bias") or "inactive"),
            "metrics_by_position_mechanics": mechanics_metrics,
            "split_efficiency": split_efficiency,
            "pyramid_efficiency": pyramid_efficiency,
            "slbe_capture_rate": slbe_capture_rate,
            "hold_drag_score": hold_drag_score,
            "close_quality_score": close_quality_score,
            "feature_profile": challenger_metrics.get("feature_profile"),
            "dataset_id": challenger_metrics.get("dataset_id"),
            "dataset_source": challenger_metrics.get("dataset_source"),
            "mechanics_profile_version": challenger_metrics.get("mechanics_profile_version"),
            "dataset_coverage": challenger_metrics.get("dataset_coverage") or {},
        }
        return {
            "allowed": all(checks.values()),
            "status": "eligible" if all(checks.values()) else "blocked",
            "reason": failure_reason,
            "gate_profile": normalized_gate_profile,
            "failure_mode": self._derive_failure_mode(
                reason=failure_reason,
                metrics=metrics_payload,
                checks=checks,
            ),
            "checks": checks,
            "thresholds": thresholds,
            "metrics": metrics_payload,
        }

    def resolve_promotion_gate(
        self,
        manifest: dict[str, Any] | None,
        arena_report: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Determine le verdict de promotion a partir des artefacts disponibles.

        Args:
            manifest (dict[str, Any] | None): Manifeste de promotion live.
            arena_report (dict[str, Any] | None): Rapport de run historique.

        Returns:
            dict[str, Any]: Verdict detaille de la barriere de promotion.
        """
        gate_profile = self._resolve_gate_profile(manifest, arena_report)
        if manifest and isinstance(manifest.get("promotion_gate"), dict):
            gate_payload = dict(manifest["promotion_gate"])
            gate_payload.setdefault("gate_profile", gate_profile)
            return gate_payload

        if manifest and isinstance(manifest.get("battle_report"), dict):
            return self.evaluate_promotion_gate(
                manifest["battle_report"],
                gate_profile=gate_profile,
            )

        if arena_report and isinstance(arena_report.get("battle_report"), dict):
            return self.evaluate_promotion_gate(
                arena_report["battle_report"],
                gate_profile=gate_profile,
            )

        return self.evaluate_promotion_gate(None, gate_profile=gate_profile)

    def load_manifest(self, horizon: str, engine: str = "muzero") -> dict[str, Any] | None:
        """Charge le manifeste d'un champion si disponible.

        Args:
            horizon (str): Horizon cible.
            engine (str): Moteur cible.

        Returns:
            dict[str, Any] | None: Manifeste charge ou ``None``.
        """
        manifest_path = self.get_manifest_path(horizon, engine=engine)
        if not manifest_path.exists():
            return None

        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Lecture manifeste champion impossible pour %s: %s", horizon, exc)
            return None

    def load_arena_report(self, horizon: str, engine: str = "muzero") -> dict[str, Any] | None:
        """Charge le dernier rapport Arena pour un horizon.

        Args:
            horizon (str): Horizon cible.
            engine (str): Moteur cible.

        Returns:
            dict[str, Any] | None: Rapport Arena ou ``None``.
        """
        report_path = self.get_arena_report_path(horizon, engine=engine)
        if not report_path.exists():
            return None

        try:
            return json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Lecture rapport Arena impossible pour %s: %s", horizon, exc)
            return None

    @staticmethod
    def _describe_path(path: Path | None) -> dict[str, Any]:
        """Retourne des metadonnees simples sur un fichier.

        Args:
            path (Path | None): Fichier a decrire.

        Returns:
            dict[str, Any]: Presence, taille et date de modification.
        """
        if path is None:
            return {
                "path": None,
                "exists": False,
                "size_bytes": None,
                "modified_at": None,
            }

        exists = path.exists()
        return {
            "path": str(path),
            "exists": exists,
            "size_bytes": path.stat().st_size if exists else None,
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat() if exists else None,
        }

    def get_latest_checkpoint_path(self, horizon: str, engine: str = "muzero") -> Path | None:
        """Retourne le dernier checkpoint intermediaire d'un horizon.

        Args:
            horizon (str): Horizon cible.
            engine (str): Moteur cible.

        Returns:
            Path | None: Dernier checkpoint ou ``None``.
        """
        checkpoint_candidates = sorted(
            self.weights_dir.glob(self.get_checkpoint_glob(horizon, engine=engine)),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if checkpoint_candidates:
            return checkpoint_candidates[0]
        return None

    @staticmethod
    def _deduplicate_symbols(symbols: list[str]) -> list[str]:
        """Supprime les doublons d'une liste de symboles en conservant l'ordre.

        Args:
            symbols (list[str]): Liste brute de symboles.

        Returns:
            list[str]: Liste dedoublonnee et nettoyee.
        """
        cleaned: list[str] = []
        seen: set[str] = set()
        for symbol in symbols:
            symbol_name = str(symbol or "").strip()
            if not symbol_name or symbol_name in seen:
                continue
            cleaned.append(symbol_name)
            seen.add(symbol_name)
        return cleaned

    def get_live_symbol_thresholds(self) -> dict[str, Any]:
        """Retourne les seuils de selection par symbole pour le live.

        Returns:
            dict[str, Any]: Seuils utilises pour filtrer les meilleurs actifs.
        """
        return {
            "top_symbols": int(os.getenv("MUZERO_LIVE_TOP_SYMBOLS", "5")),
            "min_symbol_eval_games": self._to_float(
                os.getenv("MUZERO_LIVE_MIN_SYMBOL_EVAL_GAMES", "2"),
                default=2.0,
            ),
            "min_symbol_trades": self._to_float(
                os.getenv("MUZERO_LIVE_MIN_SYMBOL_TRADES", "4"),
                default=4.0,
            ),
            "min_symbol_return_pct": self._to_float(
                os.getenv("MUZERO_LIVE_MIN_SYMBOL_RETURN_PCT", "0.0"),
                default=0.0,
            ),
            "min_symbol_net_realized_pct": self._to_float(
                os.getenv("MUZERO_LIVE_MIN_SYMBOL_NET_REALIZED_PCT", "0.0"),
                default=0.0,
            ),
            "min_symbol_profit_factor": self._to_float(
                os.getenv("MUZERO_LIVE_MIN_SYMBOL_PROFIT_FACTOR", "1.0"),
                default=1.0,
            ),
            "max_symbol_drawdown_pct": self._to_float(
                os.getenv("MUZERO_LIVE_MAX_SYMBOL_DRAWDOWN_PCT", "5.0"),
                default=5.0,
            ),
            "min_symbol_long_entry_share": self._to_float(
                os.getenv("MUZERO_LIVE_MIN_SYMBOL_LONG_ENTRY_SHARE", "0.10"),
                default=0.10,
            ),
            "min_symbol_short_entry_share": self._to_float(
                os.getenv("MUZERO_LIVE_MIN_SYMBOL_SHORT_ENTRY_SHARE", "0.10"),
                default=0.10,
            ),
            "max_symbol_directional_imbalance": self._to_float(
                os.getenv("MUZERO_LIVE_MAX_SYMBOL_DIRECTIONAL_IMBALANCE", "0.85"),
                default=0.85,
            ),
        }

    @staticmethod
    def _read_live_symbol_metrics(
        manifest: dict[str, Any] | None,
        arena_report: dict[str, Any] | None,
    ) -> dict[str, dict[str, Any]]:
        """Retourne les metriques par symbole disponibles pour le live.

        Args:
            manifest (dict[str, Any] | None): Manifeste live si present.
            arena_report (dict[str, Any] | None): Rapport Arena le plus recent.

        Returns:
            dict[str, dict[str, Any]]: Metriques brutes par symbole.
        """
        payloads = [
            ((manifest or {}).get("battle_report", {}) or {}),
            ((arena_report or {}).get("battle_report", {}) or {}),
        ]
        for battle_report in payloads:
            challenger = (battle_report.get("challenger", {}) or {})
            metrics = (challenger.get("metrics", {}) or {})
            by_symbol = metrics.get("metrics_by_symbol", {}) or {}
            if isinstance(by_symbol, dict) and by_symbol:
                return by_symbol
        return {}

    def _score_live_symbol(self, metrics: dict[str, Any]) -> float:
        """Calcule un score simple de priorisation live pour un symbole.

        Args:
            metrics (dict[str, Any]): Metriques consolidees du symbole.

        Returns:
            float: Score de priorite plus eleve = symbole plus interessant.
        """
        mechanics = dict(metrics.get("metrics_by_position_mechanics") or {})
        return (
            self._to_float(metrics.get("return_pct")) * 8.0
            + self._to_float(metrics.get("net_realized_pct")) * 0.35
            + max(0.0, self._to_float(metrics.get("profit_factor")) - 1.0) * 18.0
            + self._to_float(metrics.get("win_rate")) * 0.06
            + self._to_float(mechanics.get("close_quality_score")) * 6.0
            + self._to_float(mechanics.get("split_efficiency")) * 4.0
            + self._to_float(mechanics.get("pyramid_efficiency")) * 4.0
            + self._to_float(mechanics.get("slbe_capture_rate")) * 4.0
            - self._to_float(metrics.get("max_drawdown_pct")) * 1.8
            - self._to_float(metrics.get("directional_imbalance"), default=1.0) * 10.0
            - self._to_float(mechanics.get("hold_drag_score")) * 10.0
        )

    def rank_live_symbols(
        self,
        horizon: str,
        manifest: dict[str, Any] | None,
        arena_report: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Classe les meilleurs symboles live a partir des metriques Arena.

        Args:
            horizon (str): Horizon strategique cible.
            manifest (dict[str, Any] | None): Manifeste live courant.
            arena_report (dict[str, Any] | None): Rapport Arena le plus recent.

        Returns:
            dict[str, Any]: Classement detaille, seuils et source des symboles.
        """
        thresholds = self.get_live_symbol_thresholds()
        manual_symbols, manual_env = resolve_symbol_overrides(
            [
                f"MUZERO_LIVE_SYMBOLS_{horizon.upper()}",
                "MUZERO_LIVE_SYMBOLS",
            ]
        )
        top_limit = max(1, int(thresholds["top_symbols"]))
        if manual_symbols:
            manual_symbols = manual_symbols[:top_limit]
            return {
                "source": f"manual_env:{manual_env}",
                "symbols": manual_symbols,
                "eligible_symbols": [
                    {"symbol": symbol, "allowed": True, "reason": "manual_override", "score": None}
                    for symbol in manual_symbols
                ],
                "rejected_symbols": [],
                "metrics_by_symbol": {},
                "thresholds": thresholds,
            }

        raw_metrics_by_symbol = self._read_live_symbol_metrics(manifest, arena_report)
        if not raw_metrics_by_symbol:
            return {
                "source": "none",
                "symbols": [],
                "eligible_symbols": [],
                "rejected_symbols": [],
                "metrics_by_symbol": {},
                "thresholds": thresholds,
            }

        eligible: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        normalized_metrics: dict[str, dict[str, Any]] = {}
        promotion_thresholds = self.get_promotion_thresholds()
        for symbol, raw_metrics in raw_metrics_by_symbol.items():
            metrics = dict(raw_metrics or {})
            metrics["score"] = self._score_live_symbol(metrics)
            normalized_metrics[symbol] = metrics
            mechanics = dict(metrics.get("metrics_by_position_mechanics") or {})
            close_events = self._to_float(mechanics.get("close_winner_count")) + self._to_float(
                mechanics.get("close_loser_count")
            )
            checks = {
                "evaluation_games": self._to_float(metrics.get("evaluation_games")) >= thresholds["min_symbol_eval_games"],
                "total_trades": self._to_float(metrics.get("total_trades")) >= thresholds["min_symbol_trades"],
                "return_pct": self._to_float(metrics.get("return_pct")) >= thresholds["min_symbol_return_pct"],
                "net_realized_pct": self._to_float(metrics.get("net_realized_pct")) >= thresholds["min_symbol_net_realized_pct"],
                "profit_factor": self._to_float(metrics.get("profit_factor")) >= thresholds["min_symbol_profit_factor"],
                "max_drawdown_pct": self._to_float(metrics.get("max_drawdown_pct"), default=100.0) <= thresholds["max_symbol_drawdown_pct"],
                "long_entry_share": self._to_float(metrics.get("long_entry_share")) >= thresholds["min_symbol_long_entry_share"],
                "short_entry_share": self._to_float(metrics.get("short_entry_share")) >= thresholds["min_symbol_short_entry_share"],
                "directional_imbalance": self._to_float(
                    metrics.get("directional_imbalance"),
                    default=1.0,
                ) <= thresholds["max_symbol_directional_imbalance"],
                "hold_drag_score": self._to_float(
                    mechanics.get("hold_drag_score")
                ) <= promotion_thresholds["max_hold_drag_score"],
                "close_quality_score": (
                    close_events <= 0
                    or self._to_float(mechanics.get("close_quality_score"))
                    >= promotion_thresholds["min_close_quality_score"]
                ),
            }
            verdict = {
                "symbol": symbol,
                "allowed": all(checks.values()),
                "reason": next((name for name, passed in checks.items() if not passed), "eligible"),
                "score": metrics["score"],
                "checks": checks,
                "metrics": metrics,
            }
            if verdict["allowed"]:
                eligible.append(verdict)
            else:
                rejected.append(verdict)

        eligible.sort(key=lambda item: (-float(item["score"]), item["symbol"]))
        rejected.sort(key=lambda item: (-float(item["score"]), item["symbol"]))
        return {
            "source": "arena_symbol_metrics",
            "symbols": [item["symbol"] for item in eligible[:top_limit]],
            "eligible_symbols": eligible,
            "rejected_symbols": rejected,
            "metrics_by_symbol": normalized_metrics,
            "thresholds": thresholds,
        }

    def build_live_universe(self, horizon: str, engine: str = "muzero") -> dict[str, Any]:
        """Construit l'univers live recommande pour un horizon donne.

        Args:
            horizon (str): Horizon strategique cible.
            engine (str): Moteur cible.

        Returns:
            dict[str, Any]: Univers recommande, source et garde-fou associe.
        """
        horizon = horizon.lower()
        normalized_engine = self.normalize_engine_name(engine)
        manifest = self.load_manifest(horizon, engine=normalized_engine) or {}
        arena_report = self.load_arena_report(horizon, engine=normalized_engine) or {}
        promotion_gate = self.resolve_promotion_gate(manifest, arena_report)
        candidate_metrics = (
            (arena_report.get("battle_report", {}) or {}).get("challenger", {}) or {}
        ).get("metrics", {}) or {}
        family = str(
            manifest.get("family")
            or candidate_metrics.get("family")
            or infer_family_from_symbols(list((candidate_metrics.get("metrics_by_symbol") or {}).keys()))
            or "mixed"
        )
        feature_profile = str(
            manifest.get("feature_profile")
            or candidate_metrics.get("feature_profile")
            or resolve_feature_profile(horizon, family).get("profile_name")
            or "default"
        )
        max_symbols = int(os.getenv("MUZERO_LIVE_UNIVERSE_MAX_SYMBOLS", "12"))
        ranked_symbols = self.rank_live_symbols(horizon, manifest, arena_report)

        candidates: list[tuple[str, list[str]]] = []
        top_live_symbols = self._deduplicate_symbols(list(ranked_symbols.get("symbols", []) or []))
        candidates.append((str(ranked_symbols.get("source") or "top_live_symbols"), top_live_symbols))
        manifest_battle = manifest.get("battle_report", {}) or {}
        arena_battle = arena_report.get("battle_report", {}) or {}
        candidates.append(("manifest_eval_symbols", manifest_battle.get("eval_symbols", []) or []))
        candidates.append(("arena_eval_symbols", arena_battle.get("eval_symbols", []) or []))
        candidates.append(("arena_training_symbols", arena_report.get("symbols", []) or []))

        fallback_symbols = resolve_training_symbols(
            required_timeframes={get_horizon_timeframe(horizon)},
            max_symbols=max_symbols,
            override_env_names=[
                f"MUZERO_SYMBOLS_{horizon.upper()}",
                "MUZERO_SYMBOLS",
            ],
        )
        candidates.append(("training_inventory", fallback_symbols))

        source = "none"
        symbols: list[str] = []
        for source_name, source_symbols in candidates:
            normalized = self._deduplicate_symbols(list(source_symbols or []))
            if normalized:
                source = source_name
                symbols = normalized
                break

        if max_symbols > 0:
            symbols = symbols[:max_symbols]

        return {
            "engine": normalized_engine,
            "horizon": horizon,
            "symbols": symbols,
            "count": len(symbols),
            "source": source,
            "restricted": bool(symbols),
            "promotion_gate": promotion_gate,
            "family": family,
            "feature_profile": feature_profile,
            "top_live_symbols": top_live_symbols,
            "eligible_live_symbols": ranked_symbols.get("eligible_symbols", []),
            "rejected_live_symbols": ranked_symbols.get("rejected_symbols", []),
            "metrics_by_symbol": ranked_symbols.get("metrics_by_symbol", {}),
            "symbol_thresholds": ranked_symbols.get("thresholds", {}),
        }

    def build_engine_horizon_status(
        self,
        engine: str,
        horizon: str,
        champion_id: str | None = None,
    ) -> dict[str, Any]:
        """Assemble l'etat complet d'un moteur/horizon pour le dashboard.

        Args:
            engine (str): Moteur cible.
            horizon (str): Horizon cible.
            champion_id (str | None): Identifiant genetique champion.

        Returns:
            dict[str, Any]: Etat agrege du champion et des artefacts.
        """
        normalized_engine = self.normalize_engine_name(engine)
        horizon = horizon.lower()
        manifest = self.load_manifest(horizon, engine=normalized_engine)
        arena_report = self.load_arena_report(horizon, engine=normalized_engine)
        terminal_summary = load_latest_terminal_summary(engine=normalized_engine, horizon=horizon) or {}
        live_path, live_meta = self.resolve_live_checkpoint(horizon, engine=normalized_engine)
        champion_path = self.get_champion_path(horizon, engine=normalized_engine)
        latest_path = self.get_latest_model_path(horizon, engine=normalized_engine)
        latest_checkpoint = self.get_latest_checkpoint_path(horizon, engine=normalized_engine)
        registry_champion_id = champion_id
        live_champion_id = None
        candidate_id = None

        if manifest:
            manifest_gate = self.resolve_promotion_gate(manifest, arena_report)
            if manifest.get("status") == "promoted" and manifest_gate.get("allowed", False):
                live_champion_id = (
                    manifest.get("challenger_id")
                    or manifest.get("battle_report", {}).get("challenger", {}).get("id")
                )
        if live_champion_id is None:
            live_champion_id = str(live_meta.get("live_champion_id") or "") or None

        if arena_report:
            battle_report = arena_report.get("battle_report", {}) or {}
            candidate_id = battle_report.get("challenger", {}).get("id")
            if (
                arena_report.get("promotion", {}).get("status") == "promoted"
                and live_champion_id is None
            ):
                live_champion_id = candidate_id
        if candidate_id is None:
            candidate_id = str(terminal_summary.get("latest_candidate") or "").strip() or None

        promotion_gate = live_meta.get("promotion_gate") or self.resolve_promotion_gate(
            manifest,
            arena_report,
        )
        live_universe = self.build_live_universe(horizon, engine=normalized_engine)
        candidate_metrics = (
            (arena_report or {}).get("battle_report", {}) or {}
        ).get("challenger", {}).get("metrics", {}) or {}
        family = str(
            live_meta.get("family")
            or (manifest or {}).get("family")
            or candidate_metrics.get("family")
            or terminal_summary.get("family")
            or live_universe.get("family")
            or "mixed"
        )
        feature_profile = str(
            live_meta.get("feature_profile")
            or (manifest or {}).get("feature_profile")
            or candidate_metrics.get("feature_profile")
            or terminal_summary.get("feature_profile")
            or live_universe.get("feature_profile")
            or resolve_feature_profile(horizon, family).get("profile_name")
            or "default"
        )
        mechanics_profile_version = str(
            (manifest or {}).get("mechanics_profile_version")
            or candidate_metrics.get("mechanics_profile_version")
            or terminal_summary.get("mechanics_profile_version")
            or promotion_gate.get("metrics", {}).get("mechanics_profile_version")
            or "v1"
        )
        dataset_coverage = (
            (manifest or {}).get("dataset_coverage")
            or candidate_metrics.get("dataset_coverage")
            or terminal_summary.get("dataset_coverage")
            or promotion_gate.get("metrics", {}).get("dataset_coverage")
            or {}
        )
        directional_metrics = {
            "long_entries": promotion_gate.get("metrics", {}).get("long_entries"),
            "short_entries": promotion_gate.get("metrics", {}).get("short_entries"),
            "long_entry_share": promotion_gate.get("metrics", {}).get("long_entry_share"),
            "short_entry_share": promotion_gate.get("metrics", {}).get("short_entry_share"),
            "directional_imbalance": promotion_gate.get("metrics", {}).get("directional_imbalance"),
            "directional_bias": promotion_gate.get("metrics", {}).get("directional_bias"),
        }

        return {
            "engine": normalized_engine,
            "horizon": horizon,
            "family": family,
            "feature_profile": feature_profile,
            "dataset_id": (
                (manifest or {}).get("dataset_id")
                or candidate_metrics.get("dataset_id")
                or terminal_summary.get("dataset_id")
                or (arena_report or {}).get("dataset_id")
            ),
            "dataset_source": (
                (manifest or {}).get("dataset_source")
                or candidate_metrics.get("dataset_source")
                or terminal_summary.get("dataset_source")
                or (arena_report or {}).get("dataset_source")
            ),
            "champion_id": live_champion_id or registry_champion_id,
            "registry_champion_id": registry_champion_id,
            "live_champion_id": live_champion_id,
            "candidate_id": candidate_id,
            "selection_policy": live_meta.get("policy"),
            "engine_label": live_meta.get("engine_label"),
            "selection": live_meta.get("selection"),
            "gate_allowed": promotion_gate.get("allowed"),
            "gate_profile": promotion_gate.get("gate_profile"),
            "gate_reason": promotion_gate.get("reason"),
            "failure_mode": promotion_gate.get("failure_mode") or terminal_summary.get("failure_mode"),
            "promotion_gate": promotion_gate,
            "promotion_checks": promotion_gate.get("checks", {}),
            "promotion_thresholds": promotion_gate.get("thresholds", {}),
            "candidate_metrics": candidate_metrics,
            "directional_metrics": directional_metrics,
            "mechanics_profile_version": mechanics_profile_version,
            "dataset_coverage": dataset_coverage,
            "metrics_by_position_mechanics": promotion_gate.get("metrics", {}).get(
                "metrics_by_position_mechanics",
                {},
            ),
            "top_live_symbols": live_universe.get("top_live_symbols", []),
            "metrics_by_symbol": live_universe.get("metrics_by_symbol", {}),
            "live_checkpoint": self._describe_path(live_path),
            "champion_checkpoint": self._describe_path(champion_path),
            "latest_model": self._describe_path(latest_path),
            "latest_checkpoint": self._describe_path(latest_checkpoint),
            "latest_run_id": terminal_summary.get("run_id"),
            "latest_candidate": terminal_summary.get("latest_candidate"),
            "latest_verdict": terminal_summary.get("latest_verdict"),
            "failed_step": terminal_summary.get("failed_step"),
            "artifact_state": terminal_summary.get("artifact_state"),
            "terminal_summary": terminal_summary or None,
            "manifest": manifest,
            "arena_report": arena_report,
            "live_universe": live_universe,
        }

    def build_horizon_status(self, horizon: str, champion_id: str | None = None) -> dict[str, Any]:
        """Assemble l'etat complet d'un horizon MuZero pour le dashboard.

        Args:
            horizon (str): Horizon cible.
            champion_id (str | None): Identifiant genetique champion.

        Returns:
            dict[str, Any]: Etat agrege du champion MuZero.
        """
        return self.build_engine_horizon_status("muzero", horizon, champion_id=champion_id)

    def build_engine_matrix_status(
        self,
        horizons: list[str],
        registry_champions: dict[str, str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Construit l'etat complet des moteurs par horizon.

        Args:
            horizons (list[str]): Horizons a exposer.
            registry_champions (dict[str, str] | None): Champions MuZero issus du registre genetique.

        Returns:
            dict[str, dict[str, Any]]: Matrice `moteur -> horizon -> statut`.
        """
        registry = dict(registry_champions or {})
        return {
            engine: {
                horizon: self.build_engine_horizon_status(
                    engine,
                    horizon,
                    champion_id=registry.get(horizon) if engine == "muzero" else None,
                )
                for horizon in horizons
            }
            for engine in ("muzero", "dreamer")
        }

    def promote_challenger(
        self,
        challenger_path: str | Path,
        horizon: str,
        battle_report: dict[str, Any],
        training_metrics: dict[str, Any] | None = None,
        latest_checkpoint: str | Path | None = None,
        challenger_id: str | None = None,
        engine: str = "muzero",
        gate_profile: str | None = None,
    ) -> dict[str, Any]:
        """Promeut un challenger d'un moteur vers le live.

        Args:
            challenger_path (str | Path): Checkpoint challenger retenu.
            horizon (str): Horizon strategique concerne.
            battle_report (dict[str, Any]): Verdict de l'Arena.
            training_metrics (dict[str, Any] | None): Metriques du run.
            latest_checkpoint (str | Path | None): Dernier checkpoint du run.
            challenger_id (str | None): Identifiant genetique du challenger.
            engine (str): Moteur du challenger a promouvoir.
            gate_profile (str | None): Profil de gate a appliquer.

        Returns:
            dict[str, Any]: Resultat detaille de la promotion.
        """
        normalized_engine = self.normalize_engine_name(engine)
        horizon = horizon.lower()
        source_path = Path(challenger_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Checkpoint challenger introuvable: {source_path}")

        outcome = str(battle_report.get("outcome", "DEFEAT")).upper()
        if outcome != "VICTORY":
            logger.info("Promotion ignoree pour %s: verdict Arena=%s.", horizon, outcome)
            return {
                "status": "skipped",
                "reason": f"arena_{outcome.lower()}",
                "engine": normalized_engine,
                "horizon": horizon,
                "source_path": str(source_path),
                "champion_paths": [],
                "promotion_gate": self.evaluate_promotion_gate(
                    battle_report,
                    gate_profile=gate_profile,
                ),
            }

        promotion_gate = self.evaluate_promotion_gate(
            battle_report,
            gate_profile=gate_profile,
        )
        if not promotion_gate["allowed"]:
            logger.warning(
                "Promotion live refusee pour %s: garde-fou=%s | metrics=%s",
                horizon,
                promotion_gate["reason"],
                promotion_gate["metrics"],
            )
            return {
                "status": "skipped",
                "reason": f"promotion_gate_{promotion_gate['reason']}",
                "engine": normalized_engine,
                "horizon": horizon,
                "source_path": str(source_path),
                "champion_paths": [],
                "promotion_gate": promotion_gate,
            }

        champion_paths: list[str] = []
        horizon_champion = self.get_champion_path(horizon, engine=normalized_engine)
        shutil.copy2(source_path, horizon_champion)
        champion_paths.append(str(horizon_champion))

        legacy_path: str | None = None
        if normalized_engine == "muzero" and horizon == "intraday":
            legacy_champion = self.weights_dir / "muzero_champion.pkl"
            shutil.copy2(source_path, legacy_champion)
            legacy_path = str(legacy_champion)
            champion_paths.append(legacy_path)

        manifest = {
            "status": "promoted",
            "promoted_at": datetime.now().isoformat(),
            "engine": normalized_engine,
            "horizon": horizon,
            "family": str(
                battle_report.get("challenger", {}).get("metrics", {}).get("family") or "mixed"
            ),
            "feature_profile": str(
                battle_report.get("challenger", {}).get("metrics", {}).get("feature_profile") or "default"
            ),
            "dataset_id": battle_report.get("challenger", {}).get("metrics", {}).get("dataset_id"),
            "dataset_source": battle_report.get("challenger", {}).get("metrics", {}).get("dataset_source"),
            "gate_profile": promotion_gate.get("gate_profile"),
            "selection_policy": "champion_only",
            "engine_label": self.get_engine_label(normalized_engine, variant="champion"),
            "challenger_id": challenger_id or battle_report.get("challenger", {}).get("id"),
            "source_path": str(source_path),
            "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint else None,
            "champion_path": str(horizon_champion),
            "legacy_champion_path": legacy_path,
            "battle_report": battle_report,
            "training_metrics": training_metrics or {},
            "promotion_gate": promotion_gate,
        }
        manifest_path = self.get_manifest_path(horizon, engine=normalized_engine)
        manifest_path.write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")
        logger.info(
            "Champion %s/%s promu et manifeste ecrit dans %s.",
            normalized_engine,
            horizon,
            manifest_path,
        )
        return {**manifest, "champion_paths": champion_paths}

    def promote_muzero_challenger(
        self,
        challenger_path: str | Path,
        horizon: str,
        battle_report: dict[str, Any],
        training_metrics: dict[str, Any] | None = None,
        latest_checkpoint: str | Path | None = None,
        challenger_id: str | None = None,
        gate_profile: str | None = None,
    ) -> dict[str, Any]:
        """Promeut un challenger MuZero en champion live.

        Args:
            challenger_path (str | Path): Checkpoint challenger retenu.
            horizon (str): Horizon MuZero concerne.
            battle_report (dict[str, Any]): Verdict de l'Arena.
            training_metrics (dict[str, Any] | None): Metriques du run.
            latest_checkpoint (str | Path | None): Dernier checkpoint du run.
            challenger_id (str | None): Identifiant genetique du challenger.
            gate_profile (str | None): Profil de gate a appliquer.

        Returns:
            dict[str, Any]: Resultat detaille de la promotion.
        """
        return self.promote_challenger(
            challenger_path=challenger_path,
            horizon=horizon,
            battle_report=battle_report,
            training_metrics=training_metrics,
            latest_checkpoint=latest_checkpoint,
            challenger_id=challenger_id,
            engine="muzero",
            gate_profile=gate_profile,
        )

    def promote_dreamer_challenger(
        self,
        challenger_path: str | Path,
        horizon: str,
        battle_report: dict[str, Any],
        training_metrics: dict[str, Any] | None = None,
        latest_checkpoint: str | Path | None = None,
        challenger_id: str | None = None,
        gate_profile: str | None = None,
    ) -> dict[str, Any]:
        """Promeut un challenger DreamerV3 en champion live.

        Args:
            challenger_path (str | Path): Checkpoint challenger retenu.
            horizon (str): Horizon Dreamer concerne.
            battle_report (dict[str, Any]): Verdict de l'Arena.
            training_metrics (dict[str, Any] | None): Metriques du run.
            latest_checkpoint (str | Path | None): Dernier checkpoint du run.
            challenger_id (str | None): Identifiant genetique du challenger.
            gate_profile (str | None): Profil de gate a appliquer.

        Returns:
            dict[str, Any]: Resultat detaille de la promotion.
        """
        return self.promote_challenger(
            challenger_path=challenger_path,
            horizon=horizon,
            battle_report=battle_report,
            training_metrics=training_metrics,
            latest_checkpoint=latest_checkpoint,
            challenger_id=challenger_id,
            engine="dreamer",
            gate_profile=gate_profile,
        )

    def resolve_live_checkpoint(
        self,
        horizon: str,
        selection_policy: str | None = None,
        engine: str = "muzero",
    ) -> tuple[Path | None, dict[str, Any]]:
        """Determine le checkpoint live autorise pour un horizon.

        Args:
            horizon (str): Horizon de decision live.
            selection_policy (str | None): Politique de selection a imposer.
            engine (str): Moteur de prediction cible.

        Returns:
            tuple[Path | None, dict[str, Any]]: Chemin retenu et metadonnees.
        """
        normalized_engine = self.normalize_engine_name(engine)
        horizon = horizon.lower()
        policy = self.normalize_live_selection_policy(
            selection_policy,
            default=self.get_live_selection_policy(),
        )
        manifest = self.load_manifest(horizon, engine=normalized_engine) or {}
        arena_report = self.load_arena_report(horizon, engine=normalized_engine)
        promotion_gate = self.resolve_promotion_gate(manifest, arena_report)
        champion_path = self.get_champion_path(horizon, engine=normalized_engine)
        legacy_champion = self.weights_dir / "muzero_champion.pkl"
        latest_path = self.get_latest_model_path(horizon, engine=normalized_engine)
        latest_checkpoint = None

        checkpoint_candidates = sorted(
            self.weights_dir.glob(self.get_checkpoint_glob(horizon, engine=normalized_engine)),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if checkpoint_candidates:
            latest_checkpoint = checkpoint_candidates[0]

        if champion_path.exists():
            if not promotion_gate.get("allowed", False):
                logger.warning(
                    "Champion %s bloque pour le live: %s.",
                    horizon,
                    promotion_gate.get("reason"),
                )
            else:
                return champion_path, {
                    "selection": "champion",
                    "engine": normalized_engine,
                    "policy": policy,
                    "engine_label": manifest.get(
                        "engine_label",
                        self.get_engine_label(normalized_engine, variant="champion"),
                    ),
                    "live_champion_id": manifest.get("challenger_id"),
                    "family": manifest.get("family"),
                    "feature_profile": manifest.get("feature_profile"),
                    "manifest": manifest,
                    "promotion_gate": promotion_gate,
                }
            return None, {
                "selection": "blocked_champion",
                "engine": normalized_engine,
                "policy": policy,
                "engine_label": self.get_engine_label(normalized_engine, variant="blocked"),
                "live_champion_id": manifest.get("challenger_id"),
                "family": manifest.get("family"),
                "feature_profile": manifest.get("feature_profile"),
                "manifest": manifest,
                "promotion_gate": promotion_gate,
            }

        if normalized_engine == "muzero" and horizon == "intraday" and legacy_champion.exists():
            if promotion_gate.get("allowed", False):
                return legacy_champion, {
                    "selection": "legacy_champion",
                    "engine": normalized_engine,
                    "policy": policy,
                    "engine_label": self.get_engine_label(normalized_engine, variant="champion"),
                    "live_champion_id": manifest.get("challenger_id"),
                    "family": manifest.get("family"),
                    "feature_profile": manifest.get("feature_profile"),
                    "manifest": manifest,
                    "promotion_gate": promotion_gate,
                }
            return None, {
                "selection": "blocked_legacy_champion",
                "engine": normalized_engine,
                "policy": policy,
                "engine_label": self.get_engine_label(normalized_engine, variant="blocked"),
                "live_champion_id": manifest.get("challenger_id"),
                "family": manifest.get("family"),
                "feature_profile": manifest.get("feature_profile"),
                "manifest": manifest,
                "promotion_gate": promotion_gate,
            }

        if policy == "champion_then_latest" and latest_path.exists():
            logger.warning(
                "Aucun champion promu pour %s. Utilisation du latest autorisee par la politique live.",
                horizon,
            )
            return latest_path, {
                "selection": "latest",
                "engine": normalized_engine,
                "policy": policy,
                "engine_label": self.get_engine_label(normalized_engine, variant="latest"),
                "family": manifest.get("family"),
                "feature_profile": manifest.get("feature_profile"),
                "manifest": manifest,
                "promotion_gate": promotion_gate,
            }

        if policy == "checkpoint_preview" and latest_checkpoint is not None:
            logger.warning(
                "Aucun champion promu pour %s. Utilisation du dernier checkpoint preview.",
                horizon,
            )
            return latest_checkpoint, {
                "selection": "checkpoint_preview",
                "engine": normalized_engine,
                "policy": policy,
                "engine_label": self.get_engine_label(normalized_engine, variant="preview"),
                "family": manifest.get("family"),
                "feature_profile": manifest.get("feature_profile"),
                "manifest": manifest,
                "promotion_gate": promotion_gate,
            }

        return None, {
            "selection": "none",
            "engine": normalized_engine,
            "policy": policy,
            "engine_label": self.get_engine_label(normalized_engine, variant="blocked"),
            "live_champion_id": manifest.get("challenger_id"),
            "family": manifest.get("family"),
            "feature_profile": manifest.get("feature_profile"),
            "manifest": manifest,
            "promotion_gate": promotion_gate,
        }
