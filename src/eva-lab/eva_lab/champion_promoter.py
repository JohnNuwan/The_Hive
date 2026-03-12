"""Gere la promotion des challengers vers les champions live."""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from eva_lab.training_utils import get_horizon_timeframe, resolve_training_symbols

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
    def get_live_selection_policy() -> str:
        """Retourne la politique live active.

        Returns:
            str: Politique de selection live.
        """
        raw_policy = os.getenv("MUZERO_LIVE_SELECTION_POLICY", "champion_only").strip().lower()
        allowed = {"champion_only", "champion_then_latest", "checkpoint_preview"}
        if raw_policy not in allowed:
            logger.warning(
                "Politique live inconnue '%s'. Repli sur 'champion_only'.",
                raw_policy,
            )
            return "champion_only"
        return raw_policy

    def get_manifest_path(self, horizon: str) -> Path:
        """Construit le chemin du manifeste d'un horizon.

        Args:
            horizon (str): Horizon concerne.

        Returns:
            Path: Chemin du manifeste JSON.
        """
        return self.results_dir / f"champion_{horizon.lower()}.json"

    def get_arena_report_path(self, horizon: str) -> Path:
        """Construit le chemin du rapport Arena d'un horizon.

        Args:
            horizon (str): Horizon concerne.

        Returns:
            Path: Chemin du rapport Arena.
        """
        return self.results_dir / f"arena_{horizon.lower()}_latest.json"

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

    def get_promotion_thresholds(self) -> dict[str, Any]:
        """Retourne les seuils minimums de promotion live.

        Returns:
            dict[str, Any]: Configuration active de filtrage des champions.
        """
        return {
            "require_positive_metrics": self._read_boolean_env(
                "MUZERO_PROMOTION_REQUIRE_POSITIVE_METRICS",
                True,
            ),
            "min_win_rate": self._to_float(
                os.getenv("MUZERO_PROMOTION_MIN_WIN_RATE", "55.0"),
                default=55.0,
            ),
            "min_return_pct": self._to_float(
                os.getenv("MUZERO_PROMOTION_MIN_RETURN_PCT", "0.01"),
                default=0.01,
            ),
            "min_profit_factor": self._to_float(
                os.getenv("MUZERO_PROMOTION_MIN_PROFIT_FACTOR", "1.10"),
                default=1.10,
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
                os.getenv("MUZERO_PROMOTION_MIN_EXPECTANCY_PCT", "0.002"),
                default=0.002,
            ),
            "max_drawdown_pct": self._to_float(
                os.getenv("MUZERO_PROMOTION_MAX_DRAWDOWN_PCT", "4.0"),
                default=4.0,
            ),
            "min_positive_episode_rate": self._to_float(
                os.getenv("MUZERO_PROMOTION_MIN_POSITIVE_EPISODE_RATE", "20.0"),
                default=20.0,
            ),
            "min_net_realized_pct": self._to_float(
                os.getenv("MUZERO_PROMOTION_MIN_NET_REALIZED_PCT", "0.50"),
                default=0.50,
            ),
        }

    def evaluate_promotion_gate(
        self,
        battle_report: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Valide l'eligibilite d'un challenger avant de l'exposer au live.

        Args:
            battle_report (dict[str, Any] | None): Rapport Arena du challenger.

        Returns:
            dict[str, Any]: Verdict detaille avec metriques, seuils et raison.
        """
        thresholds = self.get_promotion_thresholds()
        if not battle_report:
            return {
                "allowed": False,
                "status": "blocked",
                "reason": "missing_battle_report",
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
                },
            }

        challenger_metrics = (battle_report.get("challenger", {}) or {}).get("metrics", {}) or {}
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
        checks = {
            "arena_victory": str(battle_report.get("outcome", "")).upper() == "VICTORY",
            "validation_sample": bool(validation.get("sample_size_ok", False)),
            "evaluation_games": evaluation_games >= thresholds["min_eval_games"],
            "evaluation_symbols": evaluation_symbols >= thresholds["min_eval_symbols"],
            "total_trades": total_trades >= thresholds["min_total_trades"],
            "max_drawdown_pct": max_drawdown_pct <= thresholds["max_drawdown_pct"],
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

        failure_reason = next((name for name, passed in checks.items() if not passed), "eligible")
        return {
            "allowed": all(checks.values()),
            "status": "eligible" if all(checks.values()) else "blocked",
            "reason": failure_reason,
            "checks": checks,
            "thresholds": thresholds,
            "metrics": {
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
            },
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
        if manifest and isinstance(manifest.get("promotion_gate"), dict):
            return manifest["promotion_gate"]

        if manifest and isinstance(manifest.get("battle_report"), dict):
            return self.evaluate_promotion_gate(manifest["battle_report"])

        if arena_report and isinstance(arena_report.get("battle_report"), dict):
            return self.evaluate_promotion_gate(arena_report["battle_report"])

        return self.evaluate_promotion_gate(None)

    def load_manifest(self, horizon: str) -> dict[str, Any] | None:
        """Charge le manifeste d'un champion si disponible.

        Args:
            horizon (str): Horizon cible.

        Returns:
            dict[str, Any] | None: Manifeste charge ou ``None``.
        """
        manifest_path = self.get_manifest_path(horizon)
        if not manifest_path.exists():
            return None

        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Lecture manifeste champion impossible pour %s: %s", horizon, exc)
            return None

    def load_arena_report(self, horizon: str) -> dict[str, Any] | None:
        """Charge le dernier rapport Arena pour un horizon.

        Args:
            horizon (str): Horizon cible.

        Returns:
            dict[str, Any] | None: Rapport Arena ou ``None``.
        """
        report_path = self.get_arena_report_path(horizon)
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

    def get_latest_checkpoint_path(self, horizon: str) -> Path | None:
        """Retourne le dernier checkpoint intermediaire d'un horizon.

        Args:
            horizon (str): Horizon cible.

        Returns:
            Path | None: Dernier checkpoint ou ``None``.
        """
        checkpoint_candidates = sorted(
            self.weights_dir.glob(f"muzero_{horizon.lower()}_ckpt_*.pkl"),
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

    def build_live_universe(self, horizon: str) -> dict[str, Any]:
        """Construit l'univers live recommande pour un horizon donne.

        Args:
            horizon (str): Horizon strategique cible.

        Returns:
            dict[str, Any]: Univers recommande, source et garde-fou associe.
        """
        horizon = horizon.lower()
        manifest = self.load_manifest(horizon) or {}
        arena_report = self.load_arena_report(horizon) or {}
        promotion_gate = self.resolve_promotion_gate(manifest, arena_report)
        max_symbols = int(os.getenv("MUZERO_LIVE_UNIVERSE_MAX_SYMBOLS", "12"))

        candidates: list[tuple[str, list[str]]] = []
        manifest_battle = manifest.get("battle_report", {}) or {}
        arena_battle = arena_report.get("battle_report", {}) or {}
        candidates.append(("manifest_eval_symbols", manifest_battle.get("eval_symbols", []) or []))
        candidates.append(("arena_eval_symbols", arena_battle.get("eval_symbols", []) or []))
        candidates.append(("arena_training_symbols", arena_report.get("symbols", []) or []))

        fallback_symbols = resolve_training_symbols(
            required_timeframes={get_horizon_timeframe(horizon)},
            max_symbols=max_symbols,
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
            "horizon": horizon,
            "symbols": symbols,
            "count": len(symbols),
            "source": source,
            "restricted": bool(symbols),
            "promotion_gate": promotion_gate,
        }

    def build_horizon_status(self, horizon: str, champion_id: str | None = None) -> dict[str, Any]:
        """Assemble l'etat complet d'un horizon pour le dashboard.

        Args:
            horizon (str): Horizon cible.
            champion_id (str | None): Identifiant genetique champion.

        Returns:
            dict[str, Any]: Etat agrege du champion et des artefacts.
        """
        horizon = horizon.lower()
        manifest = self.load_manifest(horizon)
        arena_report = self.load_arena_report(horizon)
        live_path, live_meta = self.resolve_live_checkpoint(horizon)
        champion_path = self.weights_dir / f"muzero_champion_{horizon}.pkl"
        latest_path = self.weights_dir / f"muzero_{horizon}_latest.pkl"
        latest_checkpoint = self.get_latest_checkpoint_path(horizon)
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

        if arena_report:
            battle_report = arena_report.get("battle_report", {}) or {}
            candidate_id = battle_report.get("challenger", {}).get("id")
            if (
                arena_report.get("promotion", {}).get("status") == "promoted"
                and live_champion_id is None
            ):
                live_champion_id = candidate_id

        promotion_gate = live_meta.get("promotion_gate") or self.resolve_promotion_gate(
            manifest,
            arena_report,
        )
        live_universe = self.build_live_universe(horizon)

        return {
            "horizon": horizon,
            "champion_id": live_champion_id or registry_champion_id,
            "registry_champion_id": registry_champion_id,
            "live_champion_id": live_champion_id,
            "candidate_id": candidate_id,
            "selection_policy": live_meta.get("policy"),
            "engine_label": live_meta.get("engine_label"),
            "selection": live_meta.get("selection"),
            "gate_allowed": promotion_gate.get("allowed"),
            "gate_reason": promotion_gate.get("reason"),
            "promotion_gate": promotion_gate,
            "live_checkpoint": self._describe_path(live_path),
            "champion_checkpoint": self._describe_path(champion_path),
            "latest_model": self._describe_path(latest_path),
            "latest_checkpoint": self._describe_path(latest_checkpoint),
            "manifest": manifest,
            "arena_report": arena_report,
            "live_universe": live_universe,
        }

    def promote_muzero_challenger(
        self,
        challenger_path: str | Path,
        horizon: str,
        battle_report: dict[str, Any],
        training_metrics: dict[str, Any] | None = None,
        latest_checkpoint: str | Path | None = None,
        challenger_id: str | None = None,
    ) -> dict[str, Any]:
        """Promeut un challenger MuZero en champion live.

        Args:
            challenger_path (str | Path): Checkpoint challenger retenu.
            horizon (str): Horizon MuZero concerne.
            battle_report (dict[str, Any]): Verdict de l'Arena.
            training_metrics (dict[str, Any] | None): Metriques du run.
            latest_checkpoint (str | Path | None): Dernier checkpoint du run.
            challenger_id (str | None): Identifiant genetique du challenger.

        Returns:
            dict[str, Any]: Resultat detaille de la promotion.
        """
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
                "horizon": horizon,
                "source_path": str(source_path),
                "champion_paths": [],
                "promotion_gate": self.evaluate_promotion_gate(battle_report),
            }

        promotion_gate = self.evaluate_promotion_gate(battle_report)
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
                "horizon": horizon,
                "source_path": str(source_path),
                "champion_paths": [],
                "promotion_gate": promotion_gate,
            }

        champion_paths: list[str] = []
        horizon_champion = self.weights_dir / f"muzero_champion_{horizon}.pkl"
        shutil.copy2(source_path, horizon_champion)
        champion_paths.append(str(horizon_champion))

        legacy_path: str | None = None
        if horizon == "intraday":
            legacy_champion = self.weights_dir / "muzero_champion.pkl"
            shutil.copy2(source_path, legacy_champion)
            legacy_path = str(legacy_champion)
            champion_paths.append(legacy_path)

        manifest = {
            "status": "promoted",
            "promoted_at": datetime.now().isoformat(),
            "horizon": horizon,
            "selection_policy": "champion_only",
            "engine_label": "MuZero JAX Champion",
            "challenger_id": challenger_id or battle_report.get("challenger", {}).get("id"),
            "source_path": str(source_path),
            "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint else None,
            "champion_path": str(horizon_champion),
            "legacy_champion_path": legacy_path,
            "battle_report": battle_report,
            "training_metrics": training_metrics or {},
            "promotion_gate": promotion_gate,
        }
        manifest_path = self.get_manifest_path(horizon)
        manifest_path.write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")
        logger.info("Champion %s promu et manifeste ecrit dans %s.", horizon, manifest_path)
        return {**manifest, "champion_paths": champion_paths}

    def resolve_live_checkpoint(self, horizon: str) -> tuple[Path | None, dict[str, Any]]:
        """Determine le checkpoint live autorise pour un horizon.

        Args:
            horizon (str): Horizon de decision live.

        Returns:
            tuple[Path | None, dict[str, Any]]: Chemin retenu et metadonnees.
        """
        horizon = horizon.lower()
        policy = self.get_live_selection_policy()
        manifest = self.load_manifest(horizon) or {}
        arena_report = self.load_arena_report(horizon)
        promotion_gate = self.resolve_promotion_gate(manifest, arena_report)
        champion_path = self.weights_dir / f"muzero_champion_{horizon}.pkl"
        legacy_champion = self.weights_dir / "muzero_champion.pkl"
        latest_path = self.weights_dir / f"muzero_{horizon}_latest.pkl"
        latest_checkpoint = None

        checkpoint_candidates = sorted(
            self.weights_dir.glob(f"muzero_{horizon}_ckpt_*.pkl"),
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
                    "policy": policy,
                    "engine_label": manifest.get("engine_label", "MuZero JAX Champion"),
                    "manifest": manifest,
                    "promotion_gate": promotion_gate,
                }
            return None, {
                "selection": "blocked_champion",
                "policy": policy,
                "engine_label": "RSI Heuristic (champion bloque)",
                "manifest": manifest,
                "promotion_gate": promotion_gate,
            }

        if horizon == "intraday" and legacy_champion.exists():
            if promotion_gate.get("allowed", False):
                return legacy_champion, {
                    "selection": "legacy_champion",
                    "policy": policy,
                    "engine_label": "MuZero JAX Champion",
                    "manifest": manifest,
                    "promotion_gate": promotion_gate,
                }
            return None, {
                "selection": "blocked_legacy_champion",
                "policy": policy,
                "engine_label": "RSI Heuristic (champion bloque)",
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
                "policy": policy,
                "engine_label": "MuZero JAX Latest",
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
                "policy": policy,
                "engine_label": "MuZero JAX Preview",
                "manifest": manifest,
                "promotion_gate": promotion_gate,
            }

        return None, {
            "selection": "none",
            "policy": policy,
            "engine_label": "RSI Heuristic (fallback)",
            "manifest": manifest,
            "promotion_gate": promotion_gate,
        }
