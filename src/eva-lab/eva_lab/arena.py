"""Arena Darwinienne pour l'evaluation des modeles MuZero sur historique reel."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from eva_lab.muzero.config import MuZeroConfigV3
from eva_lab.muzero.environment import TradingEnvironment
from eva_lab.muzero.jax_agent import JAXMuZeroAgent
from eva_lab.training_utils import (
    build_muzero_market_data,
    get_horizon_timeframe,
    load_history_frame,
    resolve_training_symbols,
)

logger = logging.getLogger(__name__)


class Arena:
    """Execute des combats entre challenger et champion sur des historiques reels."""

    def __init__(
        self,
        weights_dir: str = "data/muzero/weights",
        data_dir: str = "data/history",
    ) -> None:
        """Initialise l'Arena et ses emplacements de donnees.

        Args:
            weights_dir (str): Dossier des checkpoints MuZero.
            data_dir (str): Dossier des historiques CSV.
        """
        self.history: list[dict[str, Any]] = []
        self.weights_dir = Path(weights_dir)
        self.data_dir = data_dir

    @staticmethod
    def _read_int_env(name: str, default: int) -> int:
        """Lit une variable d'environnement entiere avec repli.

        Args:
            name (str): Nom de la variable d'environnement.
            default (int): Valeur de repli.

        Returns:
            int: Valeur normalisee.
        """
        raw_value = os.getenv(name)
        if raw_value is None:
            return default
        try:
            return int(raw_value)
        except ValueError:
            logger.warning("Variable %s invalide (%s). Repli sur %s.", name, raw_value, default)
            return default

    @staticmethod
    def _read_float_env(name: str, default: float) -> float:
        """Lit une variable d'environnement flottante avec repli.

        Args:
            name (str): Nom de la variable d'environnement.
            default (float): Valeur de repli.

        Returns:
            float: Valeur normalisee.
        """
        raw_value = os.getenv(name)
        if raw_value is None:
            return default
        try:
            return float(raw_value)
        except ValueError:
            logger.warning("Variable %s invalide (%s). Repli sur %s.", name, raw_value, default)
            return default

    @staticmethod
    def _empty_metrics() -> dict[str, float]:
        """Retourne un jeu de metriques neutres si l'evaluation echoue.

        Returns:
            dict[str, float]: Metriques nulles et tailles d'echantillons a zero.
        """
        return {
            "profit_factor": 0.0,
            "return_pct": 0.0,
            "net_realized_pct": 0.0,
            "gross_profit_pct": 0.0,
            "gross_loss_pct": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "profitable_trades": 0,
            "evaluation_games": 0,
            "evaluation_symbols": 0,
            "max_drawdown_pct": 100.0,
            "expectancy_pct": 0.0,
            "positive_episode_rate": 0.0,
        }

    @staticmethod
    def _compute_max_drawdown_pct(equity_curve: list[float] | tuple[float, ...]) -> float:
        """Calcule le drawdown maximal d'une courbe d'equite.

        Args:
            equity_curve (list[float] | tuple[float, ...]): Courbe d'equite brute.

        Returns:
            float: Drawdown maximal en pourcentage.
        """
        if not equity_curve:
            return 0.0

        peak = float(equity_curve[0])
        max_drawdown = 0.0
        for equity in equity_curve:
            equity_value = float(equity)
            peak = max(peak, equity_value)
            if peak <= 0:
                continue
            drawdown = (peak - equity_value) / peak * 100.0
            drawdown = max(0.0, min(drawdown, 100.0))
            max_drawdown = max(max_drawdown, drawdown)
        return max_drawdown

    @staticmethod
    def _score_metrics(metrics: dict[str, float]) -> float:
        """Produit un score unique robuste a partir des metriques d'Arena.

        Args:
            metrics (dict[str, float]): Metriques agregees du modele.

        Returns:
            float: Score comparatif utilise pour le duel champion/challenger.
        """
        profit_factor = min(max(float(metrics.get("profit_factor", 0.0)), 0.0), 5.0)
        return (
            float(metrics.get("return_pct", 0.0)) * 8.0
            + max(0.0, profit_factor - 1.0) * 18.0
            + float(metrics.get("expectancy_pct", 0.0)) * 120.0
            + float(metrics.get("win_rate", 0.0)) * 0.08
            + float(metrics.get("positive_episode_rate", 0.0)) * 0.06
            - float(metrics.get("max_drawdown_pct", 0.0)) * 1.5
        )

    def _resolve_model_path(self, model_id: str, horizon: str) -> Path | None:
        """Retourne le chemin d'un modele, avec fallback sur les champions historiques.

        Args:
            model_id (str): Identifiant du modele.
            horizon (str): Horizon strategique evalue.

        Returns:
            Path | None: Chemin vers les poids du modele si disponible.
        """
        direct_path = self.weights_dir / f"{model_id}.pkl"
        if direct_path.exists():
            return direct_path

        candidates = []
        if model_id == "gen_000_baseline":
            candidates.extend(
                [
                    self.weights_dir / f"muzero_champion_{horizon}.pkl",
                    self.weights_dir / "muzero_champion.pkl",
                    self.weights_dir / "muzero_global_latest.pkl",
                ]
            )
        elif model_id == "muzero_champion":
            candidates.extend(
                [
                    self.weights_dir / f"muzero_champion_{horizon}.pkl",
                    self.weights_dir / "muzero_champion.pkl",
                ]
            )

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_market_data(self, symbol: str, horizon: str) -> np.ndarray | None:
        """Charge la matrice de marche necessaire a l'evaluation historique.

        Args:
            symbol (str): Symbole a evaluer.
            horizon (str): Horizon strategique cible.

        Returns:
            np.ndarray | None: Matrice de marche MuZero ou ``None`` si absente.
        """
        timeframe = get_horizon_timeframe(horizon)
        frame = load_history_frame(symbol, timeframe, self.data_dir)
        if frame is None:
            logger.warning("Arena: historique absent pour %s sur %s.", symbol, timeframe)
            return None

        history_bars = self._read_int_env("ARENA_HISTORY_BARS", 6000)
        market_data = build_muzero_market_data(frame.tail(history_bars))
        if market_data.shape[0] < 240:
            logger.warning("Arena: historique insuffisant pour %s sur %s.", symbol, timeframe)
            return None
        return market_data

    def _build_eval_segments(
        self,
        market_data: np.ndarray,
        config: MuZeroConfigV3,
        games_per_symbol: int,
    ) -> list[np.ndarray]:
        """Decoupe un historique en plusieurs fenetres d'evaluation.

        Args:
            market_data (np.ndarray): Matrice de marche complete.
            config (MuZeroConfigV3): Configuration MuZero courante.
            games_per_symbol (int): Nombre de fenetres a evaluer par symbole.

        Returns:
            list[np.ndarray]: Liste de segments chronologiques exploitables.
        """
        min_segment_bars = max(240, config.max_moves + 120)
        target_segment_bars = max(
            min_segment_bars,
            self._read_int_env("ARENA_SEGMENT_BARS", 1400),
        )
        if market_data.shape[0] < min_segment_bars:
            return []

        if market_data.shape[0] <= target_segment_bars:
            return [market_data[-target_segment_bars:]]

        endpoint_values = np.linspace(
            min_segment_bars,
            market_data.shape[0],
            num=max(1, games_per_symbol),
            dtype=int,
        )
        segments: list[np.ndarray] = []
        for end_index in sorted({int(value) for value in endpoint_values if int(value) >= min_segment_bars}):
            start_index = max(0, end_index - target_segment_bars)
            segment = market_data[start_index:end_index]
            if segment.shape[0] >= min_segment_bars:
                segments.append(segment)
        return segments

    def _evaluate_model(self, weights_path: Path, symbols: list[str], horizon: str) -> dict[str, float]:
        """Evalue un modele sur plusieurs fenetres historiques reellement distinctes.

        Args:
            weights_path (Path): Checkpoint du modele a tester.
            symbols (list[str]): Univers historique retenu.
            horizon (str): Horizon strategique evalue.

        Returns:
            dict[str, float]: Metriques consolidees de robustesse.
        """
        if not weights_path.exists():
            logger.warning("Poids absents pour l'evaluation: %s", weights_path)
            return self._empty_metrics()

        config = MuZeroConfigV3(
            horizon=horizon,
            primary_timeframe=get_horizon_timeframe(horizon),
            symbols=symbols,
        )
        try:
            agent = JAXMuZeroAgent(config)
            agent.load(str(weights_path))
        except Exception as exc:
            logger.error("Chargement MuZero impossible pour %s: %s", weights_path, exc)
            return self._empty_metrics()

        eval_games_per_symbol = max(1, self._read_int_env("ARENA_GAMES_PER_SYMBOL", 6))
        total_return = 0.0
        gross_profit = 0.0
        gross_loss = 0.0
        total_trades = 0
        total_profitable = 0
        evaluated_symbols = 0
        evaluation_games = 0
        positive_episodes = 0
        worst_drawdown = 0.0
        total_net_realized_pct = 0.0

        for symbol in symbols:
            market_data = self._load_market_data(symbol, horizon)
            if market_data is None:
                continue

            segments = self._build_eval_segments(market_data, config, eval_games_per_symbol)
            if not segments:
                continue

            evaluated_symbols += 1
            for segment in segments:
                max_steps = min(config.max_moves, segment.shape[0] - 101)
                env = TradingEnvironment(
                    data=segment,
                    symbol=symbol,
                    config=config,
                    max_steps=max_steps,
                )
                agent.play_game(env, exploration=False)
                summary = env.get_summary()

                episode_return = float(summary.get("return_pct", 0.0))
                episode_net_realized = float(summary.get("net_realized_pct", episode_return))
                episode_gross_profit = float(summary.get("gross_profit_pct", 0.0))
                episode_gross_loss = float(summary.get("gross_loss_pct", 0.0))
                episode_trades = int(summary.get("total_trades", 0) or 0)
                episode_profitable = int(summary.get("profitable_trades", 0) or 0)
                episode_drawdown = self._compute_max_drawdown_pct(getattr(env, "equity_curve", []))

                evaluation_games += 1
                total_return += episode_return
                total_net_realized_pct += episode_net_realized
                total_trades += episode_trades
                total_profitable += episode_profitable
                worst_drawdown = max(worst_drawdown, episode_drawdown)

                gross_profit += max(0.0, episode_gross_profit)
                gross_loss += max(0.0, episode_gross_loss)

                if episode_return > 0:
                    positive_episodes += 1

        if evaluation_games == 0:
            return self._empty_metrics()

        average_return = total_return / evaluation_games
        win_rate = (total_profitable / max(total_trades, 1)) * 100.0
        expectancy_pct = total_net_realized_pct / max(total_trades, 1)
        positive_episode_rate = (positive_episodes / evaluation_games) * 100.0
        profit_factor = gross_profit / gross_loss if gross_loss > 1e-8 else (gross_profit if gross_profit > 0 else 0.0)

        return {
            "profit_factor": profit_factor,
            "return_pct": average_return,
            "net_realized_pct": total_net_realized_pct,
            "gross_profit_pct": gross_profit,
            "gross_loss_pct": gross_loss,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "profitable_trades": total_profitable,
            "evaluation_games": evaluation_games,
            "evaluation_symbols": evaluated_symbols,
            "max_drawdown_pct": worst_drawdown,
            "expectancy_pct": expectancy_pct,
            "positive_episode_rate": positive_episode_rate,
        }

    def battle(
        self,
        challenger_id: str,
        champion_id: str = "gen_000_baseline",
        horizon: str = "intraday",
    ) -> dict[str, Any]:
        """Compare deux generations MuZero et retourne le verdict ADN.

        Args:
            challenger_id (str): Identifiant du challenger.
            champion_id (str): Identifiant du champion courant.
            horizon (str): Horizon strategique evalue.

        Returns:
            dict[str, Any]: Rapport complet de duel et de validation.
        """
        horizon = horizon.lower()
        timeframe = get_horizon_timeframe(horizon)
        eval_symbols = resolve_training_symbols(
            data_dir=self.data_dir,
            required_timeframes={timeframe},
            max_symbols=self._read_int_env("ARENA_MAX_SYMBOLS", 12),
        )
        if not eval_symbols:
            eval_symbols = ["XAUUSD", "EURUSD", "BTCUSD", "GBPUSD", "USDJPY"]

        challenger_path = self._resolve_model_path(challenger_id, horizon)
        champion_path = self._resolve_model_path(champion_id, horizon)
        if challenger_path is None:
            raise FileNotFoundError(f"Modele challenger introuvable: {challenger_id}")

        logger.info(
            "Arena %s: challenger=%s | champion=%s | symboles=%s",
            horizon,
            challenger_path,
            champion_path,
            eval_symbols,
        )

        challenger_metrics = self._evaluate_model(challenger_path, eval_symbols, horizon)
        champion_metrics = (
            self._evaluate_model(champion_path, eval_symbols, horizon)
            if champion_path is not None
            else self._empty_metrics()
        )

        challenger_score = self._score_metrics(challenger_metrics)
        champion_score = self._score_metrics(champion_metrics)
        min_score_edge = self._read_float_env("ARENA_MIN_SCORE_EDGE", 0.5)
        min_games = self._read_int_env("ARENA_MIN_GAMES", 12)
        min_symbols = min(
            len(eval_symbols),
            self._read_int_env("ARENA_MIN_SYMBOLS", 3),
        )
        sample_size_ok = (
            int(challenger_metrics.get("evaluation_games", 0)) >= min_games
            and int(challenger_metrics.get("evaluation_symbols", 0)) >= min_symbols
        )

        is_bootstrap = champion_path is None
        is_victory = sample_size_ok and (is_bootstrap or challenger_score > (champion_score + min_score_edge))

        report = {
            "timestamp": datetime.now().isoformat(),
            "combat_type": "MUZERO_HISTORICAL_ARENA",
            "horizon": horizon,
            "timeframe": timeframe,
            "eval_symbols": eval_symbols,
            "validation": {
                "sample_size_ok": sample_size_ok,
                "min_games": min_games,
                "min_symbols": min_symbols,
                "score_edge_required": min_score_edge,
            },
            "challenger": {
                "id": challenger_id,
                "path": str(challenger_path),
                "score": round(challenger_score, 4),
                "metrics": challenger_metrics,
            },
            "champion": {
                "id": champion_id,
                "path": str(champion_path) if champion_path is not None else None,
                "score": round(champion_score, 4),
                "metrics": champion_metrics,
            },
            "outcome": "VICTORY" if is_victory else "DEFEAT",
            "action_required": (
                "BOOTSTRAP_CHAMPION"
                if is_bootstrap and is_victory
                else ("HOT_SWAP_DEPLOY" if is_victory else ("EXTEND_VALIDATION" if not sample_size_ok else "KEEP_CURRENT"))
            ),
        }
        self.history.append(report)
        return report


if __name__ == "__main__":
    arena = Arena()
    print(arena.battle("muzero_global_latest", "gen_000_baseline", horizon="intraday"))
