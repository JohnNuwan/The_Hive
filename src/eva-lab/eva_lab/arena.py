"""Arena Darwinienne pour l'evaluation des modeles MuZero sur historique reel."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from eva_lab.muzero.config import MuZeroConfigV3
from eva_lab.muzero.environment import TradingEnvironment
from eva_lab.muzero.jax_agent import JAXMuZeroAgent
from eva_lab.training_utils import build_muzero_market_data, get_horizon_timeframe, load_history_frame, resolve_training_symbols

logger = logging.getLogger(__name__)


class Arena:
    """Execute des combats entre challenger et champion sur des historiques reels."""

    def __init__(
        self,
        weights_dir: str = "data/muzero/weights",
        data_dir: str = "data/history",
    ):
        self.history: list[dict[str, Any]] = []
        self.weights_dir = Path(weights_dir)
        self.data_dir = data_dir

    def _resolve_model_path(self, model_id: str, horizon: str) -> Path | None:
        """Retourne le chemin d'un modele, avec fallback sur les champions historiques."""
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

    def _build_environment(self, symbol: str, config: MuZeroConfigV3, horizon: str) -> TradingEnvironment | None:
        """Construit un environnement MuZero a partir d'un historique reel."""
        timeframe = get_horizon_timeframe(horizon)
        frame = load_history_frame(symbol, timeframe, self.data_dir)
        if frame is None:
            logger.warning("Arena: historique absent pour %s sur %s.", symbol, timeframe)
            return None

        market_data = build_muzero_market_data(frame.tail(2500))
        if market_data.shape[0] < 220:
            logger.warning("Arena: historique insuffisant pour %s sur %s.", symbol, timeframe)
            return None

        max_steps = min(config.max_moves, market_data.shape[0] - 101)
        return TradingEnvironment(data=market_data, symbol=symbol, config=config, max_steps=max_steps)

    def _evaluate_model(self, weights_path: Path, symbols: list[str], horizon: str) -> dict[str, float]:
        """Evalue un modele donne sur l'univers historique selectionne."""
        if not weights_path.exists():
            logger.warning("Poids absents pour l'evaluation: %s", weights_path)
            return {"profit_factor": 0.0, "return_pct": 0.0, "win_rate": 0.0}

        config = MuZeroConfigV3(horizon=horizon, primary_timeframe=get_horizon_timeframe(horizon), symbols=symbols)
        try:
            agent = JAXMuZeroAgent(config)
            agent.load(str(weights_path))
        except Exception as exc:
            logger.error("Chargement MuZero impossible pour %s: %s", weights_path, exc)
            return {"profit_factor": 0.0, "return_pct": 0.0, "win_rate": 0.0}

        total_return = 0.0
        total_win_rate = 0.0
        valid_symbols = 0

        for symbol in symbols:
            env = self._build_environment(symbol, config, horizon)
            if env is None:
                continue
            agent.play_game(env, exploration=False)
            summary = env.get_summary()
            total_return += float(summary.get("return_pct", 0.0))
            total_win_rate += float(summary.get("win_rate", 0.0)) * 100.0
            valid_symbols += 1

        if valid_symbols == 0:
            return {"profit_factor": 0.0, "return_pct": 0.0, "win_rate": 0.0}

        average_return = total_return / valid_symbols
        return {
            "profit_factor": max(0.1, 1.0 + average_return / 100.0),
            "return_pct": average_return,
            "win_rate": total_win_rate / valid_symbols,
        }

    def battle(
        self,
        challenger_id: str,
        champion_id: str = "gen_000_baseline",
        horizon: str = "intraday",
    ) -> dict[str, Any]:
        """Compare deux generations MuZero et retourne le verdict ADN."""
        horizon = horizon.lower()
        timeframe = get_horizon_timeframe(horizon)
        eval_symbols = resolve_training_symbols(
            data_dir=self.data_dir,
            required_timeframes={timeframe},
            max_symbols=int(os.getenv("ARENA_MAX_SYMBOLS", "0")),
        )
        if not eval_symbols:
            eval_symbols = ["XAUUSD", "EURUSD", "BTCUSD"]

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
            else {"profit_factor": 0.0, "return_pct": 0.0, "win_rate": 0.0}
        )

        challenger_score = challenger_metrics["return_pct"]
        champion_score = champion_metrics["return_pct"]
        min_edge = float(os.getenv("ARENA_MIN_EDGE_PCT", "1.0"))
        is_bootstrap = champion_path is None
        is_victory = is_bootstrap or challenger_score > (champion_score + min_edge)

        report = {
            "timestamp": datetime.now().isoformat(),
            "combat_type": "MUZERO_HISTORICAL_ARENA",
            "horizon": horizon,
            "timeframe": timeframe,
            "eval_symbols": eval_symbols,
            "challenger": {
                "id": challenger_id,
                "path": str(challenger_path),
                "score": round(challenger_score, 2),
                "metrics": challenger_metrics,
            },
            "champion": {
                "id": champion_id,
                "path": str(champion_path) if champion_path is not None else None,
                "score": round(champion_score, 2),
                "metrics": champion_metrics,
            },
            "outcome": "VICTORY" if is_victory else "DEFEAT",
            "action_required": "BOOTSTRAP_CHAMPION" if is_bootstrap else ("HOT_SWAP_DEPLOY" if is_victory else "KEEP_CURRENT"),
        }
        self.history.append(report)
        return report


if __name__ == "__main__":
    arena = Arena()
    print(arena.battle("muzero_global_latest", "gen_000_baseline", horizon="intraday"))
