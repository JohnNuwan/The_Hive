"""
Importe l'historique MT5 reel vers un dataset d'entrainement Shadow Learning.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for relative in ("src/shared", "src/eva-banker"):
    candidate = PROJECT_ROOT / relative
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from eva_banker.services.mt5 import MT5Service
from shared import get_settings
from shared.indicators import IndicatorFactory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("import_mt5_deal_history")


@dataclass
class PositionSnapshot:
    """Resume une position MT5 fermee avec ses metadonnees utiles."""

    position_id: int
    symbol: str
    action: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    volume: float
    pnl: float
    profit: float
    swap: float
    commission: float
    magic: int
    entry_ticket: int
    exit_ticket: int


def parse_args() -> argparse.Namespace:
    """Construit les arguments CLI de l'importeur.

    Returns:
        argparse.Namespace: Arguments resolves.
    """
    parser = argparse.ArgumentParser(
        description="Importe l'historique des positions MT5 fermees vers un dataset shadow.",
    )
    parser.add_argument("--days", type=int, default=365, help="Fenetre d'import en jours.")
    parser.add_argument("--timeframe", type=int, default=15, help="Timeframe des bougies source (1, 5, 15, 60, 1440).")
    parser.add_argument("--warmup-bars", type=int, default=260, help="Nombre de bougies de warmup pour les indicateurs.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "shadow_learning" / "imports"),
        help="Dossier de sortie du dataset JSONL.",
    )
    parser.add_argument(
        "--state-file",
        default=str(PROJECT_ROOT / "data" / "shadow_learning" / "mt5_import_state.json"),
        help="Fichier d'etat des positions deja importe es.",
    )
    parser.add_argument("--max-positions", type=int, default=0, help="Limite optionnelle du nombre de positions a importer.")
    parser.add_argument("--force", action="store_true", help="Ignore l'etat et reimporte les positions deja vues.")
    return parser.parse_args()


def load_state(path: Path) -> dict[str, Any]:
    """Charge l'etat d'import precedent.

    Args:
        path (Path): Fichier d'etat JSON.

    Returns:
        dict[str, Any]: Etat courant ou squelette vide.
    """
    if not path.exists():
        return {"imported_position_ids": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Etat d'import illisible, reinitialisation: %s", exc)
        return {"imported_position_ids": []}


def save_state(path: Path, state: dict[str, Any]) -> None:
    """Sauvegarde l'etat d'import.

    Args:
        path (Path): Fichier cible.
        state (dict[str, Any]): Etat a ecrire.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def group_closed_positions(deals: list[dict[str, Any]]) -> list[PositionSnapshot]:
    """Regroupe les deals MT5 en positions fermees.

    Args:
        deals (list[dict[str, Any]]): Deals bruts normalises.

    Returns:
        list[PositionSnapshot]: Positions closes deduites de l'historique.
    """
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for deal in deals:
        position_id = int(deal.get("position_id") or 0)
        if position_id <= 0:
            continue
        grouped[position_id].append(deal)

    positions: list[PositionSnapshot] = []
    for position_id, items in grouped.items():
        ordered = sorted(items, key=lambda item: item["time"])
        entries = [item for item in ordered if int(item.get("entry", -1)) == 0]
        exits = [item for item in ordered if int(item.get("entry", -1)) in {1, 2, 3}]
        if not entries or not exits:
            continue

        first_entry = entries[0]
        last_exit = exits[-1]
        total_profit = sum(float(item.get("profit", 0.0) or 0.0) for item in exits)
        total_swap = sum(float(item.get("swap", 0.0) or 0.0) for item in exits)
        total_commission = sum(float(item.get("commission", 0.0) or 0.0) for item in exits)
        total_pnl = total_profit + total_swap + total_commission
        total_volume = sum(float(item.get("volume", 0.0) or 0.0) for item in entries)
        if total_volume <= 0:
            total_volume = float(first_entry.get("volume", 0.0) or 0.0)

        positions.append(
            PositionSnapshot(
                position_id=position_id,
                symbol=str(first_entry["symbol"]),
                action=str(first_entry["type"]).upper(),
                entry_time=first_entry["time"],
                exit_time=last_exit["time"],
                entry_price=float(first_entry.get("price", 0.0) or 0.0),
                exit_price=float(last_exit.get("price", 0.0) or 0.0),
                volume=float(total_volume),
                pnl=float(total_pnl),
                profit=float(total_profit),
                swap=float(total_swap),
                commission=float(total_commission),
                magic=int(first_entry.get("magic", 0) or 0),
                entry_ticket=int(first_entry.get("ticket", 0) or 0),
                exit_ticket=int(last_exit.get("ticket", 0) or 0),
            )
        )

    return sorted(positions, key=lambda item: (item.exit_time, item.position_id))


def enrich_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Calcule les indicateurs utilises par le banker sur une serie OHLCV.

    Args:
        frame (pd.DataFrame): Bougies brutes indexees par date.

    Returns:
        pd.DataFrame: Frame enrichie avec les indicateurs necessaires.
    """
    enriched = frame.copy()
    closes = enriched["close"]
    highs = enriched["high"]
    lows = enriched["low"]
    volumes = enriched["tick_volume"]

    enriched["ema_200"] = IndicatorFactory.ema(closes, 200)
    enriched["rsi"] = IndicatorFactory.rsi(closes, 14)
    macd = IndicatorFactory.macd(closes)
    enriched["macd_hist"] = macd["histogram"]
    enriched["vwap"] = IndicatorFactory.vwap(highs, lows, closes, volumes)
    enriched["obv"] = IndicatorFactory.obv(closes, volumes)
    enriched["momentum"] = IndicatorFactory.momentum(closes)
    enriched["trix"] = IndicatorFactory.trix(closes)
    stochastic = IndicatorFactory.stochastic(highs, lows, closes)
    enriched["stoch_k"] = stochastic["percent_k"]
    enriched["stoch_d"] = stochastic["percent_d"]
    enriched["cci"] = IndicatorFactory.cci(highs, lows, closes)
    adx = IndicatorFactory.adx(highs, lows, closes)
    enriched["adx"] = adx["adx"]
    enriched["adx_plus_di"] = adx["plus_di"]
    enriched["adx_minus_di"] = adx["minus_di"]
    ichimoku = IndicatorFactory.ichimoku(highs, lows, closes)
    enriched["ichi_tenkan"] = ichimoku["tenkan_sen"]
    enriched["ichi_kijun"] = ichimoku["kijun_sen"]
    enriched["ichi_senkou_a"] = ichimoku["senkou_span_a"]
    enriched["ichi_senkou_b"] = ichimoku["senkou_span_b"]
    enriched["atr"] = IndicatorFactory.atr(highs, lows, closes)
    bollinger = IndicatorFactory.bollinger_bands(closes)
    band_range = (bollinger["upper"] - bollinger["lower"]).replace(0.0, pd.NA)
    enriched["bb_pct"] = ((closes - bollinger["lower"]) / band_range).fillna(0.5)
    enriched["return_1"] = closes.pct_change().fillna(0.0)

    return enriched.bfill().ffill().fillna(0.0)


def build_observation(row: pd.Series, symbol: str, timeframe: int) -> dict[str, Any]:
    """Construit une observation shadow alignee avec le format live.

    Args:
        row (pd.Series): Ligne enrichie de la frame.
        symbol (str): Symbole du trade.
        timeframe (int): Timeframe utilise pour les bougies.

    Returns:
        dict[str, Any]: Observation JSON serialisable.
    """
    close_price = float(row["close"])
    spread_value = float(row.get("spread", 0.0) or 0.0)
    horizon = "intraday"
    if timeframe >= 1440:
        horizon = "swing"
    elif timeframe <= 5:
        horizon = "scalp"

    return {
        "symbol": symbol,
        "horizon": horizon,
        "price": close_price,
        "timestamp": row.name.isoformat(),
        "latest_candle": {
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": close_price,
            "tick_volume": float(row["tick_volume"]),
            "spread": spread_value,
        },
        "indicators": {
            "EMA_200": float(row["ema_200"]),
            "RSI": float(row["rsi"]),
            "MACD_Hist": float(row["macd_hist"]),
            "VWAP": float(row["vwap"]),
            "OBV": float(row["obv"]),
            "Momentum": float(row["momentum"]),
            "TRIX": float(row["trix"]),
            "Stoch_K": float(row["stoch_k"]),
            "Stoch_D": float(row["stoch_d"]),
            "CCI": float(row["cci"]),
            "ADX": float(row["adx"]),
            "ADX_Plus_DI": float(row["adx_plus_di"]),
            "ADX_Minus_DI": float(row["adx_minus_di"]),
            "Ichi_Tenkan": float(row["ichi_tenkan"]),
            "Ichi_Kijun": float(row["ichi_kijun"]),
            "Ichi_Senkou_A": float(row["ichi_senkou_a"]),
            "Ichi_Senkou_B": float(row["ichi_senkou_b"]),
            "ATR": float(row["atr"]),
            "BB_Pct": float(row["bb_pct"]),
            "Return_1": float(row["return_1"]),
            "Spread_Norm": spread_value / max(close_price, 1e-8),
        },
    }


async def fetch_symbol_frames(
    mt5_service: MT5Service,
    positions: list[PositionSnapshot],
    timeframe: int,
    warmup_bars: int,
) -> dict[str, pd.DataFrame]:
    """Charge les bougies necessaires et calcule les indicateurs par symbole.

    Args:
        mt5_service (MT5Service): Service MT5 connecte.
        positions (list[PositionSnapshot]): Positions a enrichir.
        timeframe (int): Timeframe source.
        warmup_bars (int): Nombre de bougies de chauffe.

    Returns:
        dict[str, pd.DataFrame]: Frames enrichies indexees par symbole.
    """
    frames: dict[str, pd.DataFrame] = {}
    if not positions:
        return frames

    grouped: dict[str, list[PositionSnapshot]] = defaultdict(list)
    for position in positions:
        grouped[position.symbol].append(position)

    warmup = timedelta(minutes=timeframe * warmup_bars)
    horizon_extension = timedelta(minutes=max(timeframe, 1) * 3)

    for symbol, symbol_positions in grouped.items():
        range_start = min(position.entry_time for position in symbol_positions) - warmup
        range_end = max(position.exit_time for position in symbol_positions) + horizon_extension
        candles = await mt5_service.get_candles_range(symbol, timeframe, range_start, range_end)
        if not candles:
            logger.warning("Aucune bougie importee pour %s sur la plage demandee.", symbol)
            continue

        frame = pd.DataFrame(candles)
        frame["time"] = pd.to_datetime(frame["time"])
        frame = frame.sort_values("time").drop_duplicates(subset=["time"]).set_index("time")
        frames[symbol] = enrich_frame(frame)
        logger.info("Bougies chargees pour %s: %s lignes.", symbol, len(frames[symbol]))

    return frames


def build_position_transitions(
    position: PositionSnapshot,
    frame: pd.DataFrame,
    timeframe: int,
) -> list[dict[str, Any]]:
    """Construit les transitions d'une position fermee.

    Args:
        position (PositionSnapshot): Position MT5 fermee.
        frame (pd.DataFrame): Frame enrichie du symbole.
        timeframe (int): Timeframe de reference.

    Returns:
        list[dict[str, Any]]: Transitions JSONL du trade.
    """
    episode_frame = frame.loc[(frame.index >= position.entry_time) & (frame.index <= position.exit_time)].copy()
    if len(episode_frame) < 2:
        around = frame.loc[frame.index <= position.exit_time].tail(2).copy()
        if len(around) < 2:
            return []
        episode_frame = around

    direction = 1.0 if position.action == "BUY" else -1.0
    cost_pct = 0.0
    if position.entry_price > 0:
        cost_pct = abs(position.swap + position.commission) / position.entry_price * 100.0

    rows = list(episode_frame.iterrows())
    total_steps = max(1, len(rows) - 1)
    episode_id = f"mt5:{position.position_id}"
    transitions: list[dict[str, Any]] = []

    for step_index in range(len(rows) - 1):
        current_ts, current_row = rows[step_index]
        next_ts, next_row = rows[step_index + 1]
        current_price = float(current_row["close"])
        next_price = float(next_row["close"])
        step_reward = direction * ((next_price - current_price) / max(current_price, 1e-8)) * 100.0
        if step_index == 0:
            step_reward -= cost_pct

        action_type = position.action if step_index == 0 else "HOLD"
        done = step_index == len(rows) - 2
        if done:
            action_type = "CLOSE"

        transitions.append(
            {
                "timestamp": current_ts.isoformat(),
                "observation": build_observation(current_row, position.symbol, timeframe),
                "action": {
                    "type": action_type,
                    "volume": position.volume,
                    "symbol": position.symbol,
                },
                "reward": float(step_reward),
                "next_observation": build_observation(next_row, position.symbol, timeframe),
                "metadata": {
                    "episode_id": episode_id,
                    "source": "mt5_history",
                    "position_id": position.position_id,
                    "symbol": position.symbol,
                    "magic": position.magic,
                    "entry_ticket": position.entry_ticket,
                    "exit_ticket": position.exit_ticket,
                    "entry_time": position.entry_time.isoformat(),
                    "exit_time": position.exit_time.isoformat(),
                    "raw_pnl": position.pnl,
                    "profit": position.profit,
                    "swap": position.swap,
                    "commission": position.commission,
                    "step_index": step_index,
                    "steps_total": total_steps,
                },
                "done": done,
            }
        )

    return transitions


async def run_import(args: argparse.Namespace) -> dict[str, Any]:
    """Execute l'import complet MT5 -> dataset shadow.

    Args:
        args (argparse.Namespace): Arguments resolves.

    Returns:
        dict[str, Any]: Resume de l'import realise.
    """
    settings = get_settings()
    mt5_service = MT5Service(
        mock_mode=False,
        login=settings.mt5_login,
        password=settings.mt5_password.get_secret_value(),
        server=settings.mt5_server,
    )

    connected = await mt5_service.connect()
    if not connected or mt5_service.mock_mode:
        raise RuntimeError("Connexion MT5 indisponible. L'import historique reel est impossible.")

    state_path = Path(args.state_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = load_state(state_path)
    imported_ids = {int(item) for item in state.get("imported_position_ids", [])}

    to_dt = datetime.now()
    from_dt = to_dt - timedelta(days=max(1, args.days))
    deals = await mt5_service.get_deal_history(from_dt, to_dt, closed_only=False)
    positions = group_closed_positions(deals)

    candidates = [
        position
        for position in positions
        if args.force or position.position_id not in imported_ids
    ]
    if args.max_positions > 0:
        candidates = candidates[: args.max_positions]

    logger.info(
        "Historique MT5: %s deals, %s positions fermees, %s candidates a l'import.",
        len(deals),
        len(positions),
        len(candidates),
    )

    frames = await fetch_symbol_frames(mt5_service, candidates, args.timeframe, args.warmup_bars)
    transitions: list[dict[str, Any]] = []
    imported_positions: list[int] = []

    for position in candidates:
        frame = frames.get(position.symbol)
        if frame is None or frame.empty:
            logger.warning("Import ignore pour %s/%s: frame indisponible.", position.symbol, position.position_id)
            continue

        episode_transitions = build_position_transitions(position, frame, args.timeframe)
        if not episode_transitions:
            logger.warning("Import ignore pour %s/%s: episode vide.", position.symbol, position.position_id)
            continue

        transitions.extend(episode_transitions)
        imported_positions.append(position.position_id)

    output_file: str | None = None
    if transitions:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"mt5_history_{timestamp}.jsonl"
        with output_path.open("w", encoding="utf-8") as file_obj:
            for item in transitions:
                json.dump(item, file_obj, ensure_ascii=False)
                file_obj.write("\n")
        output_file = str(output_path)
        logger.info("Dataset MT5 ecrit dans %s (%s transitions).", output_path, len(transitions))

    state["imported_position_ids"] = sorted(imported_ids.union(imported_positions))
    state["last_import_at"] = datetime.now().isoformat()
    state["last_window"] = {"from": from_dt.isoformat(), "to": to_dt.isoformat()}
    save_state(state_path, state)

    return {
        "deals_total": len(deals),
        "closed_positions_total": len(positions),
        "positions_imported": len(imported_positions),
        "transitions_written": len(transitions),
        "output_file": output_file,
        "state_file": str(state_path),
        "symbols": sorted({position.symbol for position in candidates}),
    }


def main() -> None:
    """Point d'entree CLI."""
    args = parse_args()
    summary = asyncio.run(run_import(args))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
