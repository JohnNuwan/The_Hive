"""Entraine le GNN multi-timeframe sur l'historique reel disponible."""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

package_root = Path(__file__).resolve().parents[1]
shared_root = package_root.parent / 'shared'
for candidate in (package_root, shared_root):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from eva_lab.models.gnn_model import TFTGNNModel
from eva_lab.gold_cpu_prep import (
    build_gnn_dataset as build_gnn_cpu_dataset,
    load_gnn_dataset_cache,
    save_gnn_dataset_cache,
)
from eva_lab.training_status import append_training_log, mark_step_running
from eva_lab.training_utils import MTF_HORIZONS, build_inventory_report, get_gnn_model_kwargs, load_history_frame, resolve_training_symbols
from shared.indicators import IndicatorFactory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eva_lab.train_gnn")

EPOCHS = int(os.getenv("TRAIN_GNN_EPOCHS", "500"))
BATCH_SIZE = int(os.getenv("TRAIN_GNN_BATCH_SIZE", "64"))
CHECKPOINT_EVERY = int(os.getenv("TRAIN_GNN_CHECKPOINT_EVERY", "25"))
MAX_SYMBOLS = int(os.getenv("TRAIN_GNN_MAX_SYMBOLS", "0"))
MODEL_DIR = Path(os.getenv("TRAIN_GNN_MODEL_DIR", "data/models"))
MODEL_PATH = MODEL_DIR / os.getenv("TRAIN_GNN_MODEL_NAME", "gnn_master.pth")
METRICS_PATH = MODEL_DIR / os.getenv("TRAIN_GNN_METRICS_NAME", "gnn_master_metrics.json")
CLASSES = ["BULLISH", "BEARISH", "RANGING"]
FOCUS_SYMBOL = str(os.getenv("TRAIN_GNN_FOCUS_SYMBOL", "")).strip() or None
CONTEXT_SYMBOLS = [
    item.strip()
    for item in str(os.getenv("TRAIN_GNN_CONTEXT_SYMBOLS", "")).split(",")
    if item.strip()
]
DEPLOYMENT_CLASS = str(os.getenv("TRAIN_GNN_DEPLOYMENT_CLASS", "")).strip() or "consultative"


def get_label(current_price: float, future_price: float, atr: float) -> int:
    """Retourne le label de tendance future pour un echantillon."""
    delta = future_price - current_price
    threshold = max(atr * 0.4, current_price * 0.0005)
    if delta > threshold:
        return 0
    if delta < -threshold:
        return 1
    return 2



def compute_feature_sequences(frame, seq_len: int, future_n: int):
    """Construit les sequences de features et leurs labels associes."""
    closes = frame["close"].astype(float)
    highs = frame["high"].astype(float)
    lows = frame["low"].astype(float)
    volumes = frame["tick_volume"].astype(float)
    opens = frame["open"].astype(float)

    rsi = IndicatorFactory.rsi(closes, 14)
    adx = IndicatorFactory.adx(highs, lows, closes, 14)["adx"]
    vwap = IndicatorFactory.vwap(highs, lows, closes, volumes)
    macd_hist = IndicatorFactory.macd(closes)["histogram"]
    atr = IndicatorFactory.atr(highs, lows, closes, 14)
    bb_pct = IndicatorFactory.bollinger_bands(closes)["pct_b"]

    features: list[list[list[float]]] = []
    labels: list[int] = []
    start_idx = max(50, seq_len)

    for current_idx in range(start_idx, len(frame) - future_n):
        seq: list[list[float]] = []
        start_seq = current_idx - seq_len + 1
        for idx in range(start_seq, current_idx + 1):
            price = float(closes.iloc[idx])
            previous_price = float(closes.iloc[idx - 1]) if idx > 0 else price
            high_price = float(highs.iloc[idx])
            low_price = float(lows.iloc[idx])
            open_price = float(opens.iloc[idx])
            avg_volume = float(volumes.iloc[max(0, idx - 10):idx].mean()) if idx > 0 else 0.0

            row = [
                (price / previous_price) - 1.0 if previous_price else 0.0,
                float(rsi.iloc[idx]) / 100.0 if not np.isnan(rsi.iloc[idx]) else 0.5,
                float(adx.iloc[idx]) / 100.0 if not np.isnan(adx.iloc[idx]) else 0.2,
                float(macd_hist.iloc[idx]) / price if (not np.isnan(macd_hist.iloc[idx]) and price) else 0.0,
                float(bb_pct.iloc[idx]) if not np.isnan(bb_pct.iloc[idx]) else 0.5,
                float(volumes.iloc[idx]) / (avg_volume + 1e-5) if avg_volume else 1.0,
                (price - float(vwap.iloc[idx])) / price if (not np.isnan(vwap.iloc[idx]) and price) else 0.0,
                float(atr.iloc[idx]) / price if (not np.isnan(atr.iloc[idx]) and price) else 0.0,
                (price - low_price) / (high_price - low_price + 1e-8),
                (high_price - max(price, open_price)) / price if price else 0.0,
                (min(price, open_price) - low_price) / price if price else 0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
            seq.append(row)

        future_price = float(closes.iloc[current_idx + future_n])
        current_price = float(closes.iloc[current_idx])
        atr_value = float(atr.iloc[current_idx]) if not np.isnan(atr.iloc[current_idx]) else current_price * 0.001
        labels.append(get_label(current_price, future_price, atr_value))
        features.append(seq)

    return features, labels



def build_dataset(symbols: list[str]):
    """Assemble le dataset multi-actifs pour les trois horizons."""
    return build_gnn_cpu_dataset(symbols)



def build_graph(num_nodes: int) -> torch.Tensor:
    """Construit un graphe complet entre tous les actifs entraines."""
    rows: list[int] = []
    cols: list[int] = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)
    if not rows:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([rows, cols], dtype=torch.long)



def train_gnn() -> dict[str, object]:
    """Execute l'entrainement complet du GNN multi-horizon."""
    inventory = build_inventory_report()
    logger.info("Inventaire historique detecte: %s", inventory)
    append_training_log("GNN: initialisation de l'entrainement multi-timeframe.", source="gnn")
    mark_step_running("gnn", phase="initialisation")

    symbols = resolve_training_symbols(
        required_timeframes={"M5", "H1", "D1"},
        max_symbols=MAX_SYMBOLS,
        override_env_names=["TRAIN_GNN_SYMBOLS"],
    )
    if not symbols:
        raise RuntimeError("Aucun symbole exploitable pour l'entrainement GNN.")

    logger.info("Univers GNN: %s", symbols)
    cache_payload = load_gnn_dataset_cache(symbols)
    cache_path_used: str | None = None
    if cache_payload:
        cache_path_used = str(cache_payload.get("cache_path") or "")
        valid_symbols = list(cache_payload.get("valid_symbols") or [])
        dataset = {}
        for symbol in valid_symbols:
            symbol_payload = {}
            for timeframe, timeframe_payload in dict(cache_payload.get("dataset", {}).get(symbol) or {}).items():
                symbol_payload[timeframe] = {
                    "features": torch.tensor(timeframe_payload["features"], dtype=torch.float32),
                    "labels": torch.tensor(timeframe_payload["labels"], dtype=torch.long),
                }
            dataset[symbol] = symbol_payload
        inventory = dict(cache_payload.get("inventory") or inventory)
        logger.info("Cache CPU GNN charge depuis %s.", cache_path_used)
        append_training_log(
            f"GNN: cache CPU charge depuis {Path(cache_path_used).name}.",
            source="gnn",
        )
    else:
        dataset_np, valid_symbols = build_dataset(symbols)
        if valid_symbols:
            cache_path = save_gnn_dataset_cache(
                symbols=symbols,
                dataset=dataset_np,
                valid_symbols=valid_symbols,
                inventory=inventory,
            )
            cache_path_used = str(cache_path)
            logger.info("Cache CPU GNN ecrit dans %s.", cache_path)
            append_training_log(
                f"GNN: cache CPU ecrit dans {cache_path.name}.",
                source="gnn",
            )
        dataset = {}
        for symbol in valid_symbols:
            symbol_payload = {}
            for timeframe, timeframe_payload in dict(dataset_np.get(symbol) or {}).items():
                symbol_payload[timeframe] = {
                    "features": torch.tensor(timeframe_payload["features"], dtype=torch.float32),
                    "labels": torch.tensor(timeframe_payload["labels"], dtype=torch.long),
                }
            dataset[symbol] = symbol_payload
    if not valid_symbols:
        raise RuntimeError("Le dataset GNN est vide apres validation.")
    append_training_log(
        f"GNN: {len(valid_symbols)} symboles valides prepares.",
        source="gnn",
    )

    num_samples = min(
        dataset[symbol][timeframe]["labels"].size(0)
        for symbol in valid_symbols
        for timeframe in MTF_HORIZONS.keys()
    )
    logger.info("Nombre d'echantillons synchronises retenus: %s", num_samples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Peripherique GNN: %s", device)

    model = TFTGNNModel(**get_gnn_model_kwargs()).to(device)
    if MODEL_PATH.exists():
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            logger.info("Reprise depuis %s", MODEL_PATH)
        except Exception as exc:
            logger.warning("Checkpoint GNN ignore: %s", exc)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(EPOCHS, 1), eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    edge_index = build_graph(len(valid_symbols)).to(device)
    torch.backends.cudnn.benchmark = True

    last_metrics = {
        "loss": None,
        "scalp_accuracy": 0.0,
        "intraday_accuracy": 0.0,
        "swing_accuracy": 0.0,
    }

    for epoch in range(EPOCHS):
        mark_step_running(
            "gnn",
            phase="optimisation",
            epoch_current=epoch + 1,
            epoch_total=EPOCHS,
            symbol_total=len(valid_symbols),
        )
        model.train()
        indices = torch.randperm(num_samples)
        total_loss = 0.0
        correct_scalp = 0
        correct_intraday = 0
        correct_swing = 0
        total_nodes = 0
        batch_count = 0

        for batch_start in range(0, num_samples, BATCH_SIZE):
            batch_indices = indices[batch_start:batch_start + BATCH_SIZE]
            optimizer.zero_grad()
            batch_loss = 0.0

            for sample_idx in batch_indices.tolist():
                ts_m5 = []
                ts_h1 = []
                ts_d1 = []
                lbl_scalp = []
                lbl_intraday = []
                lbl_swing = []

                for symbol in valid_symbols:
                    ts_m5.append(dataset[symbol]["M5"]["features"][sample_idx].to(device))
                    ts_h1.append(dataset[symbol]["H1"]["features"][sample_idx].to(device))
                    ts_d1.append(dataset[symbol]["D1"]["features"][sample_idx].to(device))
                    lbl_scalp.append(dataset[symbol]["M5"]["labels"][sample_idx].to(device))
                    lbl_intraday.append(dataset[symbol]["H1"]["labels"][sample_idx].to(device))
                    lbl_swing.append(dataset[symbol]["D1"]["labels"][sample_idx].to(device))

                scalp_targets = torch.stack(lbl_scalp)
                intraday_targets = torch.stack(lbl_intraday)
                swing_targets = torch.stack(lbl_swing)

                outputs = model(ts_m5, ts_h1, ts_d1, edge_index)
                loss = (
                    criterion(outputs["scalp"], scalp_targets)
                    + criterion(outputs["intraday"], intraday_targets)
                    + criterion(outputs["swing"], swing_targets)
                ) / 3.0
                loss.backward()
                batch_loss += float(loss.item())

                correct_scalp += (torch.argmax(outputs["scalp"], dim=1) == scalp_targets).sum().item()
                correct_intraday += (torch.argmax(outputs["intraday"], dim=1) == intraday_targets).sum().item()
                correct_swing += (torch.argmax(outputs["swing"], dim=1) == swing_targets).sum().item()
                total_nodes += len(valid_symbols)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += batch_loss / max(len(batch_indices), 1)
            batch_count += 1

        scheduler.step()
        avg_loss = total_loss / max(batch_count, 1)
        last_metrics = {
            "loss": avg_loss,
            "scalp_accuracy": round(100.0 * correct_scalp / max(total_nodes, 1), 2),
            "intraday_accuracy": round(100.0 * correct_intraday / max(total_nodes, 1), 2),
            "swing_accuracy": round(100.0 * correct_swing / max(total_nodes, 1), 2),
        }
        logger.info(
            "Epoch %03d/%03d | loss=%.4f | scalp=%.2f%% | intraday=%.2f%% | swing=%.2f%%",
            epoch + 1,
            EPOCHS,
            avg_loss,
            last_metrics["scalp_accuracy"],
            last_metrics["intraday_accuracy"],
            last_metrics["swing_accuracy"],
        )
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == EPOCHS:
            append_training_log(
                "GNN: epoch "
                f"{epoch + 1}/{EPOCHS} | "
                f"loss={avg_loss:.4f} | "
                f"scalp={last_metrics['scalp_accuracy']:.2f}%",
                source="gnn",
            )

        if CHECKPOINT_EVERY > 0 and (epoch + 1) % CHECKPOINT_EVERY == 0:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            checkpoint_path = MODEL_DIR / f"gnn_ckpt_ep{epoch + 1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("Checkpoint GNN sauvegarde: %s", checkpoint_path)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    report = {
        "device": str(device),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "symbols": valid_symbols,
        "focus_symbols": valid_symbols,
        "focus_symbol": FOCUS_SYMBOL or (valid_symbols[0] if valid_symbols else None),
        "context_symbols": CONTEXT_SYMBOLS or [
            symbol for symbol in valid_symbols if symbol != (FOCUS_SYMBOL or (valid_symbols[0] if valid_symbols else None))
        ],
        "deployment_class": DEPLOYMENT_CLASS,
        "cache_path": cache_path_used,
        "samples": num_samples,
        "inventory": inventory,
        **last_metrics,
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Modele GNN sauvegarde dans %s", MODEL_PATH)
    logger.info("Rapport GNN sauvegarde dans %s", METRICS_PATH)
    return report


if __name__ == "__main__":
    summary = train_gnn()
    logger.info("Entrainement GNN termine: %s", summary)


