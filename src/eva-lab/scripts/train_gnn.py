"""
🐝 THE HIVE: MTF OMNI-GNN Training Script (Sprint 19)
Trains the Multi-Timeframe GNN with 3 strategy heads: Scalp, Intraday, Swing.
"""

import asyncio
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from colorama import init, Fore, Style
import numpy as np
import sys

init()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Hive-MTF-GNN-Trainer")

# Add src to path
src_path = str(Path(__file__).parent.parent.parent.absolute())
if src_path not in sys.path:
    sys.path.append(src_path)

from eva_lab.models.gnn_model import TFTGNNModel
from eva_banker.services.mt5 import MT5Service
from shared.indicators import IndicatorFactory

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "US30.cash", "US100.cash", "GER40.cash",
    "BTCUSD", "ETHUSD", "XAUUSD", "AUDUSD", "NZDUSD", "USDCHF", "USDCAD",
    "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY",
    "EURAUD", "EURCAD", "EURCHF", "GBPAUD", "GBPCAD", "GBPCHF",
    "XAGUSD", "SOLUSD"
]

# Multi-Timeframe Settings
MTF = {
    "M5":  {"tf": 5,    "count": 1000, "seq_len": 15, "future": 12},   # +1H look-ahead
    "H1":  {"tf": 60,   "count": 500,  "seq_len": 15, "future": 24},   # +1D look-ahead
    "D1":  {"tf": 1440, "count": 200,  "seq_len": 10, "future": 7},    # +1W look-ahead
}

ASSET_DIM = 20
TEMPORAL_DIM = 32
HIDDEN_DIM = 64
NUM_CLASSES = 3
EPOCHS = 60
BATCH_SIZE = 24

MODEL_DIR = Path("data/models")
MODEL_PATH = MODEL_DIR / "gnn_master.pth"

CLASSES = ["BULLISH", "BEARISH", "RANGING"]


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════
def get_label(current_price, future_price, atr):
    """Label: 0=BULLISH, 1=BEARISH, 2=RANGING"""
    change = future_price - current_price
    threshold = max(atr * 0.4, current_price * 0.0005)
    if change > threshold:
        return 0
    elif change < -threshold:
        return 1
    return 2


def compute_features(candles, seq_len, start_i, end_i, atr, rsi, adx, vwap, macd_hist, bb_pct, closes, highs, lows, volumes):
    """Extracts a feature matrix of shape [seq_len, ASSET_DIM]."""
    seq_features = []
    for j in range(max(0, start_i), end_i + 1):
        f = [
            closes[j] / closes[j-1] - 1.0 if j > 0 else 0.0,
            rsi.iloc[j] / 100.0 if not np.isnan(rsi.iloc[j]) else 0.5,
            adx.iloc[j] / 100.0 if not np.isnan(adx.iloc[j]) else 0.2,
            macd_hist.iloc[j] / closes[j] if (not np.isnan(macd_hist.iloc[j]) and closes[j] != 0) else 0.0,
            bb_pct.iloc[j] if not np.isnan(bb_pct.iloc[j]) else 0.5,
            volumes[j] / (sum(volumes[max(0,j-10):j]) / 10 + 1e-5),
            (closes[j] - vwap.iloc[j]) / closes[j] if not np.isnan(vwap.iloc[j]) else 0.0,
            atr.iloc[j] / closes[j] if not np.isnan(atr.iloc[j]) else 0.0,
            (closes[j] - lows[j]) / (highs[j] - lows[j] + 1e-8),
            (highs[j] - max(closes[j], candles[j]["open"])) / closes[j] if closes[j] > 0 else 0.0,
            (min(closes[j], candles[j]["open"]) - lows[j]) / closes[j] if closes[j] > 0 else 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Padding to ASSET_DIM=20
        ]
        seq_features.append(f)
    return seq_features


async def fetch_mtf_data(mt5: MT5Service):
    """Downloads M5, H1, D1 candles for all symbols and builds feature tensors."""
    logger.info(f"📥 Fetching MTF data for {len(SYMBOLS)} symbols (M5, H1, D1)...")
    
    dataset = {}
    valid_symbols = []
    
    for symbol in SYMBOLS:
        sym_data = {}
        all_ok = True
        
        for horizon_key, cfg in MTF.items():
            candles = await mt5.get_recent_candles(symbol, timeframe=cfg["tf"], count=cfg["count"])
            
            if not candles or len(candles) < cfg["count"] * 0.7:
                logger.warning(f"⚠️ {symbol} {horizon_key}: Insufficient data ({len(candles) if candles else 0}), skipping.")
                all_ok = False
                break
            
            closes = [c["close"] for c in candles]
            highs = [c["high"] for c in candles]
            lows = [c["low"] for c in candles]
            volumes = [c["tick_volume"] for c in candles]
            
            rsi = IndicatorFactory.rsi(closes, 14)
            adx_d = IndicatorFactory.adx(highs, lows, closes, 14)
            adx = adx_d["adx"]
            vwap = IndicatorFactory.vwap(highs, lows, closes, volumes)
            macd_d = IndicatorFactory.macd(closes)
            macd_hist = macd_d["histogram"]
            atr = IndicatorFactory.atr(highs, lows, closes, 14)
            bb_d = IndicatorFactory.bollinger_bands(closes)
            bb_pct = bb_d["pct_b"]
            
            seq_len = cfg["seq_len"]
            future_n = cfg["future"]
            start_idx = 50
            
            features, labels = [], []
            for i in range(start_idx, len(candles) - future_n):
                seq = compute_features(
                    candles, seq_len,
                    i - seq_len + 1, i,
                    atr, rsi, adx, vwap, macd_hist, bb_pct,
                    closes, highs, lows, volumes
                )
                # Pad sequence if needed
                while len(seq) < seq_len:
                    seq.insert(0, seq[0] if seq else [0.0]*ASSET_DIM)
                features.append(seq[-seq_len:])
                
                cur_p = closes[i]
                fut_p = closes[i + future_n]
                cur_atr = atr.iloc[i] if not np.isnan(atr.iloc[i]) else 0.001 * cur_p
                labels.append(get_label(cur_p, fut_p, cur_atr))
            
            sym_data[horizon_key] = {
                "features": torch.tensor(features, dtype=torch.float32),  # [N, seq_len, ASSET_DIM]
                "labels": torch.tensor(labels, dtype=torch.long)          # [N]
            }
        
        if all_ok and sym_data:
            dataset[symbol] = sym_data
            valid_symbols.append(symbol)
            logger.info(f"✅ {symbol}: M5={len(dataset[symbol]['M5']['labels'])}, H1={len(dataset[symbol]['H1']['labels'])}, D1={len(dataset[symbol]['D1']['labels'])} samples")
    
    return dataset, valid_symbols


def build_graph(num_nodes):
    rows, cols = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)
    if rows:
        return torch.tensor([rows, cols], dtype=torch.long)
    return torch.empty((2, 0), dtype=torch.long)


async def train_gnn():
    print(f"\n{Fore.MAGENTA}═══════════════════════════════════════════════════════{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}🐝 THE HIVE: MTF OMNI-GNN TRAINING (Scalp+Intraday+Swing) 🐝{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}═══════════════════════════════════════════════════════{Style.RESET_ALL}\n")
    
    mt5 = MT5Service(mock_mode=False, login=1512664750, server="FTMO-Demo")
    await mt5.connect()
    
    try:
        dataset, valid_symbols = await fetch_mtf_data(mt5)
        if not dataset:
            logger.error("No valid dataset. Does MT5 have these symbols subscribed?")
            return
        
        na = len(valid_symbols)
        # Use minimum samples across all symbols and horizons
        num_samples = min(
            dataset[s][h]["labels"].size(0)
            for s in valid_symbols
            for h in MTF.keys()
        )
        
        logger.info(f"\n🌐 Graph: {na} Assets, Fully Connected")
        logger.info(f"📊 Training: {num_samples} time steps × {na} assets × 3 horizons")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"⚙️ Device: {device}")
        
        model = TFTGNNModel(asset_dim=ASSET_DIM, temporal_dim=TEMPORAL_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES)
        
        if MODEL_PATH.exists():
            try:
                model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                logger.info(f"✅ Resuming from checkpoint {MODEL_PATH}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
        
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        edge_index = build_graph(na).to(device)
        
        model.train()
        for epoch in range(EPOCHS):
            total_loss = 0
            correct_scalp = correct_intraday = correct_swing = 0
            total_nodes = 0
            
            indices = torch.randperm(num_samples)
            
            for i in range(0, num_samples, BATCH_SIZE):
                batch_idx = indices[i:i+BATCH_SIZE]
                optimizer.zero_grad()
                batch_loss = 0
                
                for b_idx in batch_idx:
                    m5_list, h1_list, d1_list = [], [], []
                    lbl_scalp, lbl_intraday, lbl_swing = [], [], []
                    
                    for sym in valid_symbols:
                        m5_list.append(dataset[sym]["M5"]["features"][b_idx].to(device))
                        h1_list.append(dataset[sym]["H1"]["features"][b_idx].to(device))
                        d1_list.append(dataset[sym]["D1"]["features"][b_idx].to(device))
                        
                        lbl_scalp.append(dataset[sym]["M5"]["labels"][b_idx].to(device))
                        lbl_intraday.append(dataset[sym]["H1"]["labels"][b_idx].to(device))
                        lbl_swing.append(dataset[sym]["D1"]["labels"][b_idx].to(device))
                    
                    lbl_scalp = torch.stack(lbl_scalp)      # [na]
                    lbl_intraday = torch.stack(lbl_intraday) # [na]
                    lbl_swing = torch.stack(lbl_swing)       # [na]
                    
                    outputs = model(m5_list, h1_list, d1_list, edge_index)
                    
                    # Combined MTF loss: equal weighting across 3 strategy horizons
                    loss = (
                        criterion(outputs["scalp"], lbl_scalp) +
                        criterion(outputs["intraday"], lbl_intraday) +
                        criterion(outputs["swing"], lbl_swing)
                    ) / 3.0
                    
                    loss.backward()
                    batch_loss += loss.item()
                    
                    # Accuracy tracking
                    correct_scalp += (torch.argmax(outputs["scalp"], 1) == lbl_scalp).sum().item()
                    correct_intraday += (torch.argmax(outputs["intraday"], 1) == lbl_intraday).sum().item()
                    correct_swing += (torch.argmax(outputs["swing"], 1) == lbl_swing).sum().item()
                    total_nodes += na
                
                optimizer.step()
                total_loss += batch_loss
            
            avg_loss = total_loss / (num_samples / BATCH_SIZE + 1)
            
            print(
                f"[{epoch+1:03d}/{EPOCHS}] "
                f"{Fore.YELLOW}Loss: {avg_loss:.4f}{Style.RESET_ALL} | "
                f"Scalp: {Fore.GREEN}{100*correct_scalp/total_nodes:.1f}%{Style.RESET_ALL} | "
                f"Intraday: {Fore.CYAN}{100*correct_intraday/total_nodes:.1f}%{Style.RESET_ALL} | "
                f"Swing: {Fore.MAGENTA}{100*correct_swing/total_nodes:.1f}%{Style.RESET_ALL}"
            )
        
        # Save
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"\n{Fore.GREEN}✅ MTF GNN CHAMPION saved to {MODEL_PATH}{Style.RESET_ALL}")
        
    finally:
        await mt5.disconnect()


if __name__ == "__main__":
    asyncio.run(train_gnn())
