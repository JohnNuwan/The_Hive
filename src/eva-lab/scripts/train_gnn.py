import asyncio
import logging
import os
import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime, timedelta
from pathlib import Path
from colorama import init, Fore, Style
import numpy as np

# Initialiser colorama (pour Windows)
init()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Hive-GNN-Trainer")

# Add src directories to sys.path so we can import modules
import sys
src_path = str(Path(__file__).parent.parent.parent.absolute())
if src_path not in sys.path:
    sys.path.append(src_path)

from eva_lab.models.gnn_model import TFTGNNModel
from eva_banker.services.mt5 import MT5Service
from shared.indicators import IndicatorFactory

# ═══════════════════════════════════════════════════════════════════════════════
# PARAMS
# ═══════════════════════════════════════════════════════════════════════════════

SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "US30.cash", "US100.cash", "GER40.cash",
    "BTCUSD", "ETHUSD", "XAUUSD", "AUDUSD", "NZDUSD", "USDCHF", "USDCAD",
    "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY",
    "EURAUD", "EURCAD", "EURCHF", "GBPAUD", "GBPCAD", "GBPCHF",
    "XAGUSD", "SOLUSD", "BNBUSD"
]

TIMEFRAME = 15 # M15
HISTORY_CANDLES = 1000 # ~10 days of M15 data
SEQ_LEN = 15 # Temporal context length for the TFT
FUTURE_CANDLES = 4 # Predict 1 hour ahead
BATCH_SIZE = 32
EPOCHS = 50

ASSET_DIM = 20 # Number of features per candle
TEMPORAL_DIM = 32
HIDDEN_DIM = 64
NUM_CLASSES = 3

MODEL_DIR = Path("data/models")
MODEL_PATH = MODEL_DIR / "gnn_master.pth"

# Label encoding: BULLISH=0, BEARISH=1, RANGING=2
def get_label(current_price, future_price, atr):
    change = future_price - current_price
    # Threshold based on 0.5 * ATR (significant movement)
    threshold = (atr if atr > 0 else (current_price * 0.001)) * 0.5
    
    if change > threshold:
        return 0 # BULLISH
    elif change < -threshold:
        return 1 # BEARISH
    else:
        return 2 # RANGING

async def fetch_and_preprocess_data(mt5: MT5Service):
    logger.info(f"📥 Fetching {HISTORY_CANDLES} candles for {len(SYMBOLS)} symbols from MT5...")
    
    dataset = {}
    valid_symbols = []
    
    for symbol in SYMBOLS:
        candles = await mt5.get_recent_candles(symbol, timeframe=TIMEFRAME, count=HISTORY_CANDLES)
        if not candles or len(candles) < HISTORY_CANDLES * 0.9:
            logger.warning(f"⚠️ Not enough data for {symbol}, skipping.")
            continue
            
        logger.info(f"✅ Fetched {len(candles)} candles for {symbol}")
        
        # Calculate Indicators
        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        volumes = [c["tick_volume"] for c in candles]
        
        rsi = IndicatorFactory.rsi(closes, 14)
        adx_data = IndicatorFactory.adx(highs, lows, closes, 14)
        adx = adx_data["adx"]
        vwap = IndicatorFactory.vwap(highs, lows, closes, volumes)
        macd_data = IndicatorFactory.macd(closes)
        macd_hist = macd_data["histogram"]
        atr = IndicatorFactory.atr(highs, lows, closes, 14)
        bb_data = IndicatorFactory.bollinger_bands(closes)
        bb_pct = bb_data["pct_b"]
        
        # Compile features for each time step
        features = []
        labels = []
        
        # We need to compute features and find the label N steps ahead
        # Start at max window size (e.g. 50 for ema, etc. though we use small windows here) + SEQ_LEN
        start_idx = 50 
        
        for i in range(start_idx, len(candles) - FUTURE_CANDLES):
            # Extract sequence
            seq_features = []
            for j in range(i - SEQ_LEN + 1, i + 1):
                f = [
                    # Normalized price (simple scale)
                    closes[j] / closes[j-1] - 1.0 if j > 0 else 0,
                    # Indicators
                    rsi.iloc[j] / 100.0 if not np.isnan(rsi.iloc[j]) else 0.5,
                    adx.iloc[j] / 100.0 if not np.isnan(adx.iloc[j]) else 0.2,
                    macd_hist.iloc[j] / closes[j] if not np.isnan(macd_hist.iloc[j]) else 0,
                    bb_pct.iloc[j] if not np.isnan(bb_pct.iloc[j]) else 0.5,
                    volumes[j] / (sum(volumes[j-10:j])/10 + 1e-5), # Relative volume
                    # Distance to VWAP
                    (closes[j] - vwap.iloc[j]) / closes[j] if not np.isnan(vwap.iloc[j]) else 0,
                    # Volatility
                    atr.iloc[j] / closes[j] if not np.isnan(atr.iloc[j]) else 0,
                    # Candle shape
                    (closes[j] - lows[j]) / (highs[j] - lows[j] + 1e-8), # Close pos
                    (highs[j] - max(closes[j], candles[j]["open"])) / closes[j], # Upper wick
                    (min(closes[j], candles[j]["open"]) - lows[j]) / closes[j], # Lower wick
                    # Extra padding to reach ASSET_DIM = 20
                    0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
                seq_features.append(f)
                
            features.append(seq_features)
            
            # Label
            current_price = closes[i]
            future_price = closes[i + FUTURE_CANDLES]
            current_atr = atr.iloc[i] if not np.isnan(atr.iloc[i]) else 0.001 * current_price
            
            lbl = get_label(current_price, future_price, current_atr)
            labels.append(lbl)
            
        dataset[symbol] = {
            "features": torch.tensor(features, dtype=torch.float32), # [N, SEQ_LEN, ASSET_DIM]
            "labels": torch.tensor(labels, dtype=torch.long) # [N]
        }
        valid_symbols.append(symbol)
        
    return dataset, valid_symbols

def build_graph(num_nodes):
    """Build a fully connected graph for the nodes (assets)"""
    rows, cols = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)
    return torch.tensor([rows, cols], dtype=torch.long)

async def train_gnn():
    print(f"\n{Fore.MAGENTA}======================================================{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}🐝 THE HIVE: MULTI-ASSET GNN EVOLUTION (HYDRA) 🐝{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}======================================================{Style.RESET_ALL}\n")
    
    # 1. Connect to MT5
    mt5 = MT5Service(mock=False, login=1512664750, server="FTMO-Demo")
    await mt5.connect()
    
    try:
        # 2. Fetch Data
        dataset, valid_symbols = await fetch_and_preprocess_data(mt5)
        if not dataset:
            logger.error("No valid dataset generated. Aborting.")
            return

        num_assets = len(valid_symbols)
        num_samples = min([data["features"].size(0) for data in dataset.values()])
        
        logger.info(f"🌐 Graph Topology: {num_assets} Nodes (Assets), Fully Connected.")
        logger.info(f"📊 Training Samples: {num_samples} time steps per asset.")

        # 3. Model Setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"⚙️ Using device: {device}")
        
        model = TFTGNNModel(asset_dim=ASSET_DIM, temporal_dim=TEMPORAL_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES)
        
        # Load weights if exist
        if MODEL_PATH.exists():
            try:
                model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                logger.info(f"✅ Loaded existing master weights from {MODEL_PATH}")
            except Exception as e:
                logger.warning(f"Could not load weights: {e}. Starting fresh.")
                
        model = model.to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        edge_index = build_graph(num_assets).to(device)
        
        # 4. Training Loop
        logger.info("🚀 Engaging Neural Training Sequence...")
        
        model.train()
        
        for epoch in range(EPOCHS):
            total_loss = 0
            correct = 0
            total = 0
            
            # Shuffle indices
            indices = torch.randperm(num_samples)
            
            for i in range(0, num_samples, BATCH_SIZE):
                batch_indices = indices[i:i+BATCH_SIZE]
                current_batch_size = len(batch_indices)
                
                optimizer.zero_grad()
                
                batch_loss = 0
                
                # In a real GNN timeseries setup, we process batch by batch
                # But our GNN takes one graph at a time here [num_assets, temporal_dim]
                # We will average the loss over the mini-batch conceptually by looping, 
                # or we could batch the graphs. Since the graph is tiny (27 nodes), looping is viable.
                
                # Accumulate gradients over the batch
                for b_idx in batch_indices:
                    # Construct node features for this time step
                    node_features = []
                    node_labels = []
                    for sym in valid_symbols:
                        nf = dataset[sym]["features"][b_idx].to(device) # [SEQ_LEN, ASSET_DIM]
                        nl = dataset[sym]["labels"][b_idx].to(device) # []
                        node_features.append(nf)
                        node_labels.append(nl)
                        
                    labels = torch.stack(node_labels) # [num_assets]
                    
                    # Forward pass
                    # node_features is List of [SEQ_LEN, ASSET_DIM] lengths = num_assets
                    logits = model(node_features, edge_index) # [num_assets, num_classes]
                    
                    loss = criterion(logits, labels)
                    loss.backward()
                    
                    batch_loss += loss.item()
                    
                    # Metrics
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == labels).sum().item()
                    total += num_assets
                
                optimizer.step()
                total_loss += batch_loss
                
            avg_loss = total_loss / num_samples
            accuracy = 100 * correct / total
            
            print(f"[{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
            
        # 5. Save Model
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        logger.info(f"✅ GNN Master saved to {MODEL_PATH}")

    finally:
        await mt5.disconnect()

if __name__ == "__main__":
    asyncio.run(train_gnn())
