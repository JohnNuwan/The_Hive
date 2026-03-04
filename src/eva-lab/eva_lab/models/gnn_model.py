import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv, global_mean_pool
    TORCH_GEO_AVAILABLE = True
except ImportError:
    TORCH_GEO_AVAILABLE = False
    GATConv = None
    global_mean_pool = None

import logging
logger = logging.getLogger(__name__)


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer (TFT).
    Capture les dynamiques temporelles d'un seul horizon.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.gate = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        # Gate: controls how much LSTM vs Attention to trust
        gated = torch.sigmoid(self.gate(lstm_out)) * attn_out
        return gated[:, -1, :]  # Return last timestep as embedding


class CrossTimeframeFuser(nn.Module):
    """
    Fuse 3 Horizons (M5 / H1 / D1) via Cross-Attention.
    L'horizon D1 (tendance) guide la lecture de M5 (execution).
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim * 3, embed_dim)

    def forward(self, emb_m5, emb_h1, emb_d1):
        # Simple fusion by concatenation then projection
        # [batch, 3, embed_dim]
        stack = torch.stack([emb_m5, emb_h1, emb_d1], dim=1)
        # Cross attention: each timeframe attends to all others
        fused, _ = self.cross_attn(stack, stack, stack)
        fused = self.norm(fused + stack)  # Residual
        # Concatenate all horizon outputs and project to embed_dim
        flat = fused.reshape(fused.size(0), -1)  # [batch, embed_dim * 3]
        return self.proj(flat)  # [batch, embed_dim]


class MultiAssetGNN(nn.Module):
    """
    GNN with Graph Attention, operating on the fused MTF embeddings.
    Chaque noeud = 1 actif. Les messages passent entre actifs pour capturer les correlations.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        if not TORCH_GEO_AVAILABLE:
            logger.warning("⚠️ torch_geometric non installé. GNN en mode linéaire.")
            self.stub = True
            self.fallback = nn.Linear(in_channels, out_channels)
            return
        self.stub = False
        self.conv1 = GATConv(in_channels, 32, heads=4)
        self.conv2 = GATConv(128, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        if self.stub:
            return self.fallback(x)
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        return x  # [num_nodes, out_channels] — node-level embeddings


class TFTGNNModel(nn.Module):
    """
    === OMNI-ARCHITECTURE: Multi-Timeframe GNN ===
    
    Combine 3 horizons temporels par actif (M5, H1, D1) pour prédire
    3 stratégies distinctes: Scalp, Intraday, Swing.
    
    Input:
        ts_data_m5:  Liste de Tenseurs [SEQ_LEN, ASSET_DIM] pour chaque actif (court terme)
        ts_data_h1:  Liste de Tenseurs [SEQ_LEN, ASSET_DIM] pour chaque actif (moyen terme)
        ts_data_d1:  Liste de Tenseurs [SEQ_LEN, ASSET_DIM] pour chaque actif (long terme)
        edge_index:  Graphe de corrélation entre les actifs [2, num_edges]
    
    Output:
        dict:
            'scalp':    Tensor [num_assets, 3] — Logits BULLISH/BEARISH/RANGING sur M5 (+1H)
            'intraday': Tensor [num_assets, 3] — Logits BULLISH/BEARISH/RANGING sur H1 (+1D)
            'swing':    Tensor [num_assets, 3] — Logits BULLISH/BEARISH/RANGING sur D1 (+1W)
    """
    def __init__(self, asset_dim: int, temporal_dim: int, hidden_dim: int, num_classes: int = 3):
        super().__init__()
        self.asset_dim = asset_dim
        self.temporal_dim = temporal_dim
        self.hidden_dim = hidden_dim
        
        # 3 timeframe-specific TFT encoders
        self.tft_m5 = TemporalFusionTransformer(asset_dim, temporal_dim)
        self.tft_h1 = TemporalFusionTransformer(asset_dim, temporal_dim)
        self.tft_d1 = TemporalFusionTransformer(asset_dim, temporal_dim)
        
        # Cross-Timeframe Fusion
        self.fuser = CrossTimeframeFuser(embed_dim=temporal_dim)
        
        # Spatial Graph Correlation
        self.gnn = MultiAssetGNN(temporal_dim, hidden_dim)
        
        # 3 Strategy Classification Heads
        def _make_head(in_dim, num_classes):
            return nn.Sequential(
                nn.Linear(in_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, num_classes)
            )
        
        # The scalp head sees more M5 signal, fused with GNN context
        self.scalp_head = _make_head(hidden_dim + temporal_dim, num_classes)
        # The intraday head sees equal weight from GNN + H1 signal
        self.intraday_head = _make_head(hidden_dim + temporal_dim, num_classes)
        # The swing head sees more D1 signal + GNN for correlation context
        self.swing_head = _make_head(hidden_dim + temporal_dim, num_classes)

    @staticmethod
    def _stack_list(ts_list):
        """Stack list of [seq_len, asset_dim] into [num_assets, seq_len, asset_dim]"""
        return torch.stack([t.unsqueeze(0) if t.dim() == 2 else t for t in ts_list])

    def forward(self, ts_data_m5: list, ts_data_h1: list, ts_data_d1: list, edge_index: torch.Tensor) -> dict:
        # 1. Temporal encoding per horizon for each asset
        # Each stack: [num_assets, seq_len, asset_dim]
        x_m5 = self._stack_list(ts_data_m5)  # Scalp context
        x_h1 = self._stack_list(ts_data_h1)  # Intraday context
        x_d1 = self._stack_list(ts_data_d1)  # Swing context
        
        # TFT outputs: [num_assets, temporal_dim]
        emb_m5 = self.tft_m5(x_m5)
        emb_h1 = self.tft_h1(x_h1)
        emb_d1 = self.tft_d1(x_d1)
        
        # 2. Cross-Timeframe fusion: [num_assets, temporal_dim]
        fused_emb = self.fuser(emb_m5, emb_h1, emb_d1)
        
        # 3. Spatial Graph Correlation (GNN message passing between assets)
        if edge_index.numel() > 0:
            gnn_emb = self.gnn(fused_emb, edge_index)  # [num_assets, hidden_dim]
        else:
            # Single asset — no message passing possible
            if self.gnn.stub:
                gnn_emb = self.gnn.fallback(fused_emb)
            else:
                gnn_emb = F.elu(self.gnn.conv2(F.elu(self.gnn.conv1(fused_emb, edge_index)), edge_index))
        
        # 4. Strategy-Specific Classification
        # Each head sees the GNN macro context + its own timeframe's micro signal
        scalp_features = torch.cat([gnn_emb, emb_m5], dim=-1)
        intraday_features = torch.cat([gnn_emb, emb_h1], dim=-1)
        swing_features = torch.cat([gnn_emb, emb_d1], dim=-1)
        
        return {
            "scalp": self.scalp_head(scalp_features),      # [num_assets, 3]
            "intraday": self.intraday_head(intraday_features),  # [num_assets, 3]
            "swing": self.swing_head(swing_features),      # [num_assets, 3]
        }


