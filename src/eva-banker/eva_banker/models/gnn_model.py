import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    TORCH_GEO_AVAILABLE = True
except ImportError:
    TORCH_GEO_AVAILABLE = False
    GATConv = None
    global_mean_pool = None

import logging
logger = logging.getLogger(__name__)


class MultiAssetGNN(nn.Module):
    """
    Graph Neural Network pour modéliser les corrélations entre actifs (Hydra).
    Utilise Graph Attention Networks (GAT) pour pondérer dynamiquement les influences.
    """
    def __init__(self, in_channels, out_channels):
        super(MultiAssetGNN, self).__init__()
        if not TORCH_GEO_AVAILABLE:
            logger.warning("⚠️ torch_geometric non installé. GNN désactivé (mode stub).")
            self.stub = True
            self.fallback = nn.Linear(in_channels, out_channels)
            return
        self.stub = False
        self.conv1 = GATConv(in_channels, 32, heads=4)
        self.conv2 = GATConv(128, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index, batch):
        if self.stub:
            return self.fallback(x.mean(dim=0, keepdim=True))
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return global_mean_pool(x, batch)

class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer (TFT) simplifié.
    Capture les dynamiques temporelles et les importances relatives des features.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        self.gate = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)
        # Auto-attention sur la séquence temporelle
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        return attn_out[:, -1, :] # Dernière étape temporelle

class TFTGNNModel(nn.Module):
    """
    Modèle Hybride TFT-GNN.
    Combine l'analyse temporelle (TFT) et structurelle du marché (GNN).
    """
    def __init__(self, asset_dim, temporal_dim, hidden_dim):
        super().__init__()
        self.tft = TemporalFusionTransformer(asset_dim, temporal_dim)
        self.gnn = MultiAssetGNN(temporal_dim, hidden_dim)

    def forward(self, ts_data_list, edge_index):
        # ts_data_list: Liste de tenseurs [seq_len, asset_dim] pour chaque actif
        # 1. Traitement temporel pour chaque actif
        temporal_embeddings = torch.stack([self.tft(d.unsqueeze(0)) for d in ts_data_list]).squeeze(1)
        
        # 2. Traitement structurel (GNN)
        batch = torch.zeros(temporal_embeddings.size(0), dtype=torch.long)
        market_state = self.gnn(temporal_embeddings, edge_index, batch)
        
        return market_state
