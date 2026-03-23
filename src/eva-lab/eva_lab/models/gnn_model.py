import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv
    TORCH_GEO_AVAILABLE = True
except ImportError:
    TORCH_GEO_AVAILABLE = False
    GATConv = None

logger = logging.getLogger(__name__)


class TemporalFusionTransformer(nn.Module):
    """Encode la dynamique temporelle d'un horizon unique."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.gate = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projette une sequence temporelle vers un embedding latent."""
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        gated = torch.sigmoid(self.gate(lstm_out)) * attn_out
        return gated[:, -1, :]


class CrossTimeframeFuser(nn.Module):
    """Fusionne M5, H1 et D1 dans un embedding unique."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim * 3, embed_dim)

    def forward(
        self,
        emb_m5: torch.Tensor,
        emb_h1: torch.Tensor,
        emb_d1: torch.Tensor,
    ) -> torch.Tensor:
        """Applique une fusion croisee des horizons temporels."""
        stacked = torch.stack([emb_m5, emb_h1, emb_d1], dim=1)
        fused, _ = self.cross_attn(stacked, stacked, stacked)
        fused = self.norm(fused + stacked)
        flat = fused.reshape(fused.size(0), -1)
        return self.proj(flat)


class MultiAssetGNN(nn.Module):
    """Capture les correlations inter-actifs via un graphe d'attention."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        if not TORCH_GEO_AVAILABLE:
            logger.warning("torch_geometric non installe. GNN en mode lineaire.")
            self.stub = True
            self.fallback = nn.Linear(in_channels, out_channels)
            return

        self.stub = False
        self.conv1 = GATConv(in_channels, 32, heads=4)
        self.conv2 = GATConv(128, out_channels, heads=1, concat=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Diffuse le contexte de marche entre noeuds du graphe."""
        if self.stub:
            return self.fallback(x)
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        return x


class TFTGNNModel(nn.Module):
    """Modele multi-timeframe et multi-strategy pour le trading."""

    def __init__(self, asset_dim: int, temporal_dim: int, hidden_dim: int, num_classes: int = 3):
        super().__init__()
        self.asset_dim = asset_dim
        self.temporal_dim = temporal_dim
        self.hidden_dim = hidden_dim

        self.tft_m5 = TemporalFusionTransformer(asset_dim, temporal_dim)
        self.tft_h1 = TemporalFusionTransformer(asset_dim, temporal_dim)
        self.tft_d1 = TemporalFusionTransformer(asset_dim, temporal_dim)
        self.fuser = CrossTimeframeFuser(embed_dim=temporal_dim)
        self.gnn = MultiAssetGNN(temporal_dim, hidden_dim)

        def make_head(in_dim: int, classes: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, classes),
            )

        self.scalp_head = make_head(hidden_dim + temporal_dim, num_classes)
        self.intraday_head = make_head(hidden_dim + temporal_dim, num_classes)
        self.swing_head = make_head(hidden_dim + temporal_dim, num_classes)

    @staticmethod
    def _stack_list(ts_list: list[torch.Tensor]) -> torch.Tensor:
        """Empile les sequences par actif sans ajouter de dimension parasite."""
        normalized = [tensor.squeeze(0) if tensor.dim() == 3 else tensor for tensor in ts_list]
        return torch.stack(normalized)

    def forward(
        self,
        ts_data_m5: list[torch.Tensor],
        ts_data_h1: list[torch.Tensor],
        ts_data_d1: list[torch.Tensor],
        edge_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Retourne trois tetes de prediction: scalp, intraday et swing."""
        x_m5 = self._stack_list(ts_data_m5)
        x_h1 = self._stack_list(ts_data_h1)
        x_d1 = self._stack_list(ts_data_d1)

        emb_m5 = self.tft_m5(x_m5)
        emb_h1 = self.tft_h1(x_h1)
        emb_d1 = self.tft_d1(x_d1)
        fused_emb = self.fuser(emb_m5, emb_h1, emb_d1)

        if edge_index.numel() > 0:
            gnn_emb = self.gnn(fused_emb, edge_index)
        elif self.gnn.stub:
            gnn_emb = self.gnn.fallback(fused_emb)
        else:
            gnn_emb = F.elu(self.gnn.conv2(F.elu(self.gnn.conv1(fused_emb, edge_index)), edge_index))

        scalp_features = torch.cat([gnn_emb, emb_m5], dim=-1)
        intraday_features = torch.cat([gnn_emb, emb_h1], dim=-1)
        swing_features = torch.cat([gnn_emb, emb_d1], dim=-1)

        return {
            "scalp": self.scalp_head(scalp_features),
            "intraday": self.intraday_head(intraday_features),
            "swing": self.swing_head(swing_features),
        }
