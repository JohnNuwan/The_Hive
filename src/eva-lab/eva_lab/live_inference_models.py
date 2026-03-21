"""Modeles partages pour l'inference live CPU MuZero."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LivePredictRequest(BaseModel):
    """Charge utile minimale pour une inference live CPU.

    Args:
        symbol (str): Symbole live cible.
        horizon (str): Horizon demande. En v1, seul ``scalp`` est accepte.
        selection_policy (str): Politique de selection voulue. En v1,
            ``champion_only`` est obligatoire.
        price (float): Prix courant du symbole.
        timestamp (str | None): Horodatage ISO de l'observation.
        latest_candle (dict[str, Any]): Derniere bougie utile au modele.
        indicators (dict[str, Any]): Indicateurs techniques deja calcules.
        training_compat_mode (str | None): Mode runtime du banker, a titre informatif.
        cortex_required (bool | None): Indique si le banker attend encore le Cortex.
        gnn_mode (str | None): Mode d'usage du GNN cote banker.
    """

    symbol: str = Field(..., min_length=1)
    horizon: str = Field(default="scalp")
    selection_policy: str = Field(default="champion_only")
    price: float = Field(default=0.0)
    timestamp: str | None = Field(default=None)
    latest_candle: dict[str, Any] = Field(default_factory=dict)
    indicators: dict[str, Any] = Field(default_factory=dict)
    training_compat_mode: str | None = Field(default=None)
    cortex_required: bool | None = Field(default=None)
    gnn_mode: str | None = Field(default=None)
