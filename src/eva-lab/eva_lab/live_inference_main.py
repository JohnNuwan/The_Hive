"""API dediee a l'inference live CPU MuZero et Ensemble."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx
from pydantic import BaseModel

# Ce service doit rester epingle au CPU, meme si le serveur heberge aussi des
# runs JAX sur GPU. L'objectif est de decoupler le live du training lourd.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from eva_lab.dreamer_gate import DreamerGate
from eva_lab.live_inference_models import LivePredictRequest
from eva_lab.training_status import (
    build_effective_training_universe_summary,
    load_nightly_summary,
    load_training_status,
    tail_training_log,
)
from eva_lab.training_utils import get_gnn_model_kwargs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GNNPredictRequest(BaseModel):
    """Decrit une charge de prediction GNN multi-horizon."""

    assets_data: dict[str, list[list[float]]]


def _resolve_focus_symbols() -> list[str]:
    """Resout l'univers live voulu depuis l'environnement.

    Returns:
        list[str]: Liste ordonnee de symboles cibles.
    """
    env_names = (
        "TRAINING_FOCUS_SYMBOLS",
        "TRAIN_GNN_SYMBOLS",
        "MUZERO_SYMBOLS_SCALP",
        "MUZERO_SYMBOLS",
        "BANKER_CPU_LIVE_SYMBOLS",
    )
    for env_name in env_names:
        raw_value = str(os.getenv(env_name, "")).strip()
        if not raw_value:
            continue
        symbols: list[str] = []
        for item in raw_value.split(","):
            normalized = str(item).strip()
            if normalized and normalized not in symbols:
                symbols.append(normalized)
        if symbols:
            return symbols
    fallback = "XAUUSD,US30.cash,GER40.cash,EURUSD,US100.cash,US500.cash,BTCUSD"
    return [symbol.strip() for symbol in fallback.split(",") if symbol.strip()]


def _resolve_lab_proxy_base_url() -> str:
    """Construit l'URL interne du service Lab a proxifier.

    Returns:
        str: URL HTTP de base vers le service Lab interne.
    """
    host = str(os.getenv("LIVE_INFERENCE_LAB_PROXY_HOST", "lab")).strip() or "lab"
    port = str(os.getenv("LIVE_INFERENCE_LAB_PROXY_PORT", "8600")).strip() or "8600"
    return f"http://{host}:{port}"


async def _proxy_lab_request(
    *,
    method: str,
    path: str,
    params: dict[str, Any] | None = None,
    json_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Transmet une requete read-only vers le service Lab interne.

    Args:
        method (str): Methode HTTP a utiliser.
        path (str): Chemin cible sur le service Lab.
        params (dict[str, Any] | None): Parametres de requete optionnels.
        json_payload (dict[str, Any] | None): Charge JSON optionnelle.

    Returns:
        dict[str, Any]: Reponse JSON du service Lab.
    """
    base_url = _resolve_lab_proxy_base_url().rstrip("/")
    url = f"{base_url}{path}"
    timeout_seconds = max(2.0, float(os.getenv("LIVE_INFERENCE_PROXY_TIMEOUT_SECONDS", "6")))
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.request(
            method=method.upper(),
            url=url,
            params=params,
            json=json_payload,
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {"payload": payload}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise le service d'inference live CPU.

    Args:
        app (FastAPI): Instance FastAPI a initialiser.

    Yields:
        None: Rend le controle apres l'initialisation.
    """
    app.state.dreamer_gate = DreamerGate(enable_training=False)
    app.state.gnn_model = None
    try:
        from eva_lab.models.gnn_model import TFTGNNModel
        import torch

        app.state.gnn_model = TFTGNNModel(**get_gnn_model_kwargs())
        model_path = "data/models/gnn_master.pth"
        if os.path.exists(model_path):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            app.state.gnn_model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info("GNN live CPU: poids charges depuis %s.", model_path)
        else:
            logger.warning("GNN live CPU: poids absents, mode neutre conserve.")
        app.state.gnn_model.eval()
    except Exception as exc:
        app.state.gnn_model = None
        logger.warning("GNN live CPU indisponible: %s", exc)
    logger.info("Service d'inference live CPU demarre.")
    yield
    logger.info("Service d'inference live CPU arrete.")


app = FastAPI(
    title="EVA Live Inference API",
    description="Inference scalp CPU pour MuZero seul ou ensemble MuZero/Dreamer.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Retourne l'etat de sante minimal du service.

    Returns:
        dict[str, str]: Etat nominal du processus.
    """
    return {"status": "online", "service": "live_inference_cpu"}


@app.get("/status")
async def status() -> dict[str, object]:
    """Retourne les capacites du service CPU live.

    Returns:
        dict[str, object]: Etat du gate et contraintes d'usage.
    """
    gate: DreamerGate = app.state.dreamer_gate
    gate_status = await asyncio.to_thread(gate.get_status)
    return {
        "status": "ok",
        "service": "live_inference_cpu",
        "device": "cpu",
        "supported_horizons": ["scalp"],
        "selection_policy_required": "champion_only",
        "gate": gate_status,
    }


@app.post("/predict/live")
async def predict_live(request: LivePredictRequest) -> dict[str, object]:
    """Execute une inference live CPU stricte.

    Args:
        request (LivePredictRequest): Observation live du banker.

    Returns:
        dict[str, object]: Action brute, confiance et metadonnees du champion.
    """
    gate: DreamerGate = app.state.dreamer_gate
    return gate.run_live_inference(request.model_dump())


@app.post("/predict/ensemble")
async def predict_ensemble(request: LivePredictRequest) -> dict[str, object]:
    """Execute une inference d'ensemble CPU entre MuZero et Dreamer.

    Args:
        request (LivePredictRequest): Observation live du banker.

    Returns:
        dict[str, object]: Decision arbitree entre MuZero et Dreamer.
    """
    gate: DreamerGate = app.state.dreamer_gate
    return gate.run_ensemble_inference(request.model_dump())


@app.get("/live/universe")
async def live_universe(
    horizon: str = "intraday",
    engine: str = "muzero",
) -> dict[str, Any]:
    """Retourne l'univers live actif sans dependre du reseau inter-conteneurs.

    Args:
        horizon (str): Horizon demande.
        engine (str): Moteur demande.

    Returns:
        dict[str, Any]: Univers live minimal et coherent pour le banker.
    """
    run_status = load_training_status()
    focus_symbols = list(run_status.get("focus_symbols") or _resolve_focus_symbols())
    universe_summary = build_effective_training_universe_summary(
        focus_symbols,
        base_universe=dict(run_status.get("universe") or {}),
    )
    return {
        "status": "ok",
        "engine": str(engine or "muzero").lower(),
        "horizon": str(horizon or "intraday").lower(),
        "family": run_status.get("family"),
        "feature_profile": run_status.get("feature_profile"),
        "dataset_id": run_status.get("dataset_id"),
        "selection_policy": os.getenv("MUZERO_LIVE_SELECTION_POLICY", "champion_only"),
        "selection": "champion",
        "live_champion_id": os.getenv("BANKER_FORCE_LIVE_CHAMPION_ID_MUZERO", ""),
        "live_champion_id_muzero": os.getenv("BANKER_FORCE_LIVE_CHAMPION_ID_MUZERO", ""),
        "live_champion_id_dreamer": None,
        "active_live_engine": "muzero",
        "live_universe": {
            "symbols": focus_symbols,
            "count": len(focus_symbols),
            "source": "live_inference_cpu_env",
            "summary": universe_summary,
        },
    }


@app.post("/gnn/predict")
async def gnn_predict(request: GNNPredictRequest) -> dict[str, Any]:
    """Predis les biais GNN localement dans le service CPU live.

    Args:
        request (GNNPredictRequest): Charge multi-horizon du banker.

    Returns:
        dict[str, Any]: Biais scalp, intraday et swing.
    """
    if not hasattr(app.state, "gnn_model") or app.state.gnn_model is None:
        return {
            "scalp": {"bias": "NEUTRAL", "confidence": 0.0},
            "intraday": {"bias": "NEUTRAL", "confidence": 0.0},
            "swing": {"bias": "NEUTRAL", "confidence": 0.0},
            "reason": "GNN indisponible",
        }

    try:
        import torch
        import torch.nn.functional as F

        gnn = app.state.gnn_model
        classes = ["BULLISH", "BEARISH", "RANGING"]

        def _prep_tensor(raw: list[list[float]] | list[float], seq_len: int = 15, feat_dim: int = 20):
            """Normalise une serie en tenseur [seq_len, feat_dim]."""
            tensor = torch.tensor(raw, dtype=torch.float32)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            if tensor.size(1) < feat_dim:
                tensor = F.pad(tensor, (0, feat_dim - tensor.size(1)))
            if tensor.size(0) < seq_len:
                pad_len = seq_len - tensor.size(0)
                tensor = torch.cat([tensor, tensor[-1:].repeat(pad_len, 1)], dim=0)
            return tensor[-seq_len:]

        ts_m5: list[Any] = []
        ts_h1: list[Any] = []
        ts_d1: list[Any] = []

        for asset, raw in request.assets_data.items():
            tensor = _prep_tensor(raw)
            if "_M5" in asset or "_5" in asset:
                ts_m5.append(tensor)
            elif "_H1" in asset or "_60" in asset:
                ts_h1.append(tensor)
            elif "_D1" in asset or "_1440" in asset:
                ts_d1.append(tensor)
            else:
                ts_m5.append(tensor)
                ts_h1.append(tensor)
                ts_d1.append(tensor)

        if ts_m5 and not ts_h1:
            ts_h1 = ts_m5[:]
        if ts_m5 and not ts_d1:
            ts_d1 = ts_m5[:]
        if not ts_m5:
            return {
                "scalp": {"bias": "NEUTRAL", "confidence": 0.0},
                "intraday": {"bias": "NEUTRAL", "confidence": 0.0},
                "swing": {"bias": "NEUTRAL", "confidence": 0.0},
                "reason": "Aucune serie exploitable",
            }

        while len(ts_h1) < len(ts_m5):
            ts_h1.append(ts_h1[-1] if ts_h1 else ts_m5[len(ts_h1)])
        while len(ts_d1) < len(ts_m5):
            ts_d1.append(ts_d1[-1] if ts_d1 else ts_m5[len(ts_d1)])

        asset_count = len(ts_m5)
        rows: list[int] = []
        cols: list[int] = []
        for source_index in range(asset_count):
            for target_index in range(asset_count):
                if source_index != target_index:
                    rows.append(source_index)
                    cols.append(target_index)
        edge_index = (
            torch.tensor([rows, cols], dtype=torch.long)
            if asset_count > 1
            else torch.empty((2, 0), dtype=torch.long)
        )

        with torch.no_grad():
            outputs = gnn(ts_m5, ts_h1, ts_d1, edge_index)

        def _decode(logits):
            probs = torch.softmax(logits[0], dim=0)
            index = int(torch.argmax(probs).item())
            return {"bias": classes[index], "confidence": round(float(probs[index].item()), 3)}

        return {
            "scalp": _decode(outputs["scalp"]),
            "intraday": _decode(outputs["intraday"]),
            "swing": _decode(outputs["swing"]),
            "reason": "GNN live_inference_local",
        }
    except Exception as exc:
        logger.warning("Prediction GNN live impossible: %s", exc)
        return {
            "scalp": {"bias": "NEUTRAL", "confidence": 0.0},
            "intraday": {"bias": "NEUTRAL", "confidence": 0.0},
            "swing": {"bias": "NEUTRAL", "confidence": 0.0},
            "reason": f"Erreur GNN: {exc}",
        }


@app.post("/shadow/record")
async def shadow_record_proxy(payload: dict[str, Any]) -> dict[str, Any]:
    """Expose l'enregistrement Shadow via le service CPU externe.

    Args:
        payload (dict[str, Any]): Charge d'ouverture envoyee par le banker.

    Returns:
        dict[str, Any]: Reponse JSON du Lab interne.
    """
    return await _proxy_lab_request(
        method="POST",
        path="/shadow/record",
        json_payload=payload,
    )


@app.post("/shadow/feedback")
async def shadow_feedback_proxy(payload: dict[str, Any]) -> dict[str, Any]:
    """Expose le feedback de cloture Shadow via le service CPU externe.

    Args:
        payload (dict[str, Any]): Charge de feedback envoyee par le banker.

    Returns:
        dict[str, Any]: Reponse JSON du Lab interne.
    """
    return await _proxy_lab_request(
        method="POST",
        path="/shadow/feedback",
        json_payload=payload,
    )


@app.get("/training/status")
async def training_status_proxy(limit: int = 50) -> dict[str, Any]:
    """Expose un statut training local et rapide.

    Args:
        limit (int): Nombre maximal de lignes de log a remonter.

    Returns:
        dict[str, Any]: Statut JSON sans dependre du reseau vers le Lab.
    """
    run_status = load_training_status()
    focus_symbols = list(run_status.get("focus_symbols") or _resolve_focus_symbols())
    universe_summary = build_effective_training_universe_summary(
        focus_symbols,
        base_universe=dict(run_status.get("universe") or {}),
    )
    return {
        "status": "ok",
        "run": {
            **run_status,
            "focus_symbols": focus_symbols,
            "universe": universe_summary,
        },
        "universe": universe_summary,
        "logs": tail_training_log(limit=limit),
        "nightly_summary": load_nightly_summary(),
        "source": "live_inference_cpu_local",
    }
