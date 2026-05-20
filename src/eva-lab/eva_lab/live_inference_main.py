"""API dediee a l'inference live CPU MuZero et Ensemble."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx

# Ce service doit rester epingle au CPU, meme si le serveur heberge aussi des
# runs JAX sur GPU. L'objectif est de decoupler le live du training lourd.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from eva_lab.dreamer_gate import DreamerGate
from eva_lab.live_inference_models import LivePredictRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    gate_status = gate.get_status()
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
async def live_universe_proxy(
    horizon: str = "intraday",
    engine: str = "muzero",
) -> dict[str, Any]:
    """Expose l'univers live via le service CPU externe.

    Args:
        horizon (str): Horizon demande.
        engine (str): Moteur demande.

    Returns:
        dict[str, Any]: Reponse JSON du Lab interne.
    """
    return await _proxy_lab_request(
        method="GET",
        path="/live/universe",
        params={"horizon": horizon, "engine": engine},
    )


@app.post("/gnn/predict")
async def gnn_predict_proxy(payload: dict[str, Any]) -> dict[str, Any]:
    """Expose la prediction GNN via le service CPU externe.

    Args:
        payload (dict[str, Any]): Charge identique a celle attendue par le Lab.

    Returns:
        dict[str, Any]: Reponse JSON du Lab interne.
    """
    return await _proxy_lab_request(
        method="POST",
        path="/gnn/predict",
        json_payload=payload,
    )
