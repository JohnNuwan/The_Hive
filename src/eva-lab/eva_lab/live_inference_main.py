"""API dediee a l'inference live CPU MuZero."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ce service doit rester epingle au CPU, meme si le serveur heberge aussi des
# runs JAX sur GPU. L'objectif est de decoupler le live du training lourd.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from eva_lab.dreamer_gate import DreamerGate
from eva_lab.live_inference_models import LivePredictRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    description="Inference MuZero scalp epinglee CPU pour le live.",
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
