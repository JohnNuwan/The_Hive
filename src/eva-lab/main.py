import logging
import time
import jax
import jax.numpy as jnp
from fastapi import FastAPI
from shared import get_settings, BaseHealthResponse
from shared.grpc_client import SwarmGRPCClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="THE HIVE - Research Lab (JAX)")
settings = get_settings()

# Client gRPC pour le monitoring haute fréquence
grpc_client = SwarmGRPCClient(host=settings.redis_host.replace("redis", "nervous"), port=9091)

@app.on_event("startup")
async def startup_event():
    logger.info("🔬 Lab (JAX) initialisé")
    logger.info(f"🚀 JAX Devices: {jax.devices()}")
    
    # Test gRPC (optionnel au démarrage)
    if grpc_client.connect():
        logger.info("✅ Nervous gRPC accessible")
    else:
        logger.warning("⚠️ Nervous gRPC non accessible (fallback Redis attendu)")

@app.get("/health", response_model=BaseHealthResponse)
async def health():
    return {
        "status": "operational",
        "version": "1.0.0",
        "details": {
            "jax_backend": jax.lib.xla_bridge.get_backend().platform,
            "device_count": jax.device_count()
        }
    }

@app.get("/lab/test-jax")
async def test_jax():
    """Vérification simple de calcul matriciel sur GPU/TPU via JAX"""
    start = time.time()
    x = jnp.ones((1000, 1000))
    y = jnp.dot(x, x).block_until_ready()
    elapsed = time.time() - start
    
    return {
        "matrix_size": "1000x1000",
        "compute_time_ms": elapsed * 1000,
        "result_sum": float(jnp.sum(y))
    }
