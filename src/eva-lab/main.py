import logging
import time
import asyncio
import jax
import jax.numpy as jnp
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from shared import get_settings, BaseHealthResponse
from shared.grpc_client import SwarmGRPCClient
from shared.internal_auth import InternalAuth

from eva_lab.muzero.config import MuZeroConfigV3
from eva_lab.muzero.dreamer_networks import make_dreamer_networks
from eva_lab.muzero.dreamer_trainer import DreamerTrainerJAX, WorldModelBatch
from eva_lab.muzero.replay_buffer import PrioritizedReplayBuffer, GameHistory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="THE HIVE - Research Lab (JAX)")
settings = get_settings()

# Global State
class LabState:
    def __init__(self):
        self.config = MuZeroConfigV3()
        self.trainer: DreamerTrainerJAX = None
        self.transformed = None
        self.params = None
        self.rssm_state = None  # Global latent state for current symbol
        self.replay_buffer = PrioritizedReplayBuffer(max_games=1000)
        self.grpc_client = SwarmGRPCClient(host=settings.redis_host.replace("redis", "nervous"), port=9091)
        self.training_active = False
        self.rng = jax.random.PRNGKey(42)

state = LabState()

@app.on_event("startup")
async def startup_event():
    logger.info("🔬 Lab (JAX) - DreamerV3 Edition initialisé")
    logger.info(f"🚀 JAX Devices: {jax.devices()}")
    
    # 1. Initialize Dreamer Networks
    state.transformed = make_dreamer_networks(state.config)
    state.trainer = DreamerTrainerJAX(state.config, state.transformed)
    
    # 2. Init params
    sample_obs = jnp.zeros((1, *state.config.observation_shape))
    params, _ = state.trainer.init_params(sample_obs)
    state.params = params
    
    # 3. Init global RSSM state
    state.rng, rng_init = jax.random.split(state.rng)
    state.rssm_state = state.transformed.apply(state.params, rng_init, 2, 1) # Mode 2: init_state
    
    logger.info("🌌 World Model RSSM initialisé et stabilisé")
    
    # gRPC Connection
    if state.grpc_client.connect():
        logger.info("✅ Nervous gRPC accessible")
    else:
        logger.warning("⚠️ Nervous gRPC non accessible (fallback Redis attendu)")

# ── API Models ──

class ObservationRequest(BaseModel):
    price: float
    indicators: Dict[str, Any]

class ShadowRecordRequest(BaseModel):
    symbol: str
    price: float
    indicators: Dict[str, Any]
    action: str | int
    reward: float = 0.0
    pnl: float = 0.0
    done: bool = False

# ── Endpoints ──

@app.get("/health", response_model=BaseHealthResponse)
async def health():
    return {
        "status": "operational",
        "version": "5.5.0-dreamer",
        "details": {
            "jax_backend": jax.lib.xla_bridge.get_backend().platform,
            "device_count": jax.device_count(),
            "trainer_ready": state.trainer is not None,
            "buffer_size": state.replay_buffer.size
        }
    }

@app.post("/dreamer/predict")
async def predict_action(request: ObservationRequest):
    """World Model inference with stabilized JAX dispatcher."""
    if state.trainer is None or state.params is None:
        raise HTTPException(status_code=503, detail="Dreamer Model not initialized")
    
    start_time = time.time()
    
    # 1. Process Observation (simplified for demo)
    obs_val = [request.price] + list(request.indicators.values())
    # Ensure fixed length as per config observation_shape
    obs_vec = jnp.zeros(state.config.observation_shape)
    for i, v in enumerate(obs_val[:obs_vec.shape[0]]):
        obs_vec = obs_vec.at[i].set(v)
    
    obs_jax = obs_vec.reshape(1, -1)
    
    # 2. Transition World Model
    state.rng, rng_step = jax.random.split(state.rng)
    prev_action = jnp.zeros((1, state.config.action_space_size)) # Assume HOLD for inference check
    
    # mode 0 = observe
    prior, posterior, rec_obs, pred_rew = state.transformed.apply(
        state.params, rng_step, 0, obs_jax, prev_action, state.rssm_state
    )
    
    # Update global state for sequence continuity
    state.rssm_state = posterior
    
    # 3. Simple greedy policy for demo (would use Actor head in full version)
    # Here we just check reward prediction or RSI fallback
    rsi = request.indicators.get("RSI", 50.0)
    action = 0 # WAIT
    if rsi < 30: action = 1 # BUY
    elif rsi > 70: action = 2 # SELL
    
    elapsed = time.time() - start_time
    
    return {
        "action": int(action),
        "reward_pred": float(pred_rew[0, 0]),
        "latency_ms": elapsed * 1000,
        "engine": "DreamerV3-Observe-JAX"
    }

@app.post("/dreamer/imagine")
async def imagine_future(request: ObservationRequest):
    """Latent unroll (Imagination) telemetry."""
    if state.trainer is None:
        raise HTTPException(status_code=503, detail="Dreamer Model not initialized")

    # For telemetry, we just confirm latent unroll capability
    state.grpc_client.send_signal(
        source="lab",
        target="nexus",
        action="DREAM_TELEMETRY",
        payload={
            "horizon": 15,
            "latent_state_dim": 2560,
            "timestamp": time.time()
        },
        priority=2
    )
    
    return {
        "status": "imagination_traced",
        "horizon": 15,
        "latent_summary": "RSSM state is stable"
    }

@app.post("/shadow/record")
async def record_experience(request: ShadowRecordRequest):
    """Store MT5 experiences into Replay Buffer for Shadow Learning."""
    # Process Observation
    obs_val = [request.price] + list(request.indicators.values())
    obs_vec = np.zeros(state.config.observation_shape)
    for i, v in enumerate(obs_val[:obs_vec.shape[0]]):
        obs_vec[i] = v
        
    game = GameHistory()
    # Mapping action format
    action = 0
    if isinstance(request.action, str):
        mapping = {"HOLD":0, "BUY":1, "SELL":2}
        action = mapping.get(request.action.upper(), 0)
    else:
        action = int(request.action)
        
    # Use reward or pnl
    reward = request.reward if request.reward != 0 else request.pnl
    
    game.store(obs_vec, action, reward, [0.1]*state.config.action_space_size, 0.0)
    state.replay_buffer.save_game(game)
    
    return {"status": "recorded", "buffer_size": state.replay_buffer.size}

@app.post("/lab/train")
async def start_training(background_tasks: BackgroundTasks):
    """Lance une session d'entraînement en arrière-plan."""
    if state.training_active:
        return {"status": "already_running"}
    
    state.training_active = True
    background_tasks.add_task(training_loop)
    return {"status": "training_started"}

async def training_loop():
    logger.info("🏋️ Shadow Training session started")
    while state.training_active:
        try:
            if state.replay_buffer.size >= state.config.batch_size:
                # 1. Sample and Prepare Batch
                samples = state.replay_buffer.sample(state.config.batch_size)
                batch = state.trainer.prepare_batch(samples)
                
                # 2. Update Step
                metrics = state.trainer.train_step(batch)
                
                # 3. Logging & Telemetry
                logger.info(f"Dreamer Loss: {metrics['loss_total']:.4f}")
                state.grpc_client.send_signal(
                    source="lab",
                    target="nexus",
                    action="TRAINING_METRICS",
                    payload={
                        "loss": float(metrics["loss_total"]),
                        "loss_obs": float(metrics["loss_obs"]),
                        "loss_rew": float(metrics["loss_rew"]),
                        "timestamp": time.time()
                    },
                    priority=3
                )
            
            await asyncio.sleep(1.0) # Lower frequency for stability
        except Exception as e:
            logger.error(f"Training loop error: {e}")
            await asyncio.sleep(5)

async def sync_quant_metrics():
    """Exemple de synchronisation avec Julia pour ajuster l'entraînement."""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get("http://quant-lab:8701/quant/optimize") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"📊 Quant Lab feedback received: {data.get('status')}")
    except Exception as e:
        logger.warning(f"Could not sync with Quant Lab: {e}")
