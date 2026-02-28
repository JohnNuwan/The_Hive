import io
import torch
import torchaudio
from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# To prevent importing and downloading Model on global context scope
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info("Initializing AudioCraft Model (MusicGen-Small)...")
    try:
        from audiocraft.models import MusicGen
        model = MusicGen.get_pretrained("facebook/musicgen-small")
        logger.info("✅ MusicGen model loaded on GPU.")
    except Exception as e:
        logger.error(f"Failed to load MusicGen: {e}")
        model = None
    yield
    logger.info("Shutting down AudioCraft API.")

app = FastAPI(title="The Hive - Audio Factory", lifespan=lifespan)

class AudioRequest(BaseModel):
    prompt: str = Field(..., description="Description of the music (ex: 80s synthwave fast beat)")
    duration: int = Field(default=15, ge=1, le=30, description="Duration in seconds (max 30s)")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/generate")
async def generate_audio(request: AudioRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Audio model is not loaded or initializing.")
        
    try:
        logger.info(f"Generating audio for prompt: '{request.prompt}', duration: {request.duration}s")
        model.set_generation_params(duration=request.duration)
        wav = model.generate([request.prompt])  # shape: [B, C, T]
        
        # Take the first matched output and move to CPU
        wav = wav[0].cpu()
        
        # Save to memory buffer
        buffer = io.BytesIO()
        torchaudio.save(buffer, wav, model.sample_rate, format="wav")
        buffer.seek(0)
        
        logger.info("Audio generated successfully.")
        return Response(content=buffer.read(), media_type="audio/wav")
        
    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))
